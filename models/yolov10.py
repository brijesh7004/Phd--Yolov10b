"""
YOLOv10 Model Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.credentials import MODEL_CONFIG

class ConvBlock(nn.Module):
    """Basic convolution block with batch normalization and activation."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()  # SiLU (Swish) activation

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, 1)
        self.conv2 = ConvBlock(channels, channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class CSPBlock(nn.Module):
    """Cross Stage Partial Block."""
    def __init__(self, in_channels, out_channels, num_blocks):
        super().__init__()
        mid_channels = out_channels // 2
        
        self.conv1 = ConvBlock(in_channels, mid_channels, 1)
        self.conv2 = ConvBlock(in_channels, mid_channels, 1)
        self.conv3 = ConvBlock(2 * mid_channels, out_channels, 1)
        
        self.blocks = nn.Sequential(*[ResidualBlock(mid_channels) for _ in range(num_blocks)])

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        return self.conv3(torch.cat([y1, y2], dim=1))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer."""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = ConvBlock(in_channels, mid_channels, 1)
        self.conv2 = ConvBlock(mid_channels * 4, out_channels, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))

class DetectionHead(nn.Module):
    """YOLOv10 Detection Head."""
    def __init__(self, in_channels, num_classes, num_anchors=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors or MODEL_CONFIG['ANCHOR_PER_SCALE']
        
        self.conv = nn.Conv2d(in_channels, self.num_anchors * (5 + num_classes), 1)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Reshape output for each anchor
        x = self.conv(x)
        x = x.view(batch_size, self.num_anchors, 5 + self.num_classes, x.shape[2], x.shape[3])
        x = x.permute(0, 1, 3, 4, 2)  # batch_size, num_anchors, height, width, num_outputs
        
        # Apply activations
        box_xy = torch.sigmoid(x[..., :2])  # center x, y
        box_wh = torch.exp(x[..., 2:4])     # width, height
        obj_conf = torch.sigmoid(x[..., 4:5])  # objectness
        cls_conf = torch.sigmoid(x[..., 5:])  # class probabilities
        
        return torch.cat([box_xy, box_wh, obj_conf, cls_conf], dim=-1)

class YOLOv10(nn.Module):
    """YOLOv10 Model."""
    def __init__(self, num_classes=None, input_channels=3):
        super().__init__()
        
        # Get model configuration
        self.num_classes = num_classes or len(MODEL_CONFIG.get('CLASS_NAMES', ['person']))
        self.anchors = MODEL_CONFIG['ANCHORS']
        self.strides = MODEL_CONFIG['STRIDES']
        
        # Backbone
        self.conv1 = ConvBlock(input_channels, 32, 3, stride=2, padding=1)
        self.conv2 = ConvBlock(32, 64, 3, stride=2, padding=1)
        
        self.csp1 = CSPBlock(64, 128, num_blocks=3)
        self.conv3 = ConvBlock(128, 128, 3, stride=2, padding=1)
        
        self.csp2 = CSPBlock(128, 256, num_blocks=6)
        self.conv4 = ConvBlock(256, 256, 3, stride=2, padding=1)
        
        self.csp3 = CSPBlock(256, 512, num_blocks=9)
        self.conv5 = ConvBlock(512, 512, 3, stride=2, padding=1)
        
        self.csp4 = CSPBlock(512, 1024, num_blocks=3)
        self.sppf = SPPF(1024, 1024)
        
        # Detection heads for different scales
        self.head1 = DetectionHead(1024, self.num_classes)  # Large objects
        self.head2 = DetectionHead(512, self.num_classes)   # Medium objects
        self.head3 = DetectionHead(256, self.num_classes)   # Small objects
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Backbone
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        
        x3 = self.csp1(x2)
        x4 = self.conv3(x3)
        
        x5 = self.csp2(x4)
        x6 = self.conv4(x5)
        
        x7 = self.csp3(x6)
        x8 = self.conv5(x7)
        
        x9 = self.csp4(x8)
        x10 = self.sppf(x9)
        
        # Detection heads
        out1 = self.head1(x10)  # Large objects
        out2 = self.head2(x7)   # Medium objects
        out3 = self.head3(x5)   # Small objects
        
        return [out1, out2, out3]

    def load_weights(self, weights_path):
        """Load model weights from file."""
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            
            # Handle different weight file formats
            if isinstance(state_dict, dict):
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                # If weights are from ultralytics format, convert them
                if any('model.0' in k for k in state_dict.keys()):
                    new_state_dict = {}
                    ultralytics_to_custom = {
                        'model.0': 'conv1',
                        'model.1': 'conv2',
                        'model.2': 'csp1',
                        'model.3': 'conv3',
                        'model.4': 'csp2',
                        'model.5': 'conv4',
                        'model.6': 'csp3',
                        'model.7': 'conv5',
                        'model.8': 'csp4',
                        'model.9': 'sppf',
                        'model.10': 'head1',
                        'model.11': 'head2',
                        'model.12': 'head3'
                    }
                    
                    for k, v in state_dict.items():
                        # Find matching prefix
                        matched = False
                        for old_prefix, new_prefix in ultralytics_to_custom.items():
                            if k.startswith(old_prefix):
                                new_key = k.replace(old_prefix, new_prefix)
                                new_state_dict[new_key] = v
                                matched = True
                                break
                        if not matched and not k.startswith('model'):
                            # Keep any non-model keys as is
                            new_state_dict[k] = v
                    
                    state_dict = new_state_dict
            
            # Load the processed state dict
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys in state dict: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in state dict: {unexpected_keys}")
                
            print("Model weights loaded successfully!")
            return self
            
        except Exception as e:
            print(f"Error loading weights: {str(e)}")
            raise

    def compute_loss(self, predictions, targets):
        """Compute training loss."""
        device = predictions[0].device
        loss_dict = {
            'box_loss': torch.tensor(0., device=device),
            'obj_loss': torch.tensor(0., device=device),
            'cls_loss': torch.tensor(0., device=device)
        }
        
        # Process each scale
        for i, pred in enumerate(predictions):
            # Get anchors for this scale
            anchors = self.anchors[i]
            stride = self.strides[i]
            
            # Get grid size
            grid_size = pred.shape[2:4]
            
            # Create grid
            grid_y, grid_x = torch.meshgrid(torch.arange(grid_size[0], device=device),
                                          torch.arange(grid_size[1], device=device))
            grid = torch.stack((grid_x, grid_y), 2).view(1, 1, grid_size[0], grid_size[1], 2).float()
            
            # Convert anchors to tensor
            anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
            
            # Transform predictions
            pred_xy = (torch.sigmoid(pred[..., :2]) + grid) * stride
            pred_wh = torch.exp(pred[..., 2:4]) * anchors.view(1, -1, 1, 1, 2)
            pred_obj = torch.sigmoid(pred[..., 4:5])
            pred_cls = torch.sigmoid(pred[..., 5:])
            
            # Match predictions with targets
            for target in targets:
                # Convert target boxes to current scale
                target_boxes = target['boxes'] / stride
                target_cls = target['labels']
                
                # Assign targets to anchors
                anchor_indices = self._assign_targets_to_anchors(
                    target_boxes,
                    anchors,
                    grid_size
                )
                
                # Compute losses
                if len(anchor_indices) > 0:
                    matched_gt_boxes = target_boxes[anchor_indices[:, 0]]
                    matched_anchor_idx = anchor_indices[:, 1]
                    matched_grid_idx = anchor_indices[:, 2:4]
                    
                    # Box loss
                    box_loss = self._compute_box_loss(
                        pred_xy[0, matched_anchor_idx, matched_grid_idx[:, 1], matched_grid_idx[:, 0]],
                        pred_wh[0, matched_anchor_idx, matched_grid_idx[:, 1], matched_grid_idx[:, 0]],
                        matched_gt_boxes
                    )
                    
                    # Objectness loss
                    obj_loss = self._compute_obj_loss(
                        pred_obj,
                        anchor_indices,
                        grid_size
                    )
                    
                    # Classification loss
                    cls_loss = self._compute_cls_loss(
                        pred_cls[0, matched_anchor_idx, matched_grid_idx[:, 1], matched_grid_idx[:, 0]],
                        target_cls[anchor_indices[:, 0]]
                    )
                    
                    # Update loss dict
                    loss_dict['box_loss'] += box_loss
                    loss_dict['obj_loss'] += obj_loss
                    loss_dict['cls_loss'] += cls_loss
        
        return loss_dict

    def _assign_targets_to_anchors(self, target_boxes, anchors, grid_size):
        """Assign ground truth boxes to anchors."""
        device = target_boxes.device
        num_anchors = len(anchors)
        
        # Calculate IoU between anchors and targets
        anchor_ious = torch.zeros((len(target_boxes), num_anchors), device=device)
        for i, box in enumerate(target_boxes):
            box_wh = box[2:4] - box[0:2]
            anchor_ious[i] = self._box_iou(box_wh, anchors)
        
        # Assign each target to best matching anchor
        best_anchor_indices = anchor_ious.argmax(dim=1)
        
        # Get grid cell assignment
        grid_xy = (target_boxes[:, :2] + target_boxes[:, 2:4]) / 2  # center point
        grid_xy = torch.clamp(grid_xy, 0, grid_size[0] - 1)
        grid_xy_i = grid_xy.long()
        
        # Combine indices
        indices = torch.cat([
            torch.arange(len(target_boxes), device=device).view(-1, 1),
            best_anchor_indices.view(-1, 1),
            grid_xy_i
        ], dim=1)
        
        return indices

    def _compute_box_loss(self, pred_xy, pred_wh, target_boxes):
        """Compute box regression loss."""
        # Convert target boxes to center format
        target_xy = (target_boxes[:, :2] + target_boxes[:, 2:4]) / 2
        target_wh = target_boxes[:, 2:4] - target_boxes[:, :2]
        
        # Compute loss
        xy_loss = torch.mean((pred_xy - target_xy) ** 2)
        wh_loss = torch.mean((pred_wh - target_wh) ** 2)
        
        return xy_loss + wh_loss

    def _compute_obj_loss(self, pred_obj, anchor_indices, grid_size):
        """Compute objectness loss."""
        device = pred_obj.device
        obj_target = torch.zeros_like(pred_obj)
        
        # Set positive samples
        obj_target[0, anchor_indices[:, 1], anchor_indices[:, 2], anchor_indices[:, 3]] = 1
        
        # Binary cross entropy loss
        obj_loss = nn.BCEWithLogitsLoss()(pred_obj, obj_target)
        
        return obj_loss

    def _compute_cls_loss(self, pred_cls, target_cls):
        """Compute classification loss."""
        # Convert target classes to one-hot
        target_one_hot = torch.zeros_like(pred_cls)
        target_one_hot[torch.arange(len(target_cls)), target_cls] = 1
        
        # Binary cross entropy loss
        cls_loss = nn.BCEWithLogitsLoss()(pred_cls, target_one_hot)
        
        return cls_loss

    def _box_iou(self, box1, box2):
        """Calculate IoU between box1 and box2."""
        # Box areas
        area1 = box1[0] * box1[1]
        area2 = box2[:, 0] * box2[:, 1]
        
        # Find intersection
        inter_w = torch.min(box1[0], box2[:, 0])
        inter_h = torch.min(box1[1], box2[:, 1])
        intersection = inter_w * inter_h
        
        # Calculate IoU
        union = area1 + area2 - intersection
        iou = intersection / union
        
        return iou

    def process_predictions(self, predictions, conf_thres=0.25, iou_thres=0.45):
        """Process raw predictions to get final detections."""
        device = predictions[0].device
        batch_detections = []
        
        # Process each scale
        for i, pred in enumerate(predictions):
            # Get anchors and stride for this scale
            anchors = self.anchors[i]
            stride = self.strides[i]
            
            # Get grid size
            grid_size = pred.shape[2:4]
            
            # Create grid
            grid_y, grid_x = torch.meshgrid(torch.arange(grid_size[0], device=device),
                                          torch.arange(grid_size[1], device=device))
            grid = torch.stack((grid_x, grid_y), 2).view(1, 1, grid_size[0], grid_size[1], 2).float()
            
            # Convert anchors to tensor
            anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
            
            # Transform predictions
            pred_xy = (torch.sigmoid(pred[..., :2]) + grid) * stride
            pred_wh = torch.exp(pred[..., 2:4]) * anchors.view(1, -1, 1, 1, 2)
            pred_obj = torch.sigmoid(pred[..., 4:5])
            pred_cls = torch.sigmoid(pred[..., 5:])
            
            # Reshape predictions
            pred_xy = pred_xy.view(-1, 2)
            pred_wh = pred_wh.view(-1, 2)
            pred_obj = pred_obj.view(-1)
            pred_cls = pred_cls.view(-1, self.num_classes)
            
            # Filter by confidence
            conf_mask = pred_obj > conf_thres
            pred_xy = pred_xy[conf_mask]
            pred_wh = pred_wh[conf_mask]
            pred_obj = pred_obj[conf_mask]
            pred_cls = pred_cls[conf_mask]
            
            if len(pred_xy) == 0:
                continue
            
            # Get class with highest confidence
            class_conf, class_pred = pred_cls.max(1)
            
            # Convert boxes to corners format (xmin, ymin, xmax, ymax)
            pred_boxes = torch.cat([
                pred_xy - pred_wh / 2,
                pred_xy + pred_wh / 2
            ], dim=1)
            
            # Apply NMS
            keep = self._nms(pred_boxes, pred_obj * class_conf, iou_thres)
            
            # Get final detections
            for idx in keep:
                batch_detections.append({
                    'box': pred_boxes[idx].cpu().numpy(),
                    'confidence': float(pred_obj[idx] * class_conf[idx]),
                    'class_id': int(class_pred[idx]),
                    'class_name': MODEL_CONFIG['CLASS_NAMES'][int(class_pred[idx])]
                })
        
        return batch_detections

    def _nms(self, boxes, scores, iou_thres):
        """Apply Non-Maximum Suppression."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        _, order = scores.sort(0, descending=True)
        keep = []
        
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
            i = order[0]
            keep.append(i)
            
            # Compute IoU
            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])
            
            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            ids = (ovr <= iou_thres).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids + 1]
        
        return torch.tensor(keep)
