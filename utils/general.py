"""
General utility functions for YOLOv10
"""

import yaml
import torch
import numpy as np
from pathlib import Path

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def set_device(device='cuda'):
    """Set the device for computation."""
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = 'cpu'
    return torch.device(device)

def box_iou(box1, box2):
    """Calculate IoU between box1 and box2."""
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])
    
    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    
    # Get the coordinates of intersecting rectangles
    inter_coords = torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])
    inter_coords = torch.clamp(inter_coords, min=0)
    inter_area = inter_coords.prod(2)
    
    # IoU = intersection area / union area
    union_area = area1[:, None] + area2 - inter_area
    return inter_area / (union_area + 1e-6)

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    """Perform Non-Maximum Suppression (NMS) on inference results."""
    bs = prediction.shape[0]  # batch size
    max_nms = 30000  # maximum number of boxes into NMS
    
    # Confidence thresholding
    conf_mask = prediction[..., 4] > conf_thres
    
    output = [None] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[conf_mask[xi]]  # confidence
        
        if not x.shape[0]:  # no boxes
            continue
            
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = x[:, :4].clone()
        box[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
        box[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
        box[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
        box[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
        
        # Get score and class with highest confidence
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        
        # Filter by class
        if x.shape[0] > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        
        # Apply NMS
        boxes, scores = x[:, :4], x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
            
        output[xi] = x[i]
        
    return output

def scale_coords(img1_shape, coords, img0_shape):
    """Rescale coords (xyxy) from img1_shape to img0_shape."""
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, img_shape):
    """Clip bounding xyxy bounding boxes to image shape (height, width)."""
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """Increment file or directory path.
    
    If path exists, add a numeric suffix until a unique path is found.
    """
    path = Path(path)
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"
            if not Path(p).exists():
                break
        path = Path(p)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path
