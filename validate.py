"""
YOLOv10 Validation Script
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json

from models.yolov10 import YOLOv10
from utils.credentials import (
    MODEL_CONFIG, DATA_CONFIG,
    MODEL_WEIGHTS, CLASS_NAMES_FILE
)
from data.dataset import YOLODataset

def calculate_ap(recall, precision):
    """Calculate Average Precision using 11-point interpolation."""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11.0
    return ap

def calculate_metrics(pred_boxes, true_boxes, iou_threshold=0.5):
    """Calculate precision, recall, and AP for a set of predictions."""
    true_positives = np.zeros(len(pred_boxes))
    false_positives = np.zeros(len(pred_boxes))
    
    if len(true_boxes) == 0:
        if len(pred_boxes) == 0:
            return 0, 0, 0  # No predictions, no ground truth
        else:
            return 0, 0, 0  # All predictions are false positives
    
    if len(pred_boxes) == 0:
        return 0, 0, 0  # No predictions made
    
    # Sort predictions by confidence
    pred_boxes = sorted(pred_boxes, key=lambda x: x['confidence'], reverse=True)
    
    # Calculate IoU for each prediction with ground truth
    for pred_idx, pred_box in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(true_boxes):
            if gt_box['class_id'] == pred_box['class_id']:  # Same class
                iou = calculate_iou(pred_box['box'], gt_box['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            if not true_boxes[best_gt_idx].get('matched', False):
                true_positives[pred_idx] = 1
                true_boxes[best_gt_idx]['matched'] = True
            else:
                false_positives[pred_idx] = 1
        else:
            false_positives[pred_idx] = 1
    
    # Calculate cumulative values
    cumsum_tp = np.cumsum(true_positives)
    cumsum_fp = np.cumsum(false_positives)
    recall = cumsum_tp / len(true_boxes)
    precision = cumsum_tp / (cumsum_tp + cumsum_fp)
    
    # Calculate AP
    ap = calculate_ap(recall, precision)
    
    return precision[-1], recall[-1], ap

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0

def validate_model(model, val_loader, device, conf_thres=0.25, iou_thres=0.45):
    """Validate model on validation set."""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Validating'):
            images = images.to(device)
            
            # Get predictions
            predictions = model(images)
            batch_predictions = model.process_predictions(
                predictions,
                conf_thres=conf_thres,
                iou_thres=iou_thres
            )
            
            # Store predictions and targets
            all_predictions.extend(batch_predictions)
            all_targets.extend(targets)
    
    # Calculate metrics per class
    class_metrics = {}
    for class_id in range(model.num_classes):
        class_name = MODEL_CONFIG['CLASS_NAMES'][class_id]
        
        # Filter predictions and targets for this class
        class_preds = [p for p in all_predictions if p['class_id'] == class_id]
        class_targets = [t for t in all_targets if t['class_id'] == class_id]
        
        # Calculate metrics
        precision, recall, ap = calculate_metrics(
            class_preds,
            class_targets,
            iou_thres
        )
        
        class_metrics[class_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'AP': float(ap)
        }
    
    # Calculate mAP
    mAP = np.mean([m['AP'] for m in class_metrics.values()])
    
    return class_metrics, mAP

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = YOLOv10(
        num_classes=len(MODEL_CONFIG['CLASS_NAMES']),
        input_channels=3
    ).to(device)
    
    if not MODEL_WEIGHTS.exists():
        raise FileNotFoundError(f"Model weights not found at {MODEL_WEIGHTS}")
    
    print(f"Loading model weights from {MODEL_WEIGHTS}")
    model.load_weights(MODEL_WEIGHTS)
    
    # Create validation dataset and dataloader
    val_dataset = YOLODataset(
        img_dir=Path(DATA_CONFIG['VAL_IMG_DIR']),
        label_dir=Path(DATA_CONFIG['VAL_LABEL_DIR']),
        img_size=MODEL_CONFIG['IMG_SIZE'],
        augment=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=MODEL_CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=MODEL_CONFIG['NUM_WORKERS'],
        collate_fn=val_dataset.collate_fn,
        pin_memory=True
    )
    
    # Validate model
    class_metrics, mAP = validate_model(
        model,
        val_loader,
        device,
        conf_thres=MODEL_CONFIG['CONF_THRESHOLD'],
        iou_thres=MODEL_CONFIG['IOU_THRESHOLD']
    )
    
    # Print results
    print("\nValidation Results:")
    print(f"mAP: {mAP:.4f}")
    print("\nPer-class metrics:")
    for class_name, metrics in class_metrics.items():
        print(f"\n{class_name}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  AP: {metrics['AP']:.4f}")
    
    # Save metrics to file
    metrics_path = Path('results') / 'validation_metrics.json'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_path, 'w') as f:
        json.dump({
            'mAP': float(mAP),
            'class_metrics': class_metrics
        }, f, indent=2)
    
    print(f"\nMetrics saved to {metrics_path}")

if __name__ == '__main__':
    main()
