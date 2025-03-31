"""
YOLOv10 Object Detection Script
"""

import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from models.yolov10 import YOLOv10
from utils.credentials import (
    MODEL_WEIGHTS,
    CLASS_NAMES_FILE,
    MODEL_CONFIG,
    DEVICE_CONFIG
)

class ObjectDetector:
    def __init__(self, weights_path=None, conf_thres=None, iou_thres=None, img_size=None, device=None):
        """Initialize the detector with model and parameters."""
        self.conf_thres = conf_thres or MODEL_CONFIG['CONF_THRESHOLD']
        self.iou_thres = iou_thres or MODEL_CONFIG['IOU_THRESHOLD']
        self.img_size = img_size or MODEL_CONFIG.get('IMG_SIZE', 640)
        self.device = torch.device(device if device else ('cuda' if DEVICE_CONFIG['CUDA'] and torch.cuda.is_available() else 'cpu'))
        
        # Load model
        self.model = self._load_model(weights_path or MODEL_WEIGHTS)
        self.model.eval()
        
        # Load class names
        self.class_names = self._load_classes()

    def _load_model(self, weights_path):
        """Load YOLOv10 model from weights file."""
        try:
            print(f"Loading model from {weights_path}...")
            model = YOLOv10()
            model.load_weights(weights_path)
            model = model.to(self.device)
            print("Model loaded successfully!")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _load_classes(self):
        """Load class names."""
        if CLASS_NAMES_FILE.exists():
            with open(CLASS_NAMES_FILE, 'r') as f:
                return [line.strip() for line in f.readlines()]
        return ['person', 'car', 'dog']  # Default classes if file not found

    def preprocess_image(self, img_path):
        """Preprocess image for inference."""
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get original dimensions
        height, width = img.shape[:2]
        
        # Resize and pad
        r = self.img_size / max(height, width)
        if r != 1:
            img = cv2.resize(img, (int(width * r), int(height * r)),
                           interpolation=cv2.INTER_LINEAR)
        
        new_height, new_width = img.shape[:2]
        dw, dh = self.img_size - new_width, self.img_size - new_height
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)
        
        # Add padding
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Convert to torch tensor
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        img = img.unsqueeze(0).to(self.device)  # Add batch dimension
        img = img / 255.0  # Normalize
        
        return img, (height, width), (r, (left, top))

    def postprocess(self, detections, original_size, scale_factors):
        """Process raw predictions to get final detections."""
        processed_detections = []
        
        if detections is None:
            return processed_detections
            
        # Unpack scale factors
        r, (pad_x, pad_y) = scale_factors
        orig_height, orig_width = original_size
        
        for det in detections:
            # Extract boxes, scores, and class IDs
            boxes = det[..., :4]
            scores = det[..., 4]
            class_ids = det[..., 5:]
            
            # Filter by confidence
            mask = scores > self.conf_thres
            boxes = boxes[mask]
            scores = scores[mask]
            class_ids = class_ids[mask]
            
            if len(boxes) == 0:
                continue
                
            # Get class with highest confidence
            class_ids = torch.argmax(class_ids, dim=1)
            
            # Convert boxes to original image size
            boxes = boxes.cpu().numpy()
            boxes -= np.array([pad_x, pad_y, pad_x, pad_y])  # Remove padding
            boxes /= r  # Remove scaling
            
            # Clip boxes to image bounds
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_width)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_height)
            
            # Add to processed detections
            for box, score, class_id in zip(boxes, scores, class_ids):
                processed_detections.append({
                    'box': box.astype(int),
                    'score': float(score),
                    'class_id': int(class_id),
                    'class_name': self.class_names[int(class_id)]
                })
        
        return processed_detections

    def detect(self, img_path):
        """Run detection on an image."""
        # Preprocess image
        img, original_size, scale_factors = self.preprocess_image(img_path)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(img)
        
        # Process predictions
        detections = self.postprocess(predictions, original_size, scale_factors)
        
        return detections

    def draw_detections(self, img_path, detections, output_path=None):
        """Draw detection boxes on image."""
        # Read image
        img = cv2.imread(str(img_path))
        
        # Draw each detection
        for det in detections:
            box = det['box']
            label = f"{det['class_name']} {det['score']:.2f}"
            
            # Draw box
            color = (0, 255, 0)  # Green color
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            # Draw label
            font_scale = 0.6
            font_thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            # Draw label background
            cv2.rectangle(img, (box[0], box[1] - text_height - 4),
                         (box[0] + text_width, box[1]), color, -1)
            
            # Draw label text
            cv2.putText(img, label, (box[0], box[1] - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                       font_thickness)
        
        # Save or show image
        if output_path:
            cv2.imwrite(str(output_path), img)
        else:
            cv2.imshow('Detections', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='YOLOv10 Object Detection')
    parser.add_argument('--weights', type=str, default=MODEL_WEIGHTS,
                       help='path to model weights')
    parser.add_argument('--img-path', type=str, required=True,
                       help='path to input image')
    parser.add_argument('--output', type=str, default=None,
                       help='path to output image')
    parser.add_argument('--conf-thres', type=float, default=MODEL_CONFIG['CONF_THRESHOLD'],
                       help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=MODEL_CONFIG['IOU_THRESHOLD'],
                       help='NMS IoU threshold')
    parser.add_argument('--img-size', type=int, default=MODEL_CONFIG.get('IMG_SIZE', 640),
                       help='input image size')
    parser.add_argument('--device', type=str, default=('cuda' if DEVICE_CONFIG['CUDA'] and torch.cuda.is_available() else 'cpu'),
                       help='device to run on (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ObjectDetector(
        weights_path=args.weights,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        img_size=args.img_size,
        device=args.device
    )
    
    # Run detection
    detections = detector.detect(args.img_path)
    
    # Draw and save/show results
    detector.draw_detections(args.img_path, detections, args.output)
    
    # Print results
    print("\nDetection Results:")
    for det in detections:
        print(f"Detected {det['class_name']} with confidence {det['score']:.2f}")

if __name__ == '__main__':
    main()
