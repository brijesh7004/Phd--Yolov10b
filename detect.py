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
    MODEL_CONFIG,
    DEVICE_CONFIG
)
import time

class ObjectDetector:
    def __init__(self, weights_path=None, conf_thres=0.25, iou_thres=0.45, img_size=640, 
                 device=None, variant='b', force_load=False, debug=False):
        """Initialize the detector with model and parameters."""
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        self.device = torch.device(device if device else 
                                 ('cuda' if DEVICE_CONFIG['CUDA'] and torch.cuda.is_available() else 'cpu'))
        self.variant = variant
        self.force_load = force_load
        self.debug = debug
        
        # Load model
        self.model = self._load_model(weights_path or MODEL_WEIGHTS)
        self.model.eval()
        
        # Load class names
        self.class_names = self._load_classes()

    def _detect_variant_from_weights(self, weights_path):
        """Try to detect model variant from weights file."""
        try:
            # Load checkpoint
            checkpoint = torch.load(weights_path, map_location='cpu')
            
            # Check if variant is explicitly stored
            if isinstance(checkpoint, dict):
                # Direct variant information
                if 'variant' in checkpoint:
                    return checkpoint['variant']
                    
                # Nested variant information
                if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
                    if 'variant' in checkpoint['config']:
                        return checkpoint['config']['variant']
            
            # If no explicit variant info, try to detect based on first layer
            try:
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Get input channels of first conv
                # Try different possible key patterns
                first_layer_key = None
                for key in state_dict.keys():
                    if 'conv1.conv.weight' in key:
                        first_layer_key = key
                        break
                        
                if not first_layer_key:
                    for key in state_dict.keys():
                        if '.weight' in key and '.conv' in key:
                            first_layer_key = key
                            break
                
                if first_layer_key:
                    first_layer = state_dict[first_layer_key]
                    output_channels = first_layer.shape[0]
                    
                    # Map output channels to variant (approximate)
                    if output_channels <= 8:
                        return 'n'  # nano
                    elif output_channels <= 16:
                        return 's'  # small
                    elif output_channels <= 24:
                        return 'm'  # medium
                    elif output_channels <= 32:
                        return 'b'  # base
                    elif output_channels <= 48:
                        return 'l'  # large
                    else:
                        return 'x'  # xlarge
            except Exception as e:
                if self.debug:
                    print(f"Error during weight inspection: {e}")
                pass
                
        except Exception as e:
            if self.debug:
                print(f"Error detecting variant from weights: {e}")
            pass
            
        return None

    def _load_model(self, weights_path):
        """Load YOLOv10 model from weights file."""
        try:
            weights_path = Path(weights_path)
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights file not found: {weights_path}")
                
            # Handle weights path
            if weights_path.is_dir():
                # Try variant-specific best model first
                variant_best = weights_path / f"YOLOv10{self.variant}" / "best_model.pt"
                if variant_best.exists():
                    weights_path = variant_best
                    print(f"Using weights from: {weights_path}")
                else:
                    # Fall back to any model file
                    model_files = list(weights_path.glob("**/*.pt"))
                    if model_files:
                        weights_path = model_files[0]
                        print(f"Using weights from: {weights_path}")
                    else:
                        raise FileNotFoundError(f"No model weights found in {weights_path}")
            
            # Try to detect variant from weights
            detected_variant = self._detect_variant_from_weights(weights_path)
            if detected_variant and detected_variant != self.variant:
                print(f"Detected variant '{detected_variant}' in weights, but using '{self.variant}'")
                if not self.force_load:
                    print(f"Use --variant {detected_variant} or --force-load to continue")
                    self.variant = detected_variant
            
            print(f"Loading YOLOv10{self.variant} model...")
            model = YOLOv10(
                num_classes=len(MODEL_CONFIG['CLASS_NAMES']),
                input_channels=3,
                variant=self.variant
            )
            
            # Load weights with proper error handling
            try:
                model.load_weights(weights_path, force_load=self.force_load)
                model = model.to(self.device)
                print(f"YOLOv10{self.variant} model loaded successfully!")
            except Exception as e:
                if "Variant mismatch" in str(e) or "Size mismatch" in str(e):
                    if detected_variant:
                        print(f"Detected model variant: {detected_variant}")
                        print(f"Try using: --variant {detected_variant}")
                    else:
                        print(f"Try using --force-load or a different variant (n,s,b,m,l,x)")
                    
                    # If debug is enabled, show more details
                    if self.debug:
                        print(f"Detailed error: {e}")
                raise ValueError(f"Failed to load weights: {e}")
            
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _load_classes(self):
        """Load class names."""
        return MODEL_CONFIG.get('CLASS_NAMES', ['person', 'car', 'dog'])  # Default classes

    def preprocess_image(self, img_path):
        """Preprocess image for inference."""
        # Read image
        img_path = str(img_path)
        if img_path.startswith(('http://', 'https://')):
            # Handle URL input
            import urllib.request
            try:
                with urllib.request.urlopen(img_path) as response:
                    img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            except Exception as e:
                raise ValueError(f"Failed to load image from URL: {e}")
        else:
            # Handle local file
            img = cv2.imread(img_path)
            
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Store original image for later
        self.original_img = img.copy()
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize and pad
        height, width = img.shape[:2]
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

    # Find the detect method in your detect.py file and replace it with this:
    def detect(self, img_path):
        """Run detection on an image."""
        # Preprocess image
        img, original_size, scale_factors = self.preprocess_image(img_path)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(img)
            detections = self.model.process_predictions(predictions, self.conf_thres, self.iou_thres)
        
        # Scale boxes back to original image size
        scaled_detections = []
        for det in detections:
            # Make a copy to avoid modifying the original
            new_det = det.copy()
            
            # Get box as float array
            box = det['box'].copy().astype(float)  # Convert to float first!
            r, (pad_x, pad_y) = scale_factors
            
            # Remove padding - use float arrays for all operations
            pad_array = np.array([pad_x, pad_y, pad_x, pad_y], dtype=float)
            box = box - pad_array
            
            # Rescale to original size
            box = box / r
            
            # Clip to image bounds
            orig_height, orig_width = original_size
            box[0] = np.clip(box[0], 0, orig_width)
            box[1] = np.clip(box[1], 0, orig_height)
            box[2] = np.clip(box[2], 0, orig_width)
            box[3] = np.clip(box[3], 0, orig_height)
            
            # Update detection with integer box
            new_det['box'] = box.astype(np.int32)
            scaled_detections.append(new_det)
        
        return scaled_detections

    def draw_detections(self, img_path, detections, output_path=None, draw_confidence=True):
        """Draw detection boxes on image."""
        # Use original image if available
        if hasattr(self, 'original_img') and self.original_img is not None:
            img = self.original_img.copy()
        else:
            # Read image if original not stored
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
                
        # Create a default output path if not provided
        if output_path is None:
            img_path_obj = Path(img_path)
            output_path = img_path_obj.parent / f"{img_path_obj.stem}_detected{img_path_obj.suffix}"
        
        # Color palette for different classes
        colors = [
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red (BGR)
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
            (128, 0, 0),    # Dark blue
            (0, 128, 0),    # Dark green
            (0, 0, 128),    # Dark red
            (128, 128, 0),  # Dark cyan
        ]
        
        # Draw each detection
        for i, det in enumerate(detections):
            box = det['box']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Generate box color based on class
            class_idx = self.class_names.index(class_name) if class_name in self.class_names else i
            color = colors[class_idx % len(colors)]
            
            # Create label
            if draw_confidence:
                label = f"{class_name} {confidence:.2f}"
            else:
                label = class_name
            
            # Draw box with a thickness relative to image size
            thickness = max(1, int(min(img.shape[0], img.shape[1]) / 500))
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, thickness)
            
            # Draw label
            font_scale = 0.6 * thickness
            font_thickness = max(1, thickness - 1)
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            # Draw label background
            cv2.rectangle(img, (box[0], box[1] - text_height - 4),
                         (box[0] + text_width, box[1]), color, -1)
            
            # Draw label text
            cv2.putText(img, label, (box[0], box[1] - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                       font_thickness)
        
        # Save output image
        cv2.imwrite(str(output_path), img)
        print(f"Detection image saved to: {output_path}")
        return output_path

def try_all_variants(args):
    """Try detection with all available variants and report results."""
    results = []
    all_variants = ['n', 's', 'b', 'm', 'l', 'x']
    
    print(f"Testing detection with all variants using weights: {args.weights}")
    
    for variant in all_variants:
        print(f"\n{'-'*40}")
        print(f"Trying variant: YOLOv10{variant}")
        print(f"{'-'*40}")
        
        try:
            # Create detector with this variant
            detector = ObjectDetector(
                weights_path=args.weights,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                img_size=args.img_size,
                device=args.device,
                variant=variant,
                force_load=True,  # Force loading to try all variants
                debug=args.debug
            )
            
            # Run detection
            start_time = time.time()
            detections = detector.detect(args.img_path)
            end_time = time.time()
            
            # Add result
            results.append({
                'variant': variant,
                'success': True,
                'detections': len(detections),
                'time': end_time - start_time
            })
            
            # Generate output image with variant in name
            output_path = None
            if args.output:
                # Insert variant in output filename
                path = Path(args.output)
                output_path = path.parent / f"{path.stem}_{variant}{path.suffix}"
            
            # Draw detections for this variant
            detector.draw_detections(args.img_path, detections, output_path)
            
            print(f"Variant {variant}: SUCCESS - Found {len(detections)} objects in {results[-1]['time']:.2f}s")
            
        except Exception as e:
            print(f"Variant {variant}: FAILED - {str(e)}")
            results.append({
                'variant': variant,
                'success': False,
                'error': str(e)
            })
    
    # Print summary of results
    print("\n" + "="*50)
    print("DETECTION SUMMARY FOR ALL VARIANTS")
    print("="*50)
    
    success_count = 0
    best_variant = None
    max_detections = 0
    
    for result in results:
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        details = f"Found {result['detections']} objects in {result['time']:.2f}s" if result['success'] else result['error']
        print(f"YOLOv10{result['variant']}: {status} - {details}")
        
        if result['success']:
            success_count += 1
            if best_variant is None or result['detections'] > max_detections:
                best_variant = result['variant']
                max_detections = result['detections']
    
    print(f"\nSuccessful variants: {success_count}/{len(all_variants)}")
    
    if best_variant:
        print(f"\nRecommended variant for these weights: YOLOv10{best_variant}")
        print(f"Run with: --variant {best_variant}")
    
    return 0

def main():
    parser = argparse.ArgumentParser(description='YOLOv10 Object Detection')
    parser.add_argument('--weights', type=str, default=MODEL_WEIGHTS,
                       help='path to model weights')
    parser.add_argument('--img-path', type=str, required=True,
                       help='path to input image')
    parser.add_argument('--output', type=str, default=None,
                       help='path to output image')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                       help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                       help='NMS IoU threshold')
    parser.add_argument('--img-size', type=int, default=640,
                       help='input image size')
    parser.add_argument('--device', type=str, default=None,
                       help='device to run on (cuda/cpu)')
    parser.add_argument('--variant', type=str, default='b', choices=['n', 's', 'b', 'm', 'l', 'x', 'all'],
                       help='Model variant: n(ano), s(mall), b(ase), m(edium), l(arge), x(large), or all')
    parser.add_argument('--force-load', action='store_true',
                       help='Force loading weights even if variant does not match')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--auto-variant', action='store_true',
                       help='Automatically detect variant from weights')
    parser.add_argument('--hide-confidence', action='store_true', 
                       help='Hide confidence scores in output image')
    
    args = parser.parse_args()
    
    try:
        # Try all variants if "all" is specified
        if args.variant == 'all':
            return try_all_variants(args)
        
        # Auto-detect variant if requested
        if args.auto_variant:
            print("Auto-detecting variant from weights...")
            temp_detector = ObjectDetector(
                weights_path=args.weights,
                debug=True,
                force_load=False
            )
            detected_variant = temp_detector._detect_variant_from_weights(args.weights)
            if detected_variant:
                print(f"Detected variant: {detected_variant}")
                args.variant = detected_variant
            else:
                print("Could not detect variant, using specified variant.")
        
        # Initialize detector
        detector = ObjectDetector(
            weights_path=args.weights,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            img_size=args.img_size,
            device=args.device,
            variant=args.variant,
            force_load=args.force_load,
            debug=args.debug
        )
        
        # Run detection
        detections = detector.detect(args.img_path)
        
        # Draw and save results
        draw_confidence = not args.hide_confidence
        output_path = detector.draw_detections(args.img_path, detections, args.output, draw_confidence)
        
        # Print results
        print("\nDetection Results:")
        for i, det in enumerate(detections):
            print(f"{i+1}. Detected {det['class_name']} with confidence {det['confidence']:.2f}")
        
        if not detections:
            print("No objects detected.")
        
        return 0
    
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        
    except ValueError as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Error during detection: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        
    return 1

if __name__ == '__main__':
    main()
