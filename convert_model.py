"""
YOLOv10 Model Conversion Script
This script converts YOLOv10 model weights between different formats.
"""

import torch
import argparse
from pathlib import Path
from models.yolov10 import YOLOv10
from utils.credentials import MODEL_CONFIG
from utils.model_utils import extract_model_from_checkpoint

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='YOLOv10 Model Conversion')
    parser.add_argument('--weights', type=str, required=True,
                        help='path to model weights')
    parser.add_argument('--output', type=str, default=None,
                        help='output path (default: auto-generated based on format)')
    parser.add_argument('--format', type=str, choices=['onnx', 'torchscript', 'pt'], default='pt',
                        help='output format: onnx, torchscript, or pt (pytorch)')
    parser.add_argument('--variant', type=str, default=None,
                        help='model variant: n, s, b, m, l, x (if not specified, will try to detect)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='input image size for ONNX conversion')
    parser.add_argument('--dynamic', action='store_true',
                        help='enable dynamic axes for ONNX conversion')
    return parser.parse_args()

def detect_variant_from_weights(weights_path):
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
    except Exception as e:
        print(f"Error detecting variant: {e}")
    
    return None

def convert_to_onnx(model, output_path, img_size=640, dynamic=False):
    """Convert model to ONNX format."""
    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, img_size, img_size, device='cpu')
    
    # Set export parameters
    opset_version = 11  # ONNX opset version
    
    # Dynamic axes
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    # Export the model
    torch.onnx.export(
        model,               # model being run
        dummy_input,         # model input
        output_path,         # where to save the model
        export_params=True,  # store the trained parameter weights
        opset_version=opset_version,  # ONNX version
        do_constant_folding=True,     # optimization
        input_names=['input'],        # the input name
        output_names=['output'],      # the output name
        dynamic_axes=dynamic_axes     # variable length axes
    )
    
    print(f"ONNX model saved to {output_path}")
    return output_path

def convert_to_torchscript(model, output_path):
    """Convert model to TorchScript format."""
    # Set model to evaluation mode
    model.eval()
    
    # Trace the model
    traced_model = torch.jit.script(model)
    
    # Save the traced model
    torch.jit.save(traced_model, output_path)
    
    print(f"TorchScript model saved to {output_path}")
    return output_path

def main():
    args = parse_args()
    
    # Get weights path
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Weights file not found: {weights_path}")
        return 1
    
    # Detect variant if not specified
    variant = args.variant
    if variant is None:
        detected_variant = detect_variant_from_weights(weights_path)
        if detected_variant:
            print(f"Detected variant: {detected_variant}")
            variant = detected_variant
        else:
            print("Could not detect variant, using default 'b'")
            variant = 'b'
    
    # Create model
    print(f"Creating YOLOv10{variant} model...")
    model = YOLOv10(num_classes=len(MODEL_CONFIG['CLASS_NAMES']), variant=variant)
    
    # Load weights
    try:
        model.load_weights(weights_path, force_load=True)
    except Exception as e:
        print(f"Error loading weights: {e}")
        return 1
    
    # Set model to evaluation mode
    model.eval()
    
    # Determine output path if not specified
    if args.output is None:
        if args.format == 'onnx':
            output_path = weights_path.parent / f"YOLOv10{variant}.onnx"
        elif args.format == 'torchscript':
            output_path = weights_path.parent / f"YOLOv10{variant}.torchscript"
        else:  # PyTorch format
            output_path = weights_path.parent / f"YOLOv10{variant}_converted.pt"
    else:
        output_path = Path(args.output)
    
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert model based on specified format
    if args.format == 'onnx':
        convert_to_onnx(model, output_path, args.img_size, args.dynamic)
    elif args.format == 'torchscript':
        convert_to_torchscript(model, output_path)
    else:  # PyTorch format
        # Extract model weights and save
        torch.save({
            'model_state_dict': model.state_dict(),
            'variant': variant,
            'config': MODEL_CONFIG
        }, output_path)
        print(f"PyTorch model saved to {output_path}")
    
    print(f"Model conversion complete. Output: {output_path}")
    return 0

if __name__ == '__main__':
    main() 