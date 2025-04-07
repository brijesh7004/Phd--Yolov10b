"""
YOLOv10 Variant Training Script
This script provides functionality to train different YOLOv10 variants 
or to transfer weights between variants.
"""

import sys
import argparse
import subprocess
from pathlib import Path
import torch
import time
from models.yolov10 import YOLOv10
from utils.credentials import MODEL_CONFIG, WEIGHTS_DIR

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='YOLOv10 Variant Training/Conversion')
    parser.add_argument('--variants', type=str, default='b',
                        help='Comma-separated list of variants to train (n,s,b,m,l,x)')
    parser.add_argument('--mode', type=str, choices=['train', 'convert'], default='train',
                        help='Mode: train models or convert between variants')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--source-variant', type=str, default=None,
                        help='Source variant for conversion or training initialization')
    parser.add_argument('--source-weights', type=str, default=None,
                        help='Source weights file path')
    parser.add_argument('--force-load', action='store_true',
                        help='Force loading of weights even with variant mismatch')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode for more verbose output')
    return parser.parse_args()

def setup_variant_dirs():
    """Set up directories for each variant"""
    weights_dir = Path(WEIGHTS_DIR)
    weights_dir.mkdir(exist_ok=True, parents=True)
    
    for variant in ['n', 's', 'b', 'm', 'l', 'x']:
        variant_dir = weights_dir / f"YOLOv10{variant}"
        variant_dir.mkdir(exist_ok=True, parents=True)
    
    return weights_dir

def train_model_variants(variants, epochs=10, batch_size=16, source_variant=None, 
                         source_weights=None, force_load=False, debug=False):
    """Train multiple model variants sequentially."""
    setup_variant_dirs()
    
    for variant in variants:
        print(f"\n{'='*80}")
        print(f"Starting training for YOLOv10{variant}")
        print(f"{'='*80}")
        
        # Build command
        cmd = [sys.executable, 'train.py', f'--variant={variant}', 
               f'--epochs={epochs}', f'--batch-size={batch_size}']
        
        # Add source weights if provided
        if source_weights:
            cmd.append(f'--pretrained-weights={source_weights}')
            
        # Add force load if specified
        if force_load:
            cmd.append('--force-load')
            
        # Add debug flag if specified
        if debug:
            cmd.append('--debug')
        
        # Run training process
        print(f"Running command: {' '.join(cmd)}")
        start_time = time.time()
        
        try:
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output in real-time
            for line in process.stdout:
                print(line, end='')
                
            process.wait()
            
            elapsed_time = time.time() - start_time
            if process.returncode == 0:
                print(f"\nCompleted training for YOLOv10{variant} in {elapsed_time:.1f} seconds")
            else:
                print(f"\nError training YOLOv10{variant}: Command returned non-zero exit status {process.returncode}")
        except Exception as e:
            print(f"Error training YOLOv10{variant}: {str(e)}")
        
        if process.returncode != 0:
            choice = input(f"Continue with next variant? (y/n): ")
            if choice.lower() != 'y':
                break

def transfer_weights(source_variant, source_weights, target_variants, debug=False):
    """Convert weights from source variant to target variants."""
    if not source_weights:
        print("Source weights file not specified")
        return False
    
    source_weights_path = Path(source_weights)
    if not source_weights_path.exists():
        print(f"Source weights file not found: {source_weights_path}")
        return False
    
    setup_variant_dirs()
    print(f"Loading source weights from YOLOv10{source_variant}: {source_weights_path}")
    
    try:
        # Create source model
        source_model = YOLOv10(
            num_classes=len(MODEL_CONFIG['CLASS_NAMES']), 
            variant=source_variant
        )
        
        # Load source weights
        try:
            source_model.load_weights(source_weights_path, force_load=True)
            print("Source weights loaded successfully")
        except Exception as e:
            print(f"Error loading source weights: {e}")
            print("Trying to load as direct state dict...")
            state_dict = torch.load(source_weights_path, map_location='cpu')
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            source_model.load_state_dict(state_dict, strict=False)
            print("Source weights loaded from state dict")
        
        # Process each target variant
        success_count = 0
        for target_variant in target_variants:
            print(f"\nConverting to YOLOv10{target_variant}...")
            
            # Skip if source and target variants are the same
            if target_variant == source_variant:
                print(f"Skipping conversion from {source_variant} to itself")
                continue
            
            # Create target model
            target_model = YOLOv10(
                num_classes=len(MODEL_CONFIG['CLASS_NAMES']), 
                variant=target_variant
            )
            
            # Map parameters from source to target
            source_state_dict = source_model.state_dict()
            target_state_dict = target_model.state_dict()
            
            # Track statistics
            transferred = 0
            mismatch = 0
            
            # Copy matching parameters
            for key in target_state_dict:
                if key in source_state_dict:
                    if target_state_dict[key].shape == source_state_dict[key].shape:
                        target_state_dict[key] = source_state_dict[key]
                        transferred += 1
                    else:
                        if debug:
                            print(f"Shape mismatch for {key}: "
                                  f"source={source_state_dict[key].shape}, "
                                  f"target={target_state_dict[key].shape}")
                        mismatch += 1
            
            # Update model with transferred weights
            target_model.load_state_dict(target_state_dict)
            
            # Save converted model
            weights_dir = Path(WEIGHTS_DIR)
            variant_dir = weights_dir / f"YOLOv10{target_variant}"
            target_path = variant_dir / f"converted_from_{source_variant}.pt"
            
            # Create a complete checkpoint
            checkpoint = {
                'epoch': 0,
                'model_state_dict': target_model.state_dict(),
                'variant': target_variant,
                'source_variant': source_variant,
                'config': MODEL_CONFIG
            }
            
            torch.save(checkpoint, target_path)
            
            print(f"Transferred weights to YOLOv10{target_variant}")
            print(f"  - Parameters transferred: {transferred}")
            print(f"  - Parameters mismatched: {mismatch}")
            print(f"  - Saved to: {target_path}")
            success_count += 1
        
        return success_count > 0
    
    except Exception as e:
        print(f"Error transferring weights: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return False

def main():
    args = parse_args()
    
    # Parse variants
    variants = [v.strip() for v in args.variants.split(',')]
    valid_variants = [v for v in variants if v in ['n', 's', 'b', 'm', 'l', 'x']]
    
    if not valid_variants:
        print("No valid variants specified. Please use n, s, b, m, l, or x.")
        return
    
    print(f"Selected variants: {', '.join(f'YOLOv10{v}' for v in valid_variants)}")
    
    if args.mode == 'train':
        train_model_variants(
            valid_variants, 
            epochs=args.epochs, 
            batch_size=args.batch_size,
            source_variant=args.source_variant,
            source_weights=args.source_weights,
            force_load=args.force_load,
            debug=args.debug
        )
    elif args.mode == 'convert':
        if not args.source_weights or not args.source_variant:
            print("Error: --source-weights and --source-variant are required for convert mode")
            return
            
        if args.source_variant not in ['n', 's', 'b', 'm', 'l', 'x']:
            print(f"Invalid source variant: {args.source_variant}")
            return
            
        # Remove source variant from target variants
        target_variants = [v for v in valid_variants if v != args.source_variant]
        
        if not target_variants:
            print(f"No target variants specified besides source variant {args.source_variant}")
            return
            
        print(f"Converting from YOLOv10{args.source_variant} to: "
              f"{', '.join(f'YOLOv10{v}' for v in target_variants)}")
              
        transfer_weights(
            args.source_variant, 
            args.source_weights, 
            target_variants,
            debug=args.debug
        )

if __name__ == '__main__':
    main() 