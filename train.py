"""
YOLOv10 Training Script
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
from pathlib import Path

from models.yolov10 import YOLOv10
from utils.credentials import (
    MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG,
    MODEL_WEIGHTS
)
from data.dataset import YOLODataset
from utils.model_utils import check_model_params, diagnose_nan_loss

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='YOLOv10 Training')
    parser.add_argument('--variant', type=str, default='b', choices=['n', 's', 'b', 'm', 'l', 'x'],
                        help='Model variant: n(ano), s(mall), b(ase), m(edium), l(arge), x(large)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size from config')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs from config')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Override learning rate from config')
    parser.add_argument('--pretrained-weights', type=str, default=None,
                        help='Override pretrained weights path from config')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--force-load', action='store_true',
                        help='Force loading of weights even with variant mismatch')
    parser.add_argument('--debug', action='store_true',
                        help='Enable more verbose debug output')
    return parser.parse_args()

def train_one_epoch(model, train_loader, optimizer, epoch, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    nan_loss_count = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        predictions = model(images)
        
        # Calculate loss
        loss_dict = model.compute_loss(predictions, targets)
        loss = sum(loss_dict.values())
        
        # Check for NaN loss and diagnose if detected
        if torch.isnan(loss):
            nan_loss_count += 1
            if nan_loss_count <= 3:  # Only show detailed diagnosis for first few occurrences
                diagnosis = diagnose_nan_loss(loss_dict, images, targets, model)
                print(f"\nWARNING: NaN loss detected at batch {batch_idx}.\nDiagnosis:\n{diagnosis}\n")
            else:
                print(f"WARNING: NaN loss detected at batch {batch_idx}. Skipping...")
            
            if nan_loss_count > 10:
                print("Too many NaN losses. Consider reducing learning rate or checking the data.")
            
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        
        optimizer.step()
        
        # Update progress bar
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            **{k: f'{v.item():.4f}' for k, v in loss_dict.items() if not torch.isnan(v)}
        })
    
    return avg_loss

def validate(model, val_loader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Validating'):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            predictions = model(images)
            
            # Calculate loss
            loss_dict = model.compute_loss(predictions, targets)
            loss = sum(loss_dict.values())
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)
    
    # Update config with command line arguments if provided
    if args.batch_size:
        TRAIN_CONFIG['BATCH_SIZE'] = args.batch_size
        print(f"Using batch size: {TRAIN_CONFIG['BATCH_SIZE']}")
    
    if args.epochs:
        TRAIN_CONFIG['EPOCHS'] = args.epochs
        print(f"Using epochs: {TRAIN_CONFIG['EPOCHS']}")
    
    if args.learning_rate:
        TRAIN_CONFIG['LEARNING_RATE'] = args.learning_rate
        print(f"Using learning rate: {TRAIN_CONFIG['LEARNING_RATE']}")
    
    if args.pretrained_weights:
        TRAIN_CONFIG['PRETRAINED_WEIGHTS'] = args.pretrained_weights
        print(f"Using pretrained weights: {TRAIN_CONFIG['PRETRAINED_WEIGHTS']}")
    
    # Create model with specified variant
    model = YOLOv10(
        num_classes=len(MODEL_CONFIG['CLASS_NAMES']),
        input_channels=3,
        variant=args.variant
    ).to(device)
    
    # Check model parameters for potential issues
    check_model_params(model, f"YOLOv10{args.variant}")
    
    # Set model save directory based on variant
    variant_save_dir = Path(TRAIN_CONFIG['SAVE_DIR']) / f"YOLOv10{args.variant}"
    variant_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load pretrained weights if specified
    if TRAIN_CONFIG['PRETRAINED_WEIGHTS'] and os.path.exists(TRAIN_CONFIG['PRETRAINED_WEIGHTS']):
        print(f"Loading pretrained weights from {TRAIN_CONFIG['PRETRAINED_WEIGHTS']}")
        try:
            model.load_weights(TRAIN_CONFIG['PRETRAINED_WEIGHTS'], force_load=args.force_load)
        except Exception as e:
            print(f"Error loading weights: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            if not args.force_load:
                print("Try using --force-load to load weights with mismatched shapes")
        
        # Verify loaded weights
        check_model_params(model, f"YOLOv10{args.variant} (after loading weights)")
    
    # Create datasets and dataloaders
    # Use real dataset
    train_dataset = YOLODataset(
        img_dir=Path(DATA_CONFIG['TRAIN_IMG_DIR']),
        label_dir=Path(DATA_CONFIG['TRAIN_LABEL_DIR']),
        img_size=MODEL_CONFIG['IMG_SIZE'],
        augment=True
    )
    
    val_dataset = YOLODataset(
        img_dir=Path(DATA_CONFIG['VAL_IMG_DIR']),
        label_dir=Path(DATA_CONFIG['VAL_LABEL_DIR']),
        img_size=MODEL_CONFIG['IMG_SIZE'],
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG['BATCH_SIZE'],
        shuffle=True,
        num_workers=MODEL_CONFIG['NUM_WORKERS'],
        collate_fn=train_dataset.collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=MODEL_CONFIG['NUM_WORKERS'],
        collate_fn=val_dataset.collate_fn,
        pin_memory=True
    )
    
    # Create optimizer with smaller initial learning rate
    initial_lr = TRAIN_CONFIG['LEARNING_RATE'] * 0.1  # Start with smaller learning rate
    optimizer = optim.Adam(
        model.parameters(),
        lr=initial_lr,
        weight_decay=TRAIN_CONFIG['WEIGHT_DECAY']
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(current_step):
        warmup_steps = TRAIN_CONFIG['WARMUP_EPOCHS'] * len(train_loader)
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps)) * 10.0  # Gradual increase to full LR
        else:
            # Cosine annealing
            progress = float(current_step - warmup_steps) / float(max(1, TRAIN_CONFIG['EPOCHS'] * len(train_loader) - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Initialize EMA model for more stable evaluation
    ema_model = None
    if hasattr(torch, 'optim') and hasattr(torch.optim, 'swa_utils') and hasattr(torch.optim.swa_utils, 'AveragedModel'):
        # If PyTorch version supports it, use EMA
        ema_model = torch.optim.swa_utils.AveragedModel(model)
        print("Using EMA model for evaluation")
    
    # Resume from checkpoint if requested
    start_epoch = 0
    if args.resume:
        latest_checkpoint = None
        checkpoint_pattern = f"checkpoint_epoch_*.pt"
        checkpoints = list(variant_save_dir.glob(checkpoint_pattern))
        
        if checkpoints:
            # Find latest checkpoint
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
            print(f"Resuming from checkpoint: {latest_checkpoint}")
            
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(start_epoch, TRAIN_CONFIG['EPOCHS']):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, epoch, device)
        
        # Update EMA model
        if ema_model is not None:
            ema_model.update_parameters(model)
            eval_model = ema_model.module
        else:
            eval_model = model
            
        # Validate
        val_loss = validate(eval_model, val_loader, device)
        
        print(f"\nEpoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = variant_save_dir / 'best_model.pt'
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'model': MODEL_CONFIG,
                    'train': TRAIN_CONFIG,
                    'data': DATA_CONFIG,
                    'variant': args.variant
                }
            }, save_path)
            print(f"Saved best model to {save_path}")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint every SAVE_INTERVAL epochs
        if epoch % TRAIN_CONFIG['SAVE_INTERVAL'] == 0:
            checkpoint_path = variant_save_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'variant': args.variant
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = variant_save_dir / 'final_model.pt'
    torch.save({
        'epoch': TRAIN_CONFIG['EPOCHS'] - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': {
            'model': MODEL_CONFIG,
            'train': TRAIN_CONFIG,
            'data': DATA_CONFIG,
            'variant': args.variant
        }
    }, final_path)
    print(f"Saved final model to {final_path}")

if __name__ == '__main__':
    main()
