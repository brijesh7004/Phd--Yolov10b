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
from pathlib import Path

from models.yolov10 import YOLOv10
from utils.credentials import (
    MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG,
    MODEL_WEIGHTS, CLASS_NAMES_FILE
)
from data.dataset import YOLODataset

def train_one_epoch(model, train_loader, optimizer, epoch, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        predictions = model(images)
        
        # Calculate loss
        loss_dict = model.compute_loss(predictions, targets)
        loss = sum(loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            **{k: f'{v.item():.4f}' for k, v in loss_dict.items()}
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
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = YOLOv10(
        num_classes=len(MODEL_CONFIG['CLASS_NAMES']),
        input_channels=3
    ).to(device)
    
    # Load pretrained weights if specified
    if TRAIN_CONFIG['PRETRAINED_WEIGHTS'] and os.path.exists(TRAIN_CONFIG['PRETRAINED_WEIGHTS']):
        print(f"Loading pretrained weights from {TRAIN_CONFIG['PRETRAINED_WEIGHTS']}")
        model.load_weights(TRAIN_CONFIG['PRETRAINED_WEIGHTS'])
    
    # Create datasets and dataloaders
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
        num_workers=TRAIN_CONFIG['NUM_WORKERS'],
        collate_fn=train_dataset.collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=TRAIN_CONFIG['NUM_WORKERS'],
        collate_fn=val_dataset.collate_fn,
        pin_memory=True
    )
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG['LEARNING_RATE'],
        weight_decay=TRAIN_CONFIG['WEIGHT_DECAY']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=TRAIN_CONFIG['LEARNING_RATE'],
        epochs=TRAIN_CONFIG['EPOCHS'],
        steps_per_epoch=len(train_loader)
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(TRAIN_CONFIG['EPOCHS']):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, epoch, device)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        print(f"\nEpoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = Path(TRAIN_CONFIG['SAVE_DIR']) / 'best_model.pt'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'model': MODEL_CONFIG,
                    'train': TRAIN_CONFIG,
                    'data': DATA_CONFIG
                }
            }, save_path)
            print(f"Saved best model to {save_path}")
        
        # Update learning rate
        scheduler.step()
    
    # Save final model
    final_path = Path(TRAIN_CONFIG['SAVE_DIR']) / 'final_model.pt'
    torch.save({
        'epoch': TRAIN_CONFIG['EPOCHS'] - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': {
            'model': MODEL_CONFIG,
            'train': TRAIN_CONFIG,
            'data': DATA_CONFIG
        }
    }, final_path)
    print(f"Saved final model to {final_path}")

if __name__ == '__main__':
    main()
