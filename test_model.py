"""
Test script for YOLOv10 implementation
"""

import torch
from models.yolov10 import YOLOv10
from utils.general import load_config
from utils.credentials import (
    DEFAULT_CONFIG,
    MODEL_WEIGHTS,
    TRAIN_CONFIG,
    MODEL_CONFIG,
    DEVICE_CONFIG
)

def test_model_forward():
    """Test model forward pass."""
    # Load configuration
    config = load_config(DEFAULT_CONFIG)
    
    # Create model
    model = YOLOv10(
        num_classes=config['model']['num_classes'],
        input_channels=config['model']['input_channels']
    )
    
    # Test with random input
    batch_size = 1
    img_size = TRAIN_CONFIG['WIDTH']
    x = torch.randn(batch_size, TRAIN_CONFIG['CHANNELS'], img_size, img_size)
    
    # Run forward pass
    with torch.no_grad():
        outputs = model(x)
    
    # Print output shapes
    print("\nModel Forward Pass Test:")
    print("-" * 50)
    for i, output in enumerate(outputs):
        print(f"Output {i+1} shape: {output.shape}")
    
    return all(out.shape[0] == batch_size for out in outputs)

def test_model_load():
    """Test model weight loading."""
    if not MODEL_WEIGHTS.exists():
        print(f"\nSkipping weight loading test: {MODEL_WEIGHTS} not found")
        return True
    
    try:
        model = YOLOv10()
        model.load_weights(MODEL_WEIGHTS)
        print("\nModel Weight Loading Test: Passed")
        return True
    except Exception as e:
        print(f"\nModel Weight Loading Test: Failed")
        print(f"Error: {e}")
        return False

def main():
    """Run all tests."""
    print("Running YOLOv10 Tests...")
    
    # Test model forward pass
    forward_pass_ok = test_model_forward()
    print(f"Forward Pass Test: {'Passed' if forward_pass_ok else 'Failed'}")
    
    # Test model weight loading
    weight_loading_ok = test_model_load()
    print(f"Weight Loading Test: {'Passed' if weight_loading_ok else 'Failed'}")
    
    # Overall status
    all_tests_passed = forward_pass_ok and weight_loading_ok
    print("\nOverall Test Status:", "PASSED" if all_tests_passed else "FAILED")

if __name__ == '__main__':
    main()
