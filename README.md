# YOLOv10 Implementation

This is an implementation of YOLOv10 (You Only Look Once version 10) object detection model. The implementation includes model architecture, training utilities, and inference code.

## Project Structure

```
yolov10/
├── assets/           # Example images and test assets
├── configs/          # Configuration files
├── data/            # Dataset and data loading utilities
├── models/          # Model architecture implementation
├── utils/           # Utility functions and helpers
├── weights/         # Model weights
├── detect.py        # Object detection script
├── test_model.py    # Model testing script
├── train.py         # Training script
├── validate.py      # Validation script
└── README.md        # This file
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV
- NumPy
- PyYAML
- Ultralytics (for weight compatibility)

## Installation

1. Clone this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # OR
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   pip install torch torchvision opencv-python numpy pyyaml ultralytics
   ```

## Usage

### Object Detection

To run object detection on an image:

```bash
python detect.py --weights weights/yolov10m.pt --img-path path/to/image.jpg
```

Optional arguments:
- `--output`: Path to save the output image
- `--conf-thres`: Confidence threshold (default: from config)
- `--iou-thres`: NMS IoU threshold (default: from config)
- `--img-size`: Input image size (default: from config)
- `--device`: Device to run on (cuda/cpu, default: from config)

### Testing

To run model tests:

```bash
python test_model.py
```

### Training

To train the model:

```bash
python train.py
```

### Validation

To validate the model:

```bash
python validate.py --weights weights/best_model.pt
```

## Model Architecture

The YOLOv10 model consists of:

1. **Backbone**:
   - CSP (Cross Stage Partial) blocks
   - Residual connections
   - SPPF (Spatial Pyramid Pooling - Fast)

2. **Detection Heads**:
   - Multi-scale detection
   - Three output scales for different object sizes

3. **Features**:
   - Enhanced feature extraction
   - Improved small object detection
   - Efficient architecture design

## Training Script (train.py)

The training script handles the complete training pipeline for YOLOv10. Key features:

- Custom dataset loading
- Multi-GPU support
- Learning rate scheduling
- Model checkpointing
- Loss computation
- Data augmentation

### Usage
```bash
python train.py
```

### Configuration
Training parameters can be configured in `utils/credentials.py` under `TRAIN_CONFIG`:
- Batch size
- Number of epochs
- Learning rate
- Weight decay
- Momentum
- Warmup epochs
- Save interval

## Validation Script (validate.py)

The validation script evaluates model performance on the validation set. Features:

- Precision/Recall calculation
- Average Precision (AP) metrics
- Confusion matrix generation
- Results saving

### Usage
```bash
python validate.py --weights weights/best_model.pt
```

### Configuration
Validation parameters can be configured in `utils/credentials.py` under `MODEL_CONFIG`:
- Confidence threshold
- IoU threshold
- Image size
- Number of classes

## Configuration

Model and training parameters are configured in `utils/credentials.py`. Key configurations include:

- Model parameters (number of classes, anchors, strides)
- Training parameters (batch size, learning rate, etc.)
- Data paths and directories
- Detection thresholds

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This implementation is inspired by the original YOLO series and incorporates modern deep learning practices for object detection.
