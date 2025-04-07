# YOLOv10 Object Detection

A streamlined implementation of the YOLOv10 model for object detection.

## Features

- Multiple model variants (nano, small, base, medium, large, xlarge)
- Simple detection interface
- Variant-specific weight handling
- Support for converting between model formats

## Model Variants

YOLOv10 comes in several sizes to balance speed and accuracy:

| Variant | Suffix | Size     | Description                        |
|---------|--------|----------|------------------------------------|
| Nano    | n      | Smallest | Ultra-fast, lower accuracy         |
| Small   | s      | Small    | Fast, good accuracy                |
| Base    | b      | Medium   | Balanced speed and accuracy        |
| Medium  | m      | Medium+  | Better accuracy than Base          |
| Large   | l      | Large    | High accuracy, slower              |
| XLarge  | x      | Largest  | Highest accuracy, slowest          |

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/yolov10-model.git
cd yolov10-model

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running Detection

```bash
# Run detection with default settings
python detect.py --weights weights/yolov10b.pt --img-path images/example.jpg

# Specify a different variant
python detect.py --weights weights/yolov10s.pt --variant s --img-path images/example.jpg

# Force loading weights even with variant mismatch
python detect.py --weights weights/yolov10l.pt --variant b --force-load --img-path images/example.jpg
```

### Model Conversion

```bash
# Convert a checkpoint to ONNX format
python convert_model.py --weights weights/yolov10b.pt --format onnx

# Convert a model to TorchScript
python convert_model.py --weights weights/yolov10s.pt --variant s --format torchscript
```

### Training Variants

If you need to train YOLOv10 variants, you can use:

```bash
# Train a base variant
python train.py --variant b

# Train using the train_variants script (multiple variants)
python train_variants.py --variants n,s,b --mode train

# Convert weights between variants
python train_variants.py --variants n,s,b --mode convert --source-variant b --source-weights weights/YOLOv10b/best_model.pt
```

## Directory Structure

```
yolov10/
├── models/
│   └── yolov10.py    # YOLOv10 model implementation
├── utils/
│   ├── credentials.py # Model configuration
│   └── model_utils.py # Utility functions
├── data/              # Dataset handling
│   └── dataset.py     # Dataset implementation
├── detect.py          # Detection script
├── train.py           # Training script
├── train_variants.py  # Multi-variant training
└── convert_model.py   # Model conversion
```

## Troubleshooting

Common issues and solutions:

### Variant Mismatch Errors

If you encounter errors related to variant mismatches:

1. Use the `--force-load` option to force weight loading
2. Make sure the specified variant matches your weights file

### Loading Detection Results

If your model loads but doesn't produce expected results:

1. Check that the correct class names are configured
2. Verify that the model weights are for the specified variant
3. Try with a smaller batch size if running out of memory

## License

This project is licensed under the MIT License - see the LICENSE file for details.
