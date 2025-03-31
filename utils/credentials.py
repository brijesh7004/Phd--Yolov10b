"""
YOLOv10 Configuration Settings
"""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
WEIGHTS_DIR = BASE_DIR / 'weights'
RESULTS_DIR = BASE_DIR / 'results'

# Ensure directories exist
WEIGHTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    'IMG_SIZE': 640,
    'CONF_THRESHOLD': 0.25,
    'IOU_THRESHOLD': 0.45,
    'ANCHOR_PER_SCALE': 3,
    'NUM_CLASSES': None,  # Will be set based on class names
    'CLASS_NAMES': [],    # Will be loaded from file
    'ANCHORS': [
        # Large objects
        [[116, 90], [156, 198], [373, 326]],
        # Medium objects
        [[30, 61], [62, 45], [59, 119]],
        # Small objects
        [[10, 13], [16, 30], [33, 23]]
    ],
    'STRIDES': [32, 16, 8],  # Strides for each detection scale
}

# Training configuration
TRAIN_CONFIG = {
    'BATCH_SIZE': 16,
    'NUM_WORKERS': 4,
    'EPOCHS': 100,
    'LEARNING_RATE': 0.001,
    'WEIGHT_DECAY': 0.0005,
    'MOMENTUM': 0.937,
    'WARMUP_EPOCHS': 3,
    'SAVE_INTERVAL': 10,
    'EVAL_INTERVAL': 5,
    'PRETRAINED_WEIGHTS': None,  # Path to pretrained weights if any
    'SAVE_DIR': str(WEIGHTS_DIR),
    'RESUME': False,  # Whether to resume from last checkpoint
}

# Data configuration
DATA_CONFIG = {
    'TRAIN_IMG_DIR': str(DATA_DIR / 'images' / 'train'),
    'VAL_IMG_DIR': str(DATA_DIR / 'images' / 'val'),
    'TEST_IMG_DIR': str(DATA_DIR / 'images' / 'test'),
    'TRAIN_LABEL_DIR': str(DATA_DIR / 'labels' / 'train'),
    'VAL_LABEL_DIR': str(DATA_DIR / 'labels' / 'val'),
    'TEST_LABEL_DIR': str(DATA_DIR / 'labels' / 'test'),
}

# Device configuration
DEVICE_CONFIG = {
    'CUDA': True,  # Whether to use CUDA if available
    'NUM_THREADS': 4,  # Number of threads for CPU operations
}

# Model weights path
MODEL_WEIGHTS = WEIGHTS_DIR / 'best_model.pt'

# Class names file
CLASS_NAMES_FILE = DATA_DIR / 'classes.txt'

# Load class names if available
if CLASS_NAMES_FILE.exists():
    with open(CLASS_NAMES_FILE) as f:
        MODEL_CONFIG['CLASS_NAMES'] = [line.strip() for line in f.readlines()]
    MODEL_CONFIG['NUM_CLASSES'] = len(MODEL_CONFIG['CLASS_NAMES'])
