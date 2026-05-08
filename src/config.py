import os
from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent

# Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"

# Model Configuration
DETECTOR_MODEL = "yolov8n.pt"
CLASSIFIER_MODEL = "efficientnet_attention_best.pth"  # Kaggle-trained EfficientNet-B0 (85.88% test acc)
CLASSIFIER_INPUT_SIZE = 224  # EfficientNet requires 224x224 input

# Training Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Device
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
