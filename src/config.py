from pathlib import Path
import os

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

_behavior_model_env = os.environ.get("BEHAVIOR_MODEL_PATH")
BEHAVIOR_DETECTOR_MODEL = Path(_behavior_model_env) if _behavior_model_env else (
    PROJECT_ROOT / "data" / "kaggle_outputs" / "cognivision_yolo" / "objects_v1" / "weights" / "best.pt"
)

_classifier_model_env = os.environ.get("CLASSIFIER_MODEL_PATH")
CLASSIFIER_MODEL = Path(_classifier_model_env) if _classifier_model_env else (
    PROJECT_ROOT / "data" / "kaggle_outputs" / "cognivision_outputs" / "efficientnet_attention_best.pth"
)
CLASSIFIER_INPUT_SIZE = 224  # EfficientNet requires 224x224 input
# Explicit class-index mapping used by the loaded classifier checkpoint.
# Index 0 -> ATTENTIVE, Index 1 -> DISTRACTED for current RGB model.
CLASSIFIER_LABELS = ["attentive", "distracted"]
STUDENT_LABELS = ["person", "student", "candidate", "human"]

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
