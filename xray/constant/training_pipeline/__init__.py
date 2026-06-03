"""
Training pipeline constants and hyperparameters.

All configuration values used across the training pipeline are centralized here
for easy tuning and reproducibility.
"""

from datetime import datetime
from typing import Dict, List

import torch

# ---------------------------------------------------------------------------
# Timestamps & Paths
# ---------------------------------------------------------------------------
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

ARTIFACT_DIR: str = "artifacts"

# ---------------------------------------------------------------------------
# Class Labels
# ---------------------------------------------------------------------------
CLASS_LABEL_1: str = "NORMAL"

CLASS_LABEL_2: str = "PNEUMONIA"

PREDICTION_LABEL: Dict[int, str] = {0: CLASS_LABEL_1, 1: CLASS_LABEL_2}

# ---------------------------------------------------------------------------
# Data Transformation — Augmentation
# ---------------------------------------------------------------------------
BRIGHTNESS: float = 0.15

CONTRAST: float = 0.15

SATURATION: float = 0.10

HUE: float = 0.05

RESIZE: int = 256

CENTERCROP: int = 224

RANDOMROTATION: int = 15

# ImageNet normalization (standard for transfer learning)
NORMALIZE_LIST_1: List[float] = [0.485, 0.456, 0.406]

NORMALIZE_LIST_2: List[float] = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS_KEY: str = "xray_train_transforms"

TRAIN_TRANSFORMS_FILE: str = "train_transforms.pkl"

TEST_TRANSFORMS_FILE: str = "test_transforms.pkl"

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 32

SHUFFLE: bool = True

PIN_MEMORY: bool = True

NUM_WORKERS: int = 0

VALIDATION_SPLIT: float = 0.2

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42

# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------
TRAINED_MODEL_DIR: str = "trained_model"

TRAINED_MODEL_NAME: str = "model.pt"

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH: int = 25

STEP_SIZE: int = 7

GAMMA: float = 0.5

WEIGHT_DECAY: float = 1e-4

LEARNING_RATE: float = 0.0003

MOMENTUM: float = 0.9

EARLY_STOPPING_PATIENCE: int = 25

GRADIENT_CLIP_VALUE: float = 1.0

# ---------------------------------------------------------------------------
# Mixed Precision Training (AMP)
# ---------------------------------------------------------------------------
USE_AMP: bool = True

# ---------------------------------------------------------------------------
# Transfer Learning — Backbone Freezing
# ---------------------------------------------------------------------------
FREEZE_BACKBONE_EPOCHS: int = 3

# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------
NUM_CLASSES: int = 2

DROPOUT_RATE: float = 0.4


