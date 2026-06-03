"""
Configuration entities for each pipeline stage.

These classes encapsulate all hyperparameters and paths required by each
component, making the pipeline configurable and testable.
"""

import os

from torch import device

from xray.constant.training_pipeline import (
    ARTIFACT_DIR,
    BATCH_SIZE,
    BRIGHTNESS,
    CENTERCROP,
    CONTRAST,
    DEVICE,
    DROPOUT_RATE,
    EARLY_STOPPING_PATIENCE,
    EPOCH,
    FREEZE_BACKBONE_EPOCHS,
    GAMMA,
    GRADIENT_CLIP_VALUE,
    HUE,
    LEARNING_RATE,
    MOMENTUM,
    NORMALIZE_LIST_1,
    NORMALIZE_LIST_2,
    NUM_CLASSES,
    NUM_WORKERS,
    PIN_MEMORY,
    RANDOMROTATION,
    RESIZE,
    SATURATION,
    SHUFFLE,
    STEP_SIZE,
    TEST_TRANSFORMS_FILE,
    TIMESTAMP,
    TRAIN_TRANSFORMS_FILE,
    TRAIN_TRANSFORMS_KEY,
    TRAINED_MODEL_NAME,
    USE_AMP,
    VALIDATION_SPLIT,
    WEIGHT_DECAY,
)


class DataIngestionConfig:
    """Configuration for the data ingestion stage."""

    def __init__(self) -> None:
        self.artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
        self.data_path: str = os.path.join(os.getcwd(), "data")
        self.train_data_path: str = os.path.join(self.data_path, "train")
        self.test_data_path: str = os.path.join(self.data_path, "test")


class DataTransformationConfig:
    """Configuration for the data transformation stage (augmentation, normalization)."""

    def __init__(self) -> None:
        self.color_jitter_transforms: dict = {
            "brightness": BRIGHTNESS,
            "contrast": CONTRAST,
            "saturation": SATURATION,
            "hue": HUE,
        }

        self.RESIZE: int = RESIZE
        self.CENTERCROP: int = CENTERCROP
        self.RANDOMROTATION: int = RANDOMROTATION

        self.normalize_transforms: dict = {
            "mean": NORMALIZE_LIST_1,
            "std": NORMALIZE_LIST_2,
        }

        # Separate configs for train vs test/val loaders
        self.train_loader_params: dict = {
            "batch_size": BATCH_SIZE,
            "shuffle": False,  # Disabled: using WeightedRandomSampler instead
            "pin_memory": PIN_MEMORY,
            "num_workers": NUM_WORKERS,
            "drop_last": True,
        }

        self.test_loader_params: dict = {
            "batch_size": BATCH_SIZE,
            "shuffle": False,
            "pin_memory": PIN_MEMORY,
            "num_workers": NUM_WORKERS,
            "drop_last": False,
        }

        self.validation_split: float = VALIDATION_SPLIT

        self.artifact_dir: str = os.path.join(
            ARTIFACT_DIR, TIMESTAMP, "data_transformation"
        )

        self.train_transforms_file: str = os.path.join(
            self.artifact_dir, TRAIN_TRANSFORMS_FILE
        )

        self.test_transforms_file: str = os.path.join(
            self.artifact_dir, TEST_TRANSFORMS_FILE
        )


class ModelTrainerConfig:
    """Configuration for the model training stage."""

    def __init__(self) -> None:
        self.artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP, "model_training")



        self.trained_model_path: str = os.path.join(
            self.artifact_dir, TRAINED_MODEL_NAME
        )

        self.train_transforms_key: str = TRAIN_TRANSFORMS_KEY

        self.epochs: int = EPOCH

        # AdamW optimizer params (replaces SGD for better fine-tuning)
        self.optimizer_params: dict = {
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "betas": (0.9, 0.999),
        }

        # CosineAnnealingWarmRestarts params (replaces StepLR)
        self.scheduler_params: dict = {"T_0": 5, "T_mult": 2, "eta_min": 1e-6}

        self.device: device = DEVICE

        self.early_stopping_patience: int = EARLY_STOPPING_PATIENCE

        self.gradient_clip_value: float = GRADIENT_CLIP_VALUE

        # Model architecture params
        self.num_classes: int = NUM_CLASSES
        self.dropout_rate: float = DROPOUT_RATE

        # Transfer learning
        self.freeze_backbone_epochs: int = FREEZE_BACKBONE_EPOCHS

        # Mixed precision
        self.use_amp: bool = USE_AMP


class ModelEvaluationConfig:
    """Configuration for the model evaluation stage."""

    def __init__(self) -> None:
        self.device: device = DEVICE
        self.optimizer_params: dict = {
            "lr": LEARNING_RATE,
            "momentum": MOMENTUM,
        }


