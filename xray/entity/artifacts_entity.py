"""
Artifact entities — immutable data containers passed between pipeline stages.

Each dataclass captures the outputs of a pipeline component so that downstream
components can consume them without tight coupling.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from torch.utils.data.dataloader import DataLoader


@dataclass
class DataIngestionArtifact:
    """Output of the data ingestion stage."""

    train_file_path: str
    test_file_path: str


@dataclass
class DataTransformationArtifact:
    """Output of the data transformation stage."""

    transformed_train_object: DataLoader
    transformed_test_object: DataLoader
    transformed_val_object: Optional[DataLoader]
    train_transform_file_path: str
    test_transform_file_path: str
    class_weights: Optional[List[float]] = None
    num_train_samples: int = 0


@dataclass
class ModelTrainerArtifact:
    """Output of the model training stage."""

    trained_model_path: str
    best_epoch: int = 0
    best_val_accuracy: float = 0.0
    final_train_loss: float = 0.0


@dataclass
class ModelEvaluationArtifact:
    """Output of the model evaluation stage."""

    model_accuracy: float
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    confusion_matrix: Optional[Dict] = None
    classification_report: str = ""

