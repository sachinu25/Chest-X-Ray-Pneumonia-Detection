"""
Model Evaluation component.

Provides comprehensive evaluation metrics beyond simple accuracy:
    - Precision, Recall, F1-Score (per-class and macro-averaged)
    - Confusion Matrix
    - Classification Report
    - ROC-AUC
    - Average Loss
"""

import math
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import DataLoader

from xray.constant.training_pipeline import PREDICTION_LABEL
from xray.entity.artifacts_entity import (
    DataTransformationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from xray.entity.config_entity import ModelEvaluationConfig
from xray.exception import XRayException
from xray.logger import logging
from xray.ml.model.arch import XRayClassifier


def compute_roc_auc(
    all_targets: List[int], all_probs: List[List[float]], positive_class: int = 1
) -> float:
    """
    Compute ROC-AUC without sklearn using the trapezoidal rule.

    For binary classification, we use the probability of the positive class
    and sweep across thresholds to build the ROC curve.
    """
    # Extract probability of the positive class
    scores = [p[positive_class] for p in all_probs]
    labels = [1 if t == positive_class else 0 for t in all_targets]

    # Sort by score descending
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])

    tp = 0
    fp = 0
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos

    if total_pos == 0 or total_neg == 0:
        return 0.0

    tpr_prev = 0.0
    fpr_prev = 0.0
    auc = 0.0

    for score, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1

        tpr = tp / total_pos
        fpr = fp / total_neg

        # Trapezoidal rule
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0
        tpr_prev = tpr
        fpr_prev = fpr

    return auc


def compute_metrics(
    all_targets: List[int],
    all_preds: List[int],
    all_probs: List[List[float]] = None,
    num_classes: int = 2,
) -> Dict:
    """
    Compute precision, recall, F1-score, confusion matrix, and ROC-AUC.

    Args:
        all_targets: Ground truth labels.
        all_preds: Model predictions.
        all_probs: Softmax probabilities per sample (for ROC-AUC).
        num_classes: Number of classes.

    Returns:
        Dictionary with all computed metrics.
    """
    # Confusion matrix
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(all_targets, all_preds):
        cm[t][p] += 1

    # Per-class metrics
    per_class = {}
    precisions, recalls, f1s = [], [], []

    for cls in range(num_classes):
        tp = cm[cls][cls]
        fp = sum(cm[r][cls] for r in range(num_classes)) - tp
        fn = sum(cm[cls][c] for c in range(num_classes)) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        label = PREDICTION_LABEL.get(cls, str(cls))
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": sum(cm[cls]),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    # Macro averages
    macro_precision = sum(precisions) / num_classes
    macro_recall = sum(recalls) / num_classes
    macro_f1 = sum(f1s) / num_classes

    total = len(all_targets)
    accuracy = sum(1 for t, p in zip(all_targets, all_preds) if t == p) / total * 100

    # ROC-AUC
    roc_auc = 0.0
    if all_probs is not None:
        roc_auc = compute_roc_auc(all_targets, all_probs)

    # Build text report
    report_lines = [
        "",
        f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}",
        "-" * 62,
    ]
    for label, m in per_class.items():
        report_lines.append(
            f"{label:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} "
            f"{m['f1_score']:>10.4f} {m['support']:>10}"
        )
    report_lines.append("-" * 62)
    report_lines.append(
        f"{'Macro Avg':<20} {macro_precision:>10.4f} {macro_recall:>10.4f} "
        f"{macro_f1:>10.4f} {total:>10}"
    )
    report_lines.append(f"\nAccuracy: {accuracy:.2f}%")
    report_lines.append(f"ROC-AUC: {roc_auc:.4f}")
    report_lines.append(f"\nConfusion Matrix:")
    for row_idx, row in enumerate(cm):
        label = PREDICTION_LABEL.get(row_idx, str(row_idx))
        report_lines.append(f"  {label:<15} {row}")
    report_text = "\n".join(report_lines)

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "per_class": per_class,
        "confusion_matrix": cm,
        "report": report_text,
    }


class ModelEvaluation:
    """
    Evaluates a trained model with comprehensive metrics.

    Loads the best model checkpoint and evaluates it on the held-out test set
    to produce precision, recall, F1-score, ROC-AUC, confusion matrix, and
    a full classification report.

    Auto-detects whether the checkpoint uses the legacy Net architecture
    or the newer XRayClassifier.
    """

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifact = model_trainer_artifact

    def load_model(self) -> Module:
        """Load the best model checkpoint, auto-detecting architecture."""
        logging.info("Loading trained model for evaluation")
        try:
            state = torch.load(
                self.model_trainer_artifact.trained_model_path,
                map_location=self.model_evaluation_config.device,
            )

            if isinstance(state, torch.nn.Module):
                model = state
            elif isinstance(state, dict):
                logging.info("Loading XRayClassifier checkpoint")
                model = XRayClassifier()
                model.load_state_dict(state)
            else:
                raise RuntimeError(f"Unknown checkpoint format: {type(state)}")

            model.to(self.model_evaluation_config.device)
            model.eval()
            logging.info("Model loaded successfully")
            return model

        except Exception as e:
            raise XRayException(e, sys)

    @torch.no_grad()
    def evaluate(self, model: Module) -> Dict:
        """
        Run evaluation on the test set and compute all metrics including ROC-AUC.

        Returns:
            Dictionary with accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
        """
        logging.info("Starting model evaluation on test set")

        try:
            test_loader: DataLoader = (
                self.data_transformation_artifact.transformed_test_object
            )
            criterion = nn.CrossEntropyLoss()

            all_targets: List[int] = []
            all_preds: List[int] = []
            all_probs: List[List[float]] = []
            total_loss = 0.0
            total_samples = 0

            for data, target in test_loader:
                data = data.to(self.model_evaluation_config.device)
                target = target.to(self.model_evaluation_config.device)

                output = model(data)
                loss = criterion(output, target)

                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)

                probs = F.softmax(output, dim=1)
                preds = torch.argmax(output, dim=1)

                all_targets.extend(target.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())

            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            metrics = compute_metrics(all_targets, all_preds, all_probs)
            metrics["avg_loss"] = avg_loss

            logging.info(f"Evaluation complete — {metrics['report']}")
            return metrics

        except Exception as e:
            raise XRayException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """Run the complete model evaluation pipeline."""
        logging.info("Initiating model evaluation")

        try:
            model = self.load_model()
            metrics = self.evaluate(model)

            model_evaluation_artifact = ModelEvaluationArtifact(
                model_accuracy=metrics["accuracy"],
                precision=metrics["macro_precision"],
                recall=metrics["macro_recall"],
                f1_score=metrics["macro_f1"],
                roc_auc=metrics["roc_auc"],
                confusion_matrix={"matrix": metrics["confusion_matrix"]},
                classification_report=metrics["report"],
            )

            logging.info(
                f"Model Evaluation Results:\n"
                f"  Accuracy:  {metrics['accuracy']:.2f}%\n"
                f"  Precision: {metrics['macro_precision']:.4f}\n"
                f"  Recall:    {metrics['macro_recall']:.4f}\n"
                f"  F1-Score:  {metrics['macro_f1']:.4f}\n"
                f"  ROC-AUC:   {metrics['roc_auc']:.4f}"
            )

            return model_evaluation_artifact

        except Exception as e:
            raise XRayException(e, sys)
