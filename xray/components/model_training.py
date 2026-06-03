"""
Model Training component.

Handles the complete training loop with production-grade practices:
    - Class-weighted CrossEntropyLoss (fixes class imbalance)
    - AdamW optimizer (better for fine-tuning than SGD)
    - CosineAnnealingWarmRestarts scheduler (smooth LR decay)
    - Mixed precision training (AMP) for GPU acceleration
    - Backbone freezing for first N epochs (prevents catastrophic forgetting)
    - Early stopping based on macro F1 (not val_loss — avoids biased checkpoints)
    - Best model checkpointing with per-class metrics monitoring
    - Gradient clipping for training stability
    - Reproducibility via seed setting
"""

import os
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

from xray.constant.training_pipeline import DEVICE, PREDICTION_LABEL, SEED
from xray.entity.artifacts_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from xray.entity.config_entity import ModelTrainerConfig
from xray.exception import XRayException
from xray.logger import logging
from xray.ml.model.arch import XRayClassifier


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_epoch_metrics(
    all_targets: List[int], all_preds: List[int], num_classes: int = 2
) -> Dict:
    """
    Compute per-class precision, recall, F1, and macro averages.

    Used every epoch to monitor for class collapse. If NORMAL recall
    drops to 0, we know the model is collapsing — this is logged as
    a warning so it's caught early.
    """
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(all_targets, all_preds):
        cm[t][p] += 1

    precisions, recalls, f1s = [], [], []
    per_class = {}

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
            "f1": round(f1, 4),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    accuracy = sum(1 for t, p in zip(all_targets, all_preds) if t == p) / len(all_targets) * 100

    return {
        "accuracy": accuracy,
        "macro_precision": sum(precisions) / num_classes,
        "macro_recall": sum(recalls) / num_classes,
        "macro_f1": sum(f1s) / num_classes,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


class EarlyStopping:
    """
    Stops training when the monitored metric doesn't improve for `patience` epochs.

    Now monitors macro F1 (higher is better) instead of val_loss.
    This prevents saving a biased model that has low loss but only predicts one class.

    Args:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum improvement to qualify as an improvement.
    """

    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = -float("inf")
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """Check if score improved. Returns True if training should stop."""
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class ModelTrainer:
    """
    Trains the XRayClassifier with production-grade practices.

    Key improvements:
        - Class-weighted CrossEntropyLoss (fixes the 3:1 class imbalance)
        - AdamW optimizer (better convergence for transfer learning)
        - CosineAnnealingWarmRestarts scheduler (smooth LR with warm restarts)
        - Mixed precision training (AMP) — ~2x faster on RTX GPUs
        - Backbone freezing for first N epochs (learn classifier head first)
        - Early stopping on macro F1 (not val_loss)
        - Per-epoch per-class metrics to catch collapse early
    """

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

        # Initialize model with backbone frozen initially
        self.model = XRayClassifier(
            num_classes=model_trainer_config.num_classes,
            dropout_rate=model_trainer_config.dropout_rate,
            freeze_backbone=True,  # Start with frozen backbone
        )

        # Class-weighted CrossEntropyLoss — the #1 fix for class imbalance
        class_weights = data_transformation_artifact.class_weights
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(
                model_trainer_config.device
            )
            logging.info(f"Using class-weighted loss: {class_weights}")
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            logging.warning("No class weights provided — using unweighted loss")
            self.criterion = nn.CrossEntropyLoss()

                # Mixed precision scaler
        self.use_amp = False  # Disabled AMP to avoid GradScaler issues
        self.scaler = None  # No scaler needed when AMP is disabled

    def train_one_epoch(
        self, optimizer: Optimizer, epoch: int
    ) -> Tuple[float, float, List[int], List[int]]:
        """
        Train for one epoch with optional mixed precision.

        Returns:
            Tuple of (average_loss, accuracy_percentage, all_targets, all_preds)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_targets = []
        all_preds = []

        train_loader = self.data_transformation_artifact.transformed_train_object
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()

            if self.use_amp:
                # Mixed precision forward pass
                with torch.amp.autocast("cuda"):
                    output = self.model(data)
                    loss = self.criterion(output, target)

                # Check for NaN/Inf loss to avoid illegal memory access
                if not torch.isfinite(loss):
                    logging.warning("Loss is NaN or Inf, skipping optimizer step for this batch")
                    optimizer.zero_grad(set_to_none=True)
                    # Still update scaler to keep it in sync
                    self.scaler.update()
                    continue

                self.scaler.scale(loss).backward()

                # Gradient clipping (unscale first for AMP)
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.model_trainer_config.gradient_clip_value,
                )

                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.model_trainer_config.gradient_clip_value,
                )

                optimizer.step()

            running_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            all_targets.extend(target.cpu().tolist())
            all_preds.extend(pred.cpu().tolist())

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.0 * correct / total:.2f}%",
            )

        avg_loss = running_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy, all_targets, all_preds

    @torch.no_grad()
    def validate(
        self, data_loader: DataLoader, phase: str = "Val"
    ) -> Tuple[float, float, List[int], List[int]]:
        """
        Evaluate model on a given DataLoader (validation or test).

        Returns:
            Tuple of (average_loss, accuracy_percentage, all_targets, all_preds)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_targets = []
        all_preds = []

        for data, target in data_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    output = self.model(data)
                    loss = self.criterion(output, target)
            else:
                output = self.model(data)
                loss = self.criterion(output, target)

            running_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            all_targets.extend(target.cpu().tolist())
            all_preds.extend(pred.cpu().tolist())

        avg_loss = running_loss / total if total > 0 else 0.0
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        return avg_loss, accuracy, all_targets, all_preds

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Run the complete training loop with:
            - Backbone freezing → unfreezing schedule
            - Class-weighted loss
            - Mixed precision (AMP)
            - Early stopping on macro F1
            - Per-epoch per-class metrics logging
        """
        try:
            logging.info("Starting model training")
            set_seed(SEED)

            model = self.model.to(self.model_trainer_config.device)
            logging.info(
                f"Model has {model.get_trainable_params():,} trainable parameters "
                f"(backbone frozen)"
            )

            # AdamW optimizer — better for fine-tuning than SGD
            optimizer: Optimizer = torch.optim.AdamW(
                model.parameters(), **self.model_trainer_config.optimizer_params
            )

            # CosineAnnealingWarmRestarts — smooth LR decay with periodic restarts
            scheduler = CosineAnnealingWarmRestarts(
                optimizer=optimizer, **self.model_trainer_config.scheduler_params
            )

            early_stopping = EarlyStopping(
                patience=self.model_trainer_config.early_stopping_patience
            )

            os.makedirs(self.model_trainer_config.artifact_dir, exist_ok=True)

            best_macro_f1 = -1.0
            best_val_accuracy = 0.0
            best_epoch = 0

            for epoch in range(1, self.model_trainer_config.epochs + 1):
                # ---- Backbone unfreezing schedule ----
                if epoch == self.model_trainer_config.freeze_backbone_epochs + 1:
                    model.unfreeze_backbone()
                    logging.info(
                        f"  [INFO] Backbone unfrozen at epoch {epoch}. "
                        f"Trainable params: {model.get_trainable_params():,}"
                    )
                      # Adjust optimizer and scheduler after unfreezing backbone
                    base_lr = self.model_trainer_config.optimizer_params["lr"]
                    # Separate head and backbone parameters
                    head_params = []
                    backbone_params = []
                    for name, param in model.named_parameters():
                        if not param.requires_grad:
                            continue
                        if "backbone" in name:
                            backbone_params.append(param)
                        else:
                            head_params.append(param)
                    # Create new optimizer with separate LR groups
                    optimizer = torch.optim.AdamW(
                        [
                            {"params": head_params, "lr": base_lr},
                            {"params": backbone_params, "lr": base_lr * 0.1},
                        ],
                        **{k: v for k, v in self.model_trainer_config.optimizer_params.items() if k != "lr"}
                    )
                    # Recreate scheduler for the new optimizer
                    scheduler = CosineAnnealingWarmRestarts(
                        optimizer=optimizer, **self.model_trainer_config.scheduler_params
                    )
                    # Reset GradScaler – stale state can cause CUDA illegal memory access
                    if self.use_amp:
                        self.scaler = torch.amp.GradScaler()
                        logging.info("  GradScaler reset after optimizer recreation")
                    # Clear CUDA cache to prevent memory fragmentation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Log update
                    logging.info(
                        f"  [INFO] Optimizer recreated: head LR={base_lr:.6f}, backbone LR={base_lr * 0.1:.6f}"
                    )

                # ---- Train ----
                train_loss, train_acc, train_targets, train_preds = (
                    self.train_one_epoch(optimizer, epoch)
                )
                train_metrics = compute_epoch_metrics(train_targets, train_preds)

                # ---- Validate ----
                val_loader = self.data_transformation_artifact.transformed_val_object
                val_loss, val_acc, val_targets, val_preds = self.validate(
                    val_loader, phase="Val"
                )
                val_metrics = compute_epoch_metrics(val_targets, val_preds)

                # Step scheduler
                scheduler.step(epoch)

                current_lr = optimizer.param_groups[0]["lr"]
                macro_f1 = val_metrics["macro_f1"]

                # Log comprehensive metrics
                logging.info(
                    f"Epoch {epoch}/{self.model_trainer_config.epochs} — "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                    f"Val F1: {macro_f1:.4f} | LR: {current_lr:.6f}"
                )

                # Per-class metrics — critical for catching collapse
                for label, m in val_metrics["per_class"].items():
                    logging.info(
                        f"  {label}: P={m['precision']:.4f} R={m['recall']:.4f} "
                        f"F1={m['f1']:.4f}"
                    )

                # Warn if any class has 0 recall (collapse detection)
                for label, m in val_metrics["per_class"].items():
                    if m["recall"] == 0.0:
                        logging.warning(
                            f"  [WARNING] {label} recall is 0.0 — model may be collapsing!"
                        )

                # Save best model checkpoint based on macro F1
                if macro_f1 > best_macro_f1:
                    best_macro_f1 = macro_f1
                    best_val_accuracy = val_acc
                    best_epoch = epoch
                    torch.save(
                        model.state_dict(),
                        self.model_trainer_config.trained_model_path,
                    )
                    logging.info(
                        f"  [SUCCESS] New best model saved (macro_f1={macro_f1:.4f}, "
                        f"val_acc={val_acc:.2f}%)"
                    )

                # Check early stopping (on macro F1, higher is better)
                if early_stopping(macro_f1):
                    logging.info(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"Best epoch: {best_epoch} with F1={best_macro_f1:.4f}, "
                        f"val_acc={best_val_accuracy:.2f}%"
                    )
                    break

            # Load best model weights for downstream evaluation
            model.load_state_dict(
                torch.load(
                    self.model_trainer_config.trained_model_path,
                    map_location=DEVICE,
                    weights_only=True,
                )
            )

            # Final test evaluation
            test_loader = self.data_transformation_artifact.transformed_test_object
            test_loss, test_acc, test_targets, test_preds = self.validate(
                test_loader, phase="Test"
            )
            test_metrics = compute_epoch_metrics(test_targets, test_preds)

            logging.info(
                f"Final Test — Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%, "
                f"Macro F1: {test_metrics['macro_f1']:.4f}"
            )
            for label, m in test_metrics["per_class"].items():
                logging.info(
                    f"  {label}: P={m['precision']:.4f} R={m['recall']:.4f} "
                    f"F1={m['f1']:.4f}"
                )

            # Also save model to root directory for the serving apps
            root_model_path = os.path.join(os.getcwd(), "xray_model.pth")
            torch.save(model.state_dict(), root_model_path)
            logging.info(f"Model also saved to {root_model_path}")



            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.trained_model_path,
                best_epoch=best_epoch,
                best_val_accuracy=best_val_accuracy,
                final_train_loss=train_loss,
            )

            logging.info(
                f"Training complete — Best epoch: {best_epoch}, "
                f"Best macro F1: {best_macro_f1:.4f}, "
                f"Best val accuracy: {best_val_accuracy:.2f}%, "
                f"Test accuracy: {test_acc:.2f}%"
            )

            return model_trainer_artifact

        except Exception as e:
            raise XRayException(e, sys)
