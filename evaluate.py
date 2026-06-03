"""
Standalone model evaluation script.

Evaluates xray_model.pth on the test set and prints comprehensive metrics:
    - Accuracy
    - Per-class Precision, Recall, F1-Score
    - Macro-averaged metrics
    - ROC-AUC
    - Confusion Matrix
    - Average Loss
"""

import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from xray.ml.model.arch import XRayClassifier
from xray.constant.training_pipeline import (
    NORMALIZE_LIST_1,
    NORMALIZE_LIST_2,
    PREDICTION_LABEL,
    NUM_CLASSES,
    BATCH_SIZE,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.getcwd(), "xray_model.pth")
TEST_DIR = os.path.join(os.getcwd(), "data", "test")
EVAL_DIR = os.path.join(
    os.getcwd(), "artifacts", f"evaluation_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
)

# ---------------------------------------------------------------------------
# Model Loading (auto-detect architecture)
# ---------------------------------------------------------------------------
def load_model():
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

    if not isinstance(state, dict):
        raise RuntimeError(
            f"Expected a state_dict (dict), got {type(state)}. "
            "Full-model pickle loading is disabled for security."
        )

    print("[INFO] Detected XRayClassifier checkpoint")
    model = XRayClassifier(num_classes=NUM_CLASSES)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def _save_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Actual",
        xlabel="Predicted",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_roc_curve(fpr, tpr, roc_auc, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate():
    print(f"Device: {DEVICE}")
    print(f"Model:  {MODEL_PATH}")
    print(f"Test:   {TEST_DIR}")
    print(f"Eval:   {EVAL_DIR}")
    print()

    # Test transforms (must match training validation transforms)
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_LIST_1, std=NORMALIZE_LIST_2),
    ])

    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True
    )

    os.makedirs(EVAL_DIR, exist_ok=True)

    class_names = test_dataset.classes  # folder names sorted alphabetically
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    print(f"Test samples: {len(test_dataset)}")
    print()

    model = load_model()
    criterion = nn.CrossEntropyLoss()

    all_targets = []
    all_preds = []
    all_probs = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

            probs = F.softmax(output, dim=1)
            preds = torch.argmax(output, dim=1)

            all_targets.extend(target.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    avg_loss = total_loss / total_samples

    # Core metrics
    accuracy = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )
    per_class = precision_recall_fscore_support(
        all_targets, all_preds, average=None, labels=list(range(num_classes)), zero_division=0
    )

    # ROC-AUC and ROC curve (binary)
    probs_np = np.array(all_probs)
    roc_auc = 0.0
    fpr = tpr = None
    if num_classes == 2:
        y_true = np.array(all_targets)
        y_score = probs_np[:, 1]
        try:
            roc_auc = roc_auc_score(y_true, y_score)
            fpr, tpr, _ = roc_curve(y_true, y_score)
        except ValueError:
            roc_auc = 0.0

    # Reporting artifacts
    report_str = classification_report(
        all_targets, all_preds, target_names=class_names, digits=4, zero_division=0
    )
    report_path = os.path.join(EVAL_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str)

    cm_img_path = os.path.join(EVAL_DIR, "confusion_matrix.png")
    _save_confusion_matrix(cm, class_names, cm_img_path)

    roc_img_path = None
    if fpr is not None and tpr is not None:
        roc_img_path = os.path.join(EVAL_DIR, "roc_curve.png")
        _save_roc_curve(fpr, tpr, roc_auc, roc_img_path)

    # Extra counts
    test_count = len(test_dataset)
    class_distribution = {
        class_names[i]: int(sum(1 for t in all_targets if t == i)) for i in range(num_classes)
    }
    pred_distribution = {
        class_names[i]: int(sum(1 for p in all_preds if p == i)) for i in range(num_classes)
    }
    misclassified = int(sum(1 for t, p in zip(all_targets, all_preds) if t != p))

    print("=" * 70)
    print("                     MODEL EVALUATION RESULTS")
    print("=" * 70)
    print()
    print(f"  Test Accuracy:     {accuracy * 100:.2f}%")
    print(f"  Test Loss:         {avg_loss:.4f}")
    print(f"  Precision Macro:   {prec_macro:.4f}")
    print(f"  Precision Weighted:{prec_weighted:.4f}")
    print(f"  Recall Macro:      {rec_macro:.4f}")
    print(f"  Recall Weighted:   {rec_weighted:.4f}")
    print(f"  F1 Macro:          {f1_macro:.4f}")
    print(f"  F1 Weighted:       {f1_weighted:.4f}")
    print(f"  ROC-AUC:           {roc_auc:.4f}")
    print()
    print("-" * 70)
    print(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 70)

    per_prec, per_rec, per_f1, per_support = per_class
    for idx, label in enumerate(class_names):
        print(
            f"  {label:<20} {per_prec[idx]:>10.4f} {per_rec[idx]:>10.4f} {per_f1[idx]:>10.4f} {per_support[idx]:>10}"
        )

    print("-" * 70)
    print(f"  {'Macro Avg':<20} {prec_macro:>10.4f} {rec_macro:>10.4f} {f1_macro:>10.4f} {test_count:>10}")
    print(f"  {'Weighted Avg':<20} {prec_weighted:>10.4f} {rec_weighted:>10.4f} {f1_weighted:>10.4f} {test_count:>10}")
    print()
    print("  Confusion Matrix:")
    print(cm)
    print()
    print("  Classification Report:")
    print(report_str)
    print()
    print("  Additional Counts:")
    print(f"  - Test images evaluated: {test_count}")
    print(f"  - Class distribution: {class_distribution}")
    print(f"  - Predictions per class: {pred_distribution}")
    print(f"  - Misclassified images: {misclassified}")
    print()
    print("  Saved Artifacts:")
    print(f"  - Classification report: {report_path}")
    print(f"  - Confusion matrix image: {cm_img_path}")
    if roc_img_path:
        print(f"  - ROC curve image: {roc_img_path}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    evaluate()
