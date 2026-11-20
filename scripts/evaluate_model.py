"""
Evaluate xray_model.pth on the standard Kermany chest X-ray test split (624 images)
and write reproducible metrics + plots.

Outputs
  reports/evaluation.json          — all metrics (read by the Streamlit app and README)
  assets/confusion_matrix.png      — confusion matrix at the default 0.5 threshold
  assets/roc_curve.png             — ROC curve
  assets/pr_curve.png              — precision-recall curve (PNEUMONIA = positive class)

Data sources (either works)
  --parquet PATH   Hugging Face hf-vision/chest-xray-pneumonia test parquet (label 0=NORMAL, 1=PNEUMONIA)
  --data-dir PATH  ImageFolder layout: PATH/NORMAL/*.jpeg, PATH/PNEUMONIA/*.jpeg

Usage
  python scripts/evaluate_model.py --parquet data/test-00000-of-00001.parquet
"""

import argparse
import io
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.model_utils import load_model, load_model_config, predict, preprocess_image  # noqa: E402
from xray.constant.training_pipeline import PREDICTION_LABEL  # noqa: E402

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def load_images(args):
    """Yield (PIL image, label) pairs; label 0 = NORMAL, 1 = PNEUMONIA."""
    if args.parquet:
        import pyarrow.parquet as pq
        t = pq.read_table(args.parquet)
        keep = None
        if args.split:
            keep = set(json.loads(Path(args.split).read_text())[args.subset])
        for img, label in zip(t.column("image").to_pylist(), t.column("label").to_pylist()):
            if keep is not None and img.get("path") not in keep:
                continue
            yield Image.open(io.BytesIO(img["bytes"])), int(label)
    else:
        for label, cls in PREDICTION_LABEL.items():
            for f in sorted((Path(args.data_dir) / cls).glob("*")):
                yield Image.open(f), label


def binary_metrics(y, p_pneu, thr):
    pred = (p_pneu >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn); spec = tn / (tn + fp)
    prec = tp / (tp + fp) if tp + fp else 0.0
    return {"threshold": thr, "accuracy": round((tp + tn) / len(y), 4),
            "sensitivity_pneumonia_recall": round(sens, 4), "specificity_normal_recall": round(spec, 4),
            "precision_pneumonia": round(prec, 4), "missed_pneumonia_cases": fn, "false_alarms": fp}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--parquet")
    g.add_argument("--data-dir")
    ap.add_argument("--split", help="reports/test_split.json: evaluate only one subset of the test set")
    ap.add_argument("--subset", default="holdout", choices=["holdout", "calibration"])
    args = ap.parse_args()

    from sklearn.metrics import (average_precision_score, brier_score_loss, classification_report,
                                 confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve)

    model = load_model()
    ys, probs = [], []
    with torch.no_grad():
        for img, label in load_images(args):
            _, _, prob = predict(model, preprocess_image(img))  # applies TTA if model_config.json says so
            p = prob["PNEUMONIA"]
            ys.append(label); probs.append(p)
    y, p = np.array(ys), np.array(probs)
    cfg = load_model_config()
    thr = float(cfg.get("threshold", 0.5))
    pred = (p >= thr).astype(int)

    rep = classification_report(y, pred, target_names=["NORMAL", "PNEUMONIA"], output_dict=True)
    cm = confusion_matrix(y, pred)
    result = {
        "generated": date.today().isoformat(),
        "test_set": "Kermany et al. chest X-ray test split" + (
            f" — {args.subset} subset ({args.split})" if args.split else ""),
        "subset": args.subset if args.split else "full",
        "n_images": int(len(y)), "n_normal": int((y == 0).sum()), "n_pneumonia": int((y == 1).sum()),
        "accuracy": round(rep["accuracy"], 4),
        "macro_f1": round(rep["macro avg"]["f1-score"], 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc_pneumonia": round(float(average_precision_score(y, p)), 4),
        "brier_score": round(float(brier_score_loss(y, p)), 4),
        "per_class": {c: {k: round(rep[c][k], 4) for k in ("precision", "recall", "f1-score")} | {"support": int(rep[c]["support"])}
                      for c in ("NORMAL", "PNEUMONIA")},
        "confusion_matrix": {"labels": ["NORMAL", "PNEUMONIA"], "rows_actual_cols_predicted": cm.tolist()},
        "model": {"arch": cfg.get("arch", "resnet18"), "decision_threshold": round(thr, 3),
                  "tta_hflip": bool(cfg.get("tta_hflip", False))},
        "threshold_analysis": [binary_metrics(y, p, t) for t in sorted(set(THRESHOLDS + [round(thr, 3)]))],
        "note": ("The test split is used only for reporting. The decision threshold was chosen on a patient-level "
                 "validation split (see reports/training_v2.json); the other rows show the trade-off."),
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    assets = ROOT / "assets"; assets.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["NORMAL", "PNEUMONIA"], yticklabels=["NORMAL", "PNEUMONIA"],
           xlabel="Predicted", ylabel="Actual", title=f"Confusion Matrix (n={len(y)}, threshold {thr:.2f})")
    fig.tight_layout(); fig.savefig(assets / "confusion_matrix.png", dpi=150); plt.close(fig)

    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(fpr, tpr, lw=2, label=f"ROC-AUC = {result['roc_auc']:.3f}"); ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (sensitivity)", title="ROC Curve")
    ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(assets / "roc_curve.png", dpi=150); plt.close(fig)

    prec, rec, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(rec, prec, lw=2, label=f"PR-AUC = {result['pr_auc_pneumonia']:.3f}")
    ax.set(xlabel="Recall (pneumonia)", ylabel="Precision (pneumonia)", title="Precision-Recall Curve", ylim=(0, 1.02))
    ax.legend(loc="lower left"); fig.tight_layout(); fig.savefig(assets / "pr_curve.png", dpi=150); plt.close(fig)

    print(json.dumps({k: result[k] for k in ("accuracy", "macro_f1", "roc_auc", "pr_auc_pneumonia", "per_class")}, indent=1))
    print("threshold analysis:")
    for r in result["threshold_analysis"]:
        print(" ", r)


if __name__ == "__main__":
    main()
