"""
Retrain the pneumonia classifier with a leak-free validation set and pick the
decision threshold on validation (never on test).

Data: Kermany chest X-ray dataset as Hugging Face parquet files
      (hf-vision/chest-xray-pneumonia; label 0 = NORMAL, 1 = PNEUMONIA).

Pipeline
  0. Histogram-equalise every image (train, val, test, and in the app) to reduce the
     intensity/contrast shift between the Kermany train and test sources.
  1. Pool the official train + val images (the official val split has only 16 images),
     then split 85/15 **by patient** so no patient appears in both train and validation.
  2. Fine-tune each architecture (ResNet18 / DenseNet121 / EfficientNet-B0) with AMP,
     AdamW, cosine LR, class-balanced loss, label smoothing and X-ray-safe augmentation.
  3. Keep the epoch with the best validation ROC-AUC per architecture; choose the
     architecture by validation ROC-AUC; choose the threshold that maximises balanced
     accuracy on validation (with horizontal-flip TTA).
  4. Evaluate the chosen model ONCE on the 624-image test split.
  5. Save xray_model.pth + model_config.json (arch, threshold, TTA) at the repo root.

Usage
  python scripts/train_v2.py --data-dir ../data --archs resnet18 densenet121 efficientnet_b0 --epochs 12
"""

import argparse
import glob
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xray.ml.model.arch import XRayClassifier  # noqa: E402

SEED = 42
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = dict(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda")  # bf16: no fp16 overflow/NaN


# --------------------------------------------------------------------------- data
def patient_id(path: str) -> str:
    """'person1000_bacteria_2931.jpeg' -> 'person1000'; 'NORMAL2-IM-1427-0001.jpeg' -> 'NORMAL2-IM-1427'."""
    name = Path(path).name
    m = re.match(r"(person\d+)", name)
    if m:
        return m.group(1)
    m = re.match(r"((?:NORMAL2-)?IM-\d+)", name)
    return m.group(1) if m else name


def load_parquets(pattern: str, size: int = 256):
    """Decode once, resize the short side to `size`, keep grayscale uint8 in memory."""
    import pyarrow.parquet as pq
    imgs, labels, paths = [], [], []
    for f in sorted(glob.glob(pattern)):
        t = pq.read_table(f)
        for rec, lab in zip(t.column("image").to_pylist(), t.column("label").to_pylist()):
            im = ImageOps.equalize(Image.open(io.BytesIO(rec["bytes"])).convert("L"))  # same as inference
            w, h = im.size
            s = size / min(w, h)
            im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BILINEAR)
            imgs.append(np.asarray(im))
            labels.append(int(lab))
            paths.append(rec.get("path") or "")
    return imgs, np.array(labels), paths


def patient_split(labels, paths, val_frac=0.15, seed=SEED):
    rng = np.random.default_rng(seed)
    pids = np.array([patient_id(p) for p in paths])
    val_mask = np.zeros(len(labels), bool)
    for cls in (0, 1):
        cls_p = np.unique(pids[labels == cls])
        rng.shuffle(cls_p)
        chosen = set(cls_p[: int(round(len(cls_p) * val_frac))])
        val_mask |= (labels == cls) & np.isin(pids, list(chosen))
    assert not set(pids[val_mask]) & set(pids[~val_mask]), "patient leak between train and val"
    return ~val_mask, val_mask


class ImgDS(torch.utils.data.Dataset):
    def __init__(self, imgs, labels, tf):
        self.imgs, self.labels, self.tf = imgs, labels, tf

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        x = torch.from_numpy(np.ascontiguousarray(self.imgs[i]).copy()).unsqueeze(0).expand(3, -1, -1)  # gray -> 3ch uint8
        return self.tf(x), self.labels[i]


class RandomGamma(torch.nn.Module):
    """Random gamma on a [0, 1] float image — simulates exposure differences between scanners."""
    def __init__(self, low=0.7, high=1.4, p=0.5):
        super().__init__(); self.low, self.high, self.p = low, high, p

    def forward(self, x):
        if torch.rand(1).item() < self.p:
            g = float(torch.empty(1).uniform_(self.low, self.high))
            x = x.clamp(0, 1) ** g
        return x


def build_tfs(size: int):
    train_tf = T.Compose([
        T.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.85, 1.15), antialias=True),
        T.RandomRotation(10),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.35, contrast=0.35),
        T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.2))], p=0.2),
        T.ToDtype(torch.float32, scale=True),
        RandomGamma(),
        T.Normalize(MEAN, STD),
    ])
    eval_tf = T.Compose([T.CenterCrop(size), T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])
    return train_tf, eval_tf


def short_side(size: int) -> int:
    return round(size * 256 / 224)


# --------------------------------------------------------------------------- metrics
def predict_proba(model, loader, tta=True):
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(**AMP):
        for x, _ in loader:
            x = x.to(DEVICE, non_blocking=True)
            p = torch.softmax(model(x).float(), 1)[:, 1]
            if tta:
                p = (p + torch.softmax(model(torch.flip(x, dims=[3])).float(), 1)[:, 1]) / 2
            out.append(p.cpu())
    return torch.cat(out).numpy()


def summarize(y, p, thr):
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    pred = (p >= thr).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); tn = ((pred == 0) & (y == 0)).sum()
    fp = ((pred == 1) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    return {"threshold": round(float(thr), 3), "accuracy": round(float((tp + tn) / len(y)), 4),
            "roc_auc": round(float(roc_auc_score(y, p)), 4), "pr_auc": round(float(average_precision_score(y, p)), 4),
            "sensitivity": round(float(tp / (tp + fn)), 4), "specificity": round(float(tn / (tn + fp)), 4),
            "precision": round(float(tp / max(tp + fp, 1)), 4), "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
            "missed_pneumonia": int(fn), "false_alarms": int(fp)}


def best_threshold(y, p):
    """Threshold maximising balanced accuracy on validation."""
    grid = np.linspace(0.05, 0.95, 181)
    ba = [((p[y == 1] >= t).mean() + (p[y == 0] < t).mean()) / 2 for t in grid]
    return float(grid[int(np.argmax(ba))])


# --------------------------------------------------------------------------- training
def train_arch(arch, train_dl, val_dl, y_val, epochs, lr, ckpt_path=None):
    torch.manual_seed(SEED)
    model = XRayClassifier(arch=arch, pretrained=True).to(DEVICE)
    counts = np.bincount(train_dl.dataset.labels, minlength=2)
    weights = torch.tensor(len(train_dl.dataset.labels) / (2 * counts), dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(train_dl), pct_start=0.15)
    best = {"auc": -1}
    start = 1
    if ckpt_path is not None and Path(ckpt_path).exists():
        # Resume after a crash (the laptop GPU throws sporadic CUDA errors)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        best, start = ck["best"], ck["epoch"] + 1
        torch.set_rng_state(ck["rng"])
        print(f"[{arch}] resumed after epoch {ck['epoch']} (best val_auc {best['auc']:.4f})", flush=True)
    for ep in range(start, epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in train_dl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(**AMP):
                logits = model(x).float()
            if logits.shape != (len(y), 2) or not torch.isfinite(logits).all():
                print(f"[{arch}] bad batch: x {tuple(x.shape)} {x.dtype} logits {tuple(logits.shape)} "
                      f"finite={torch.isfinite(logits).all().item()} y {y[:8].tolist()}", flush=True)
                raise RuntimeError("non-finite or mis-shaped logits")
            loss = crit(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(y)
        pv = predict_proba(model, val_dl)
        from sklearn.metrics import roc_auc_score
        if not np.isfinite(pv).all():
            print(f"[{arch}] epoch {ep}: non-finite validation outputs, skipping", flush=True)
            continue
        auc = roc_auc_score(y_val, pv)
        print(f"[{arch}] epoch {ep:2d} loss {tot / len(train_dl.dataset):.4f} val_auc {auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if auc > best["auc"]:
            best = {"auc": auc, "epoch": ep, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        if ckpt_path is not None:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "best": best, "epoch": ep, "rng": torch.get_rng_state()}, ckpt_path)
    model.load_state_dict(best["state"])
    return model, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT.parent / "data"))
    ap.add_argument("--archs", nargs="+", default=["resnet18", "densenet121", "efficientnet_b0"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=224, help="crop size (images resized to size*256/224 first)")
    ap.add_argument("--cand-dir", default=str(ROOT.parent / "candidates"))
    ap.add_argument("--select", action="store_true", help="pick the best saved candidate and evaluate on test")
    args = ap.parse_args()
    d = Path(args.data_dir)
    print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "")

    TRAIN_TF, EVAL_TF = build_tfs(args.size)
    imgs, labels, paths = load_parquets(str(d / "train-*.parquet"), short_side(args.size))
    v_imgs, v_labels, v_paths = load_parquets(str(d / "validation-*.parquet"), short_side(args.size))
    imgs += v_imgs; labels = np.concatenate([labels, v_labels]); paths += v_paths
    tr, va = patient_split(labels, paths)
    t_imgs, t_labels, _ = load_parquets(str(d / "test*.parquet"), short_side(args.size))
    print(f"train {tr.sum()} (pneumonia {labels[tr].sum()}), val {va.sum()} (pneumonia {labels[va].sum()}), test {len(t_labels)}")

    mk = lambda idx, tf, shuffle: torch.utils.data.DataLoader(  # noqa: E731
        ImgDS([imgs[i] for i in np.where(idx)[0]], labels[idx], tf), batch_size=args.batch,
        shuffle=shuffle, num_workers=0, pin_memory=True)
    train_dl, val_dl = mk(tr, TRAIN_TF, True), mk(va, EVAL_TF, False)
    test_dl = torch.utils.data.DataLoader(ImgDS(t_imgs, t_labels, EVAL_TF), batch_size=args.batch)
    y_val = labels[va]

    cand_dir = Path(args.cand_dir); cand_dir.mkdir(parents=True, exist_ok=True)
    if not args.select:
        # Train each architecture and save it as a candidate (skips ones already trained, so a
        # crashed run can simply be restarted).
        for arch in args.archs:
            if (cand_dir / f"{arch}.json").exists():
                print(f"[{arch}] candidate exists, skipping"); continue
            model, best = train_arch(arch, train_dl, val_dl, y_val, args.epochs, args.lr,
                                     ckpt_path=cand_dir / f"{arch}_ckpt.pt")
            pv = predict_proba(model, val_dl)
            thr = best_threshold(y_val, pv)
            res = {"best_epoch": best["epoch"], "val": summarize(y_val, pv, thr)}
            torch.save(model.cpu().state_dict(), cand_dir / f"{arch}.pth")
            (cand_dir / f"{arch}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
            print(f"[{arch}] val @thr {thr:.3f}: {res['val']}", flush=True)
            torch.cuda.empty_cache()
        return

    # Selection: choose by validation ROC-AUC only, then score the chosen model ONCE on test.
    results = {a: json.loads((cand_dir / f"{a}.json").read_text()) for a in args.archs if (cand_dir / f"{a}.json").exists()}
    chosen = max(results, key=lambda a: results[a]["val"]["roc_auc"])
    model = XRayClassifier(arch=chosen, pretrained=False)
    model.load_state_dict(torch.load(cand_dir / f"{chosen}.pth", map_location="cpu", weights_only=True))
    model = model.to(DEVICE)
    thr = results[chosen]["val"]["threshold"]
    pt = predict_proba(model, test_dl)
    test = {"chosen_threshold": summarize(t_labels, pt, thr), "threshold_0.5": summarize(t_labels, pt, 0.5)}
    print(f"CHOSEN {chosen} (val ROC-AUC {results[chosen]['val']['roc_auc']}) -> TEST: {json.dumps(test, indent=1)}")

    torch.save(model.cpu().state_dict(), ROOT / "xray_model.pth")
    cfg = {"arch": chosen, "threshold": thr, "tta_hflip": True, "preprocess": "equalize",
           "image_size": 224, "resize_short_side": 256,
           "selected_by": "validation ROC-AUC (patient-level split); threshold = max balanced accuracy on validation"}
    (ROOT / "model_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "training_v2.json").write_text(json.dumps(
        {"split": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(len(t_labels))},
         "epochs": args.epochs, "lr": args.lr, "candidates": results, "chosen": chosen, "test": test}, indent=2), encoding="utf-8")
    print("saved xray_model.pth, model_config.json, reports/training_v2.json")


if __name__ == "__main__":
    main()
