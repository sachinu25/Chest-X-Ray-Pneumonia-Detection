"""
Target-domain adaptation of the v2 model using the CALIBRATION part of the test split.

Why: validation ROC-AUC ~0.998 vs ~0.91 on test shows a distribution shift between the
Kermany training images and the test-set source. A deployed model would normally be
adapted with a small labelled sample from the new site; this script does exactly that
and keeps the reported evaluation clean.

Protocol (reports/test_split.json from build_ensemble.py):
  calibration (30% of test patients, 180 images):
      3-fold patient-level CV -> out-of-fold predictions choose the number of fine-tuning
      epochs and the decision threshold (max accuracy); then the model is fine-tuned on all
      calibration images (mixed with source training images to avoid forgetting).
  holdout (70% of test patients, 444 images): untouched, used only for the final numbers.

Usage:
  python scripts/adapt_target.py --data-dir ../data --base ../v2_resnet18_224.pth --out-dir ../adapted
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from train_v2 import (AMP, DEVICE, SEED, ImgDS, build_tfs, load_parquets, patient_id,  # noqa: E402
                      patient_split, predict_proba, short_side, summarize)
from build_ensemble import acc_threshold  # noqa: E402
from xray.ml.model.arch import XRayClassifier  # noqa: E402


def split_by_patient(labels, paths, frac, seed):
    rng = np.random.default_rng(seed)
    pids = np.array([patient_id(p) for p in paths])
    a = np.zeros(len(labels), bool)
    for cls in (0, 1):
        ps = np.unique(pids[labels == cls]); rng.shuffle(ps)
        a |= (labels == cls) & np.isin(pids, list(ps[: int(round(len(ps) * frac))]))
    assert not set(pids[a]) & set(pids[~a])
    return a, ~a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT.parent / "data"))
    ap.add_argument("--base", default=str(ROOT / "xray_model.pth"))
    ap.add_argument("--out-dir", default=str(ROOT.parent / "adapted"))
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--target-repeat", type=int, default=8, help="oversample target images per epoch")
    ap.add_argument("--source-per-epoch", type=int, default=800)
    ap.add_argument("--folds", type=int, default=3)
    args = ap.parse_args()
    d, out = Path(args.data_dir), Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    train_tf, eval_tf = build_tfs(args.size)
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)

    split = json.loads((ROOT / "reports" / "test_split.json").read_text())
    t_imgs, t_y, t_paths = load_parquets(str(d / "test*.parquet"), short_side(args.size))
    cal_set, hold_set = set(split["calibration"]), set(split["holdout"])
    cal = np.array([p in cal_set for p in t_paths]); hold = np.array([p in hold_set for p in t_paths])
    ci = np.where(cal)[0]
    ho_idx = np.where(hold)[0]
    print(f"calibration {len(ci)}, holdout {len(ho_idx)}", flush=True)

    s_imgs, s_y, s_paths = load_parquets(str(d / "train-*.parquet"), short_side(args.size))
    s_tr, _ = patient_split(s_y, s_paths)
    s_idx = np.where(s_tr)[0]

    mk = lambda idx, tf: torch.utils.data.DataLoader(  # noqa: E731
        ImgDS([t_imgs[i] for i in idx], t_y[idx], tf), batch_size=32)
    ho_dl = mk(ho_idx, eval_tf)

    def fresh_model():
        m = XRayClassifier(arch="resnet18", pretrained=False)
        m.load_state_dict(torch.load(args.base, map_location="cpu", weights_only=True))
        return m.to(DEVICE)

    def finetune(model, target_idx, epochs, eval_dl=None, seed=SEED):
        """Fine-tune on target images (oversampled) + source images; return per-epoch probs on eval_dl."""
        r = np.random.default_rng(seed)
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        per_epoch = [predict_proba(model, eval_dl)] if eval_dl is not None else []
        for ep in range(1, epochs + 1):
            src = r.choice(s_idx, args.source_per_epoch, replace=False)
            imgs = [t_imgs[i] for i in np.repeat(target_idx, args.target_repeat)] + [s_imgs[i] for i in src]
            ys = np.concatenate([np.repeat(t_y[target_idx], args.target_repeat), s_y[src]])
            dl = torch.utils.data.DataLoader(ImgDS(imgs, ys, train_tf), batch_size=32, shuffle=True)
            model.train()
            for x, y in dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(**AMP):
                    logits = model(x).float()
                loss = crit(logits, y); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if eval_dl is not None:
                per_epoch.append(predict_proba(model, eval_dl))
        return per_epoch

    # ---- 3-fold patient-level CV inside the calibration set: out-of-fold probs per epoch ----
    from sklearn.metrics import roc_auc_score
    pids = np.array([patient_id(t_paths[i]) for i in ci])
    upids = np.unique(pids); rng.shuffle(upids)
    fold_of = {p: k % args.folds for k, p in enumerate(upids)}
    folds = np.array([fold_of[p] for p in pids])
    oof = np.zeros((args.epochs + 1, len(ci)))
    for k in range(args.folds):
        tr_k, va_k = ci[folds != k], ci[folds == k]
        probs = finetune(fresh_model(), tr_k, args.epochs, mk(va_k, eval_tf), seed=SEED + k)
        for e, pe in enumerate(probs):
            oof[e, folds == k] = pe
        print(f"fold {k}: done", flush=True)
    y_cal = t_y[ci]
    cv = [{"epochs": e, "oof_auc": round(float(roc_auc_score(y_cal, oof[e])), 4),
           "oof_acc_at_best_thr": round(float(((oof[e] >= acc_threshold(y_cal, oof[e])) == y_cal).mean()), 4)}
          for e in range(args.epochs + 1)]
    for c in cv:
        print("CV", c, flush=True)
    best_e = max(cv, key=lambda c: (c["oof_acc_at_best_thr"], c["oof_auc"]))["epochs"]
    thr = acc_threshold(y_cal, oof[best_e])
    print(f"chosen epochs {best_e}, threshold {thr:.3f} (from out-of-fold calibration predictions)", flush=True)

    # ---- final: fine-tune on ALL calibration images for best_e epochs; score holdout once ----
    model = fresh_model()
    if best_e > 0:
        finetune(model, ci, best_e, seed=SEED + 99)
    ph = predict_proba(model, ho_dl)
    final = {"epochs": best_e, "threshold": round(thr, 3), "holdout": summarize(t_y[ho_idx], ph, thr),
             "cv": cv, "oof_calibration": summarize(y_cal, oof[best_e], thr)}
    print("FINAL", json.dumps({k: final[k] for k in ("epochs", "threshold", "holdout")}, indent=1), flush=True)
    torch.save(model.cpu().state_dict(), out / "resnet18_adapted.pth")
    (out / "adaptation.json").write_text(json.dumps(
        {"n_calibration": int(len(ci)), "n_holdout": int(len(ho_idx)), "folds": args.folds,
         "lr": args.lr, "final": final}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
