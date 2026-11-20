"""
Build the production ensemble from trained candidates and calibrate it on the target domain.

Protocol (fixed before looking at any results):
  * The Kermany test split (624 images) is split BY PATIENT, stratified by label, into
      - calibration (30%): used only to choose the decision threshold (target-domain calibration)
      - holdout     (70%): untouched; all reported numbers come from here
    The split is saved to reports/test_split.json so it is reproducible.
  * Ensemble = mean of hflip-TTA probabilities of all candidates in --archs.
  * Threshold = value maximising accuracy on the calibration split.

Writes: xray_model.pth (first member), models/<arch>.pth (other members),
        model_config.json, reports/test_split.json, reports/training_v3.json

Usage:
  python scripts/build_ensemble.py --data-dir ../data --cand-dir ../candidates_320 --size 320
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train_v2 import (DEVICE, ImgDS, SEED, build_tfs, load_parquets, patient_id,  # noqa: E402
                      patient_split, predict_proba, short_side, summarize)
from xray.ml.model.arch import XRayClassifier  # noqa: E402


def split_test(labels, paths, calib_frac=0.3, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    pids = np.array([patient_id(p) for p in paths])
    calib = np.zeros(len(labels), bool)
    for cls in (0, 1):
        cls_p = np.unique(pids[labels == cls])
        rng.shuffle(cls_p)
        chosen = set(cls_p[: int(round(len(cls_p) * calib_frac))])
        calib |= (labels == cls) & np.isin(pids, list(chosen))
    assert not set(pids[calib]) & set(pids[~calib]), "patient leak between calibration and holdout"
    return calib, ~calib


def acc_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 181)
    accs = [((p >= t).astype(int) == y).mean() for t in grid]
    return float(grid[int(np.argmax(accs))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT.parent / "data"))
    ap.add_argument("--cand-dir", default=str(ROOT.parent / "candidates_320"))
    ap.add_argument("--archs", nargs="+", default=["resnet18", "efficientnet_b0"])
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--calib-frac", type=float, default=0.3)
    args = ap.parse_args()
    d, cand = Path(args.data_dir), Path(args.cand_dir)
    _, eval_tf = build_tfs(args.size)

    imgs, labels, paths = load_parquets(str(d / "train-*.parquet"), short_side(args.size))
    v_imgs, v_labels, v_paths = load_parquets(str(d / "validation-*.parquet"), short_side(args.size))
    imgs += v_imgs; labels = np.concatenate([labels, v_labels]); paths += v_paths
    _, va = patient_split(labels, paths)
    t_imgs, t_labels, t_paths = load_parquets(str(d / "test*.parquet"), short_side(args.size))
    calib, hold = split_test(t_labels, t_paths, args.calib_frac)
    print(f"val {va.sum()}, test calib {calib.sum()}, test holdout {hold.sum()}")

    val_dl = torch.utils.data.DataLoader(ImgDS([imgs[i] for i in np.where(va)[0]], labels[va], eval_tf), batch_size=16)
    test_dl = torch.utils.data.DataLoader(ImgDS(t_imgs, t_labels, eval_tf), batch_size=16)
    y_val = labels[va]

    pv, pt = {}, {}
    for arch in args.archs:
        m = XRayClassifier(arch=arch, pretrained=False)
        m.load_state_dict(torch.load(cand / f"{arch}.pth", map_location="cpu", weights_only=True))
        m = m.to(DEVICE)
        pv[arch], pt[arch] = predict_proba(m, val_dl), predict_proba(m, test_dl)
        del m; torch.cuda.empty_cache()
    pv["ensemble"] = np.mean([pv[a] for a in args.archs], axis=0)
    pt["ensemble"] = np.mean([pt[a] for a in args.archs], axis=0)

    report = {}
    for name in args.archs + ["ensemble"]:
        thr = acc_threshold(t_labels[calib], pt[name][calib])
        report[name] = {"threshold_from_calibration": round(thr, 3),
                        "val": summarize(y_val, pv[name], thr),
                        "calibration": summarize(t_labels[calib], pt[name][calib], thr),
                        "holdout": summarize(t_labels[hold], pt[name][hold], thr)}
        print(f"{name:16s} thr {thr:.3f} | holdout {report[name]['holdout']}")

    thr = report["ensemble"]["threshold_from_calibration"]
    # artifacts: first member at the repo root (keeps Grad-CAM/FastAPI paths), others under models/
    (ROOT / "models").mkdir(exist_ok=True)
    members = []
    for i, arch in enumerate(args.archs):
        dst = "xray_model.pth" if i == 0 else f"models/{arch}.pth"
        shutil.copyfile(cand / f"{arch}.pth", ROOT / dst)
        members.append({"arch": arch, "path": dst})
    cfg = {"arch": args.archs[0], "members": members, "threshold": thr, "tta_hflip": True,
           "preprocess": "equalize", "image_size": args.size, "resize_short_side": short_side(args.size),
           "selected_by": "ensemble of all candidates; threshold = max accuracy on a 30% patient-level "
                          "calibration split of the test set; metrics reported on the other 70%"}
    (ROOT / "model_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "test_split.json").write_text(json.dumps(
        {"seed": SEED + 1, "calib_frac": args.calib_frac,
         "calibration": [p for p, c in zip(t_paths, calib) if c],
         "holdout": [p for p, h in zip(t_paths, hold) if h]}, indent=1), encoding="utf-8")
    (ROOT / "reports" / "training_v3.json").write_text(json.dumps(
        {"size": args.size, "members": args.archs, "n_val": int(va.sum()),
         "n_calibration": int(calib.sum()), "n_holdout": int(hold.sum()), "results": report}, indent=2), encoding="utf-8")
    print("saved model_config.json, reports/test_split.json, reports/training_v3.json")


if __name__ == "__main__":
    main()
