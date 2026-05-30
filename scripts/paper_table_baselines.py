#!/usr/bin/env python3
"""Generate baseline comparison table (Dice/Sens/Spec/Prec) for multiple models."""

import argparse
from pathlib import Path
import time

import numpy as np
import nibabel as nib
from joblib import Parallel, delayed


REGIONS = ["tc", "wt", "et"]


def gt_regions(seg):
    seg = seg.astype(np.uint8)
    return {"tc": (seg == 1) | (seg == 4), "wt": seg > 0, "et": seg == 4}


def load_prob(p):
    z = np.load(p)
    pr = z["prob"]
    if pr.dtype != np.uint8:
        pr = (pr * 255 if pr.max() <= 1 else pr).clip(0, 255).astype(np.uint8)
    return pr


def metrics_binary(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    tn = (~pred & ~gt).sum()
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8) if (tp + fp + fn) > 0 else 1.0
    sens = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 1.0
    spec = tn / (tn + fp + 1e-8) if (tn + fp) > 0 else 1.0
    prec = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 1.0
    return float(dice), float(sens), float(spec), float(prec)


def process_single(cid, prob_dir, ref_dir, threshold):
    try:
        prob = load_prob(prob_dir / f"{cid}.npz")
    except Exception:
        return None
    gt_path = ref_dir / cid / f"{cid}_seg.nii.gz"
    if not gt_path.exists():
        return None
    seg = np.asarray(nib.load(str(gt_path)).dataobj)
    gt = gt_regions(seg)
    thr_u8 = int(round(threshold * 255))

    row = {"case": cid}
    for i, r in enumerate(REGIONS):
        pred = prob[i] >= thr_u8
        d, se, sp, p = metrics_binary(pred, gt[r])
        row[f"dice_{r}"] = d
        row[f"sens_{r}"] = se
        row[f"spec_{r}"] = sp
        row[f"prec_{r}"] = p
    return row


def aggregate_rows(rows, model_name):
    rows = [r for r in rows if r]
    out = []
    for r in REGIONS:
        d = np.array([row[f"dice_{r}"] for row in rows])
        se = np.array([row[f"sens_{r}"] for row in rows])
        sp = np.array([row[f"spec_{r}"] for row in rows])
        pr = np.array([row[f"prec_{r}"] for row in rows])
        out.append({
            "model": model_name,
            "region": r.upper(),
            "dice": f"{d.mean():.4f}±{d.std():.3f}",
            "sensitivity": f"{se.mean():.4f}±{se.std():.3f}",
            "specificity": f"{sp.mean():.4f}±{sp.std():.3f}",
            "precision": f"{pr.mean():.4f}±{pr.std():.3f}",
        })

    d_m = np.mean([[row[f"dice_{r}"] for r in REGIONS] for row in rows], axis=1)
    se_m = np.mean([[row[f"sens_{r}"] for r in REGIONS] for row in rows], axis=1)
    sp_m = np.mean([[row[f"spec_{r}"] for r in REGIONS] for row in rows], axis=1)
    pr_m = np.mean([[row[f"prec_{r}"] for r in REGIONS] for row in rows], axis=1)
    out.append({
        "model": model_name,
        "region": "MEAN",
        "dice": f"{d_m.mean():.4f}±{d_m.std():.3f}",
        "sensitivity": f"{se_m.mean():.4f}±{se_m.std():.3f}",
        "specificity": f"{sp_m.mean():.4f}±{sp_m.std():.3f}",
        "precision": f"{pr_m.mean():.4f}±{pr_m.std():.3f}",
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob_dirs", nargs="+", required=True)
    ap.add_argument("--model_names", nargs="+", default=None)
    ap.add_argument("--ref_dir", default="data/BraTS2021_TrainingData")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out_csv", default="paper_data/csv/paper_table_baselines.csv")
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()

    prob_dirs = [Path(p) for p in args.prob_dirs]
    if args.model_names is None:
        args.model_names = [p.name.replace("_oof", "") for p in prob_dirs]

    ref_dir = Path(args.ref_dir)
    results = []

    for model_name, prob_dir in zip(args.model_names, prob_dirs):
        cases = sorted([f.stem for f in prob_dir.glob("*.npz") if not f.name.startswith("_tmp_")])
        t0 = time.time()
        rows = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(process_single)(c, prob_dir, ref_dir, args.threshold) for c in cases
        )
        rows = [r for r in rows if r]
        print(f"{model_name}: {len(rows)} cases, {time.time() - t0:.0f}s")
        results.extend(aggregate_rows(rows, model_name))

    import pandas as pd

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
