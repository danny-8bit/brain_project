#!/usr/bin/env python3
"""ROC/AUC evaluation for BraTS region-based segmentation.

Computes voxel-level and case-level ROC curves for TC/WT/ET using OOF prob maps.
"""

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from sklearn.metrics import roc_curve, auc


REGIONS = ["tc", "wt", "et"]


def gt_regions(seg):
    seg = seg.astype(np.uint8)
    return {"tc": (seg == 1) | (seg == 4), "wt": seg > 0, "et": seg == 4}


def load_prob(npz_path):
    z = np.load(npz_path)
    prob = z["prob"]
    if prob.dtype != np.uint8:
        prob = (prob * 255 if prob.max() <= 1 else prob).clip(0, 255).astype(np.uint8)
    return prob.astype(np.float32) / 255.0


def sample_voxels(prob, gt_mask, max_voxels, rng):
    flat_prob = prob.reshape(-1)
    flat_gt = gt_mask.reshape(-1)
    n = flat_gt.size
    if n <= max_voxels:
        return flat_prob, flat_gt.astype(np.uint8)
    idx = rng.choice(n, size=max_voxels, replace=False)
    return flat_prob[idx], flat_gt[idx].astype(np.uint8)


def case_score(prob, mode="max"):
    flat = prob.reshape(-1)
    if mode == "mean":
        return float(flat.mean())
    if mode == "p99":
        return float(np.percentile(flat, 99))
    return float(flat.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob_dirs", nargs="+", required=True)
    ap.add_argument("--model_names", nargs="+", default=None)
    ap.add_argument("--data_root", default="data/BraTS2021_TrainingData")
    ap.add_argument("--out_dir", default="paper_figs")
    ap.add_argument("--out_csv", default="paper_data/csv/roc_auc.csv")
    ap.add_argument("--voxels_per_case", type=int, default=50000)
    ap.add_argument("--case_score", choices=["max", "mean", "p99"], default="max")
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    prob_dirs = [Path(p) for p in args.prob_dirs]
    if args.model_names is None:
        args.model_names = [p.name.replace("_oof", "") for p in prob_dirs]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    ref_dir = Path(args.data_root)
    sets = [set(f.stem for f in d.glob("*.npz") if not f.name.startswith("_tmp_")) for d in prob_dirs]
    cases = sorted(sets[0].intersection(*sets[1:]) if len(sets) > 1 else sets[0])
    if not cases:
        raise RuntimeError("No overlapping cases found in prob_dirs")

    rng = np.random.default_rng(args.seed)

    # Collect ROC data
    roc_data = {"voxel": {}, "case": {}}
    for model_name, prob_dir in zip(args.model_names, prob_dirs):
        roc_data["voxel"][model_name] = {r: {"y": [], "p": []} for r in REGIONS}
        roc_data["case"][model_name] = {r: {"y": [], "p": []} for r in REGIONS}

        for cid in cases:
            prob = load_prob(prob_dir / f"{cid}.npz")
            gt_path = ref_dir / cid / f"{cid}_seg.nii.gz"
            if not gt_path.exists():
                continue
            seg = np.asarray(nib.load(str(gt_path)).dataobj)
            gt = gt_regions(seg)

            for i, r in enumerate(REGIONS):
                p = prob[i]
                g = gt[r]

                sp, sg = sample_voxels(p, g, args.voxels_per_case, rng)
                roc_data["voxel"][model_name][r]["p"].append(sp)
                roc_data["voxel"][model_name][r]["y"].append(sg)

                roc_data["case"][model_name][r]["p"].append(case_score(p, args.case_score))
                roc_data["case"][model_name][r]["y"].append(int(g.any()))

    # Compute ROC/AUC and save CSV
    rows = []
    for mode in ["voxel", "case"]:
        for model_name in args.model_names:
            for r in REGIONS:
                y = np.concatenate(roc_data[mode][model_name][r]["y"], axis=0)
                p = np.concatenate(roc_data[mode][model_name][r]["p"], axis=0)
                fpr, tpr, _ = roc_curve(y, p)
                roc_auc = auc(fpr, tpr)
                rows.append({
                    "mode": mode,
                    "model": model_name,
                    "region": r.upper(),
                    "auc": float(roc_auc),
                })

                np.savez_compressed(
                    out_dir / f"roc_{mode}_{model_name}_{r}.npz",
                    fpr=fpr,
                    tpr=tpr,
                    auc=roc_auc,
                )

    # Plot
    import matplotlib.pyplot as plt

    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for mode in ["voxel", "case"]:
        for r in REGIONS:
            plt.figure(figsize=(5, 4))
            for idx, model_name in enumerate(args.model_names):
                data = np.load(out_dir / f"roc_{mode}_{model_name}_{r}.npz")
                fpr = data["fpr"]
                tpr = data["tpr"]
                roc_auc = float(data["auc"])
                plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})", color=palette[idx % len(palette)])
            plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC ({mode}, {r.upper()})")
            plt.legend(fontsize=8)
            plt.tight_layout()
            out_path = out_dir / f"roc_{mode}_{r}.png"
            plt.savefig(out_path, dpi=200)
            plt.close()

    import pandas as pd

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved AUC table: {out_csv}")
    print(f"Saved ROC figures: {out_dir}")


if __name__ == "__main__":
    main()
