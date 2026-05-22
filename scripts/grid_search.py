#!/usr/bin/env python3
"""
Ensemble + Grid Search for BraTS 5-fold OOF probabilities.

Compatible with run_all.sh invocation:
  python scripts/grid_search.py \
      --prob_dirs DIR1 DIR2 [...] \
      --ref_dir data/BraTS2021_TrainingData \
      --folds_json data/folds.json    # optional, ignored

Also supports manual invocation:
  python scripts/grid_search.py \
      --oof_dirs DIR1 DIR2 \
      --data_root data/BraTS2021_TrainingData \
      --out_dir work_dir/grid_search

Phased search:
  Phase 1: ensemble weight alpha (with default threshold 0.5)
  Phase 2: per-region threshold (with best alpha)
  Phase 3: per-region min component size (with best alpha + thresholds)
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure


REGIONS = ["tc", "wt", "et"]


def gt_regions(seg):
    seg = seg.astype(np.uint8)
    return {
        "tc": np.logical_or(seg == 1, seg == 4),
        "wt": seg > 0,
        "et": seg == 4,
    }


def dice_binary(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    ps = pred.sum()
    gs = gt.sum()
    if ps == 0 and gs == 0:
        return 1.0
    if ps == 0 or gs == 0:
        return 0.0
    inter = np.logical_and(pred, gt).sum()
    return float(2.0 * inter / (ps + gs))


def remove_small_components(mask, min_size):
    if min_size <= 0 or not mask.any():
        return mask
    structure = generate_binary_structure(3, 2)
    labeled, n = label(mask, structure=structure)
    if n == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    keep = sizes >= min_size
    return keep[labeled]


def load_prob_uint8(path):
    z = np.load(path)
    p = z["prob"]
    if p.dtype != np.uint8:
        if p.dtype in (np.float32, np.float64, np.float16):
            if p.max() <= 1.0:
                p = np.clip(p * 255.0, 0, 255).astype(np.uint8)
            else:
                p = np.clip(p, 0, 255).astype(np.uint8)
        else:
            p = p.astype(np.uint8)
    if p.shape[0] != 3:
        raise ValueError(f"{path}: expected (3,H,W,D), got {p.shape}")
    return p


def list_cases(prob_dirs):
    sets = []
    for d in prob_dirs:
        files = sorted(Path(d).glob("*.npz"))
        cases = set(f.stem for f in files)
        sets.append(cases)
    common = sets[0].intersection(*sets[1:]) if len(sets) > 1 else sets[0]
    return sorted(common)


def ensemble_uint8(probs_list, weights):
    out = np.zeros_like(probs_list[0], dtype=np.float32)
    for p, w in zip(probs_list, weights):
        out += p.astype(np.float32) * float(w)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    # Compatible with both naming conventions
    ap.add_argument("--prob_dirs", "--oof_dirs", dest="prob_dirs",
                    nargs="+", required=True,
                    help="Probability directories (one per model)")
    ap.add_argument("--ref_dir", "--data_root", dest="ref_dir",
                    default="data/BraTS2021_TrainingData",
                    help="GT data root")
    ap.add_argument("--folds_json", default=None,
                    help="Optional fold split JSON (not required for OOF)")
    ap.add_argument("--model_names", nargs="+", default=None,
                    help="Model names (auto-inferred from dir names if not given)")
    ap.add_argument("--out_dir", default="work_dir/grid_search",
                    help="Output directory")
    ap.add_argument("--max_cases", type=int, default=0)
    ap.add_argument("--alphas", nargs="*", type=float,
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--thresholds", nargs="*", type=float,
                    default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    ap.add_argument("--min_sizes", nargs="*", type=int,
                    default=[0, 50, 100, 200, 500, 1000])
    args = ap.parse_args()

    # Auto-infer model names from dir basenames
    if args.model_names is None:
        args.model_names = [Path(d).name.replace("_oof", "") for d in args.prob_dirs]
    if len(args.model_names) != len(args.prob_dirs):
        raise SystemExit("--model_names length must match --prob_dirs")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = Path(args.ref_dir)
    prob_dirs = [Path(d) for d in args.prob_dirs]

    # Header
    print("=" * 70)
    print("Grid Search Configuration")
    print("=" * 70)
    for n, d in zip(args.model_names, prob_dirs):
        print(f"  {n:12s} -> {d}")
    print(f"  ref_dir:    {ref_dir}")
    print(f"  out_dir:    {out_dir}")
    print(f"  folds_json: {args.folds_json or '(not used)'}")
    print(f"  alphas:     {args.alphas}")
    print(f"  thresholds: {args.thresholds}")
    print(f"  min_sizes:  {args.min_sizes}")
    print()

    cases = list_cases(prob_dirs)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    print(f"Total cases (intersection across models): {len(cases)}")
    print()

    # ===== Phase 1: alpha sweep =====
    print("=" * 70)
    print("Phase 1: alpha sweep (threshold=0.5, no post-process)")
    print("=" * 70)
    t0 = time.time()

    phase1_acc = {a: {r: [] for r in REGIONS + ["mean"]} for a in args.alphas}

    for ci, case_id in enumerate(cases, 1):
        try:
            probs = [load_prob_uint8(d / f"{case_id}.npz") for d in prob_dirs]
        except Exception as e:
            print(f"  [skip] {case_id}: load fail ({type(e).__name__}: {e})")
            continue

        gt_path = ref_dir / case_id / f"{case_id}_seg.nii.gz"
        if not gt_path.exists():
            print(f"  [skip] {case_id}: GT not found")
            continue
        seg = np.asarray(nib.load(str(gt_path)).dataobj)
        gt = gt_regions(seg)

        for alpha in args.alphas:
            if len(probs) == 2:
                w = [alpha, 1.0 - alpha]
            else:
                w = [1.0 / len(probs)] * len(probs)
            ens = ensemble_uint8(probs, w)
            d_per = {}
            for i, r in enumerate(REGIONS):
                pred = ens[i] >= 128
                d_per[r] = dice_binary(pred, gt[r])
            d_per["mean"] = float(np.mean([d_per[r] for r in REGIONS]))
            for k in REGIONS + ["mean"]:
                phase1_acc[alpha][k].append(d_per[k])

        if ci % 50 == 0 or ci == len(cases):
            print(f"  [{ci}/{len(cases)}] {case_id}  ({time.time()-t0:.0f}s)")

    phase1_summary = []
    for a in args.alphas:
        row = {"alpha": a}
        for k in REGIONS + ["mean"]:
            vals = phase1_acc[a][k]
            row[f"dice_{k}"] = float(np.mean(vals)) if vals else 0.0
        phase1_summary.append(row)

    p1_csv = out_dir / "grid_phase1.csv"
    with p1_csv.open("w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=["alpha", "dice_tc", "dice_wt", "dice_et", "dice_mean"])
        w_.writeheader()
        w_.writerows(phase1_summary)

    best_alpha_row = max(phase1_summary, key=lambda r: r["dice_mean"])
    best_alpha = best_alpha_row["alpha"]

    print()
    print(f"Phase 1 best alpha (weight for {args.model_names[0]}): {best_alpha:.2f}")
    print(f"  dice_mean = {best_alpha_row['dice_mean']:.4f}")
    print(f"  TC={best_alpha_row['dice_tc']:.4f} "
          f"WT={best_alpha_row['dice_wt']:.4f} "
          f"ET={best_alpha_row['dice_et']:.4f}")
    print(f"  saved: {p1_csv}")
    print(f"  elapsed: {time.time()-t0:.1f}s")
    print()

    # ===== Phase 2: per-region thresholds =====
    print("=" * 70)
    print(f"Phase 2: per-region threshold (alpha={best_alpha:.2f})")
    print("=" * 70)
    t0 = time.time()

    if len(prob_dirs) == 2:
        weights = [best_alpha, 1.0 - best_alpha]
    else:
        weights = [1.0 / len(prob_dirs)] * len(prob_dirs)

    phase2_acc = {r: {t: [] for t in args.thresholds} for r in REGIONS}

    for ci, case_id in enumerate(cases, 1):
        try:
            probs = [load_prob_uint8(d / f"{case_id}.npz") for d in prob_dirs]
        except Exception:
            continue
        gt_path = ref_dir / case_id / f"{case_id}_seg.nii.gz"
        if not gt_path.exists():
            continue
        seg = np.asarray(nib.load(str(gt_path)).dataobj)
        gt = gt_regions(seg)

        ens = ensemble_uint8(probs, weights)
        for i, r in enumerate(REGIONS):
            for th in args.thresholds:
                pred = ens[i] >= int(round(th * 255))
                phase2_acc[r][th].append(dice_binary(pred, gt[r]))

        if ci % 100 == 0 or ci == len(cases):
            print(f"  [{ci}/{len(cases)}] {case_id}  ({time.time()-t0:.0f}s)")

    phase2_rows = []
    best_thresh = {}
    for r in REGIONS:
        scored = {t: float(np.mean(phase2_acc[r][t])) if phase2_acc[r][t] else 0.0
                  for t in args.thresholds}
        best_thresh[r] = max(scored, key=scored.get)
        for t in args.thresholds:
            phase2_rows.append({"region": r, "threshold": t, "dice": scored[t]})

    p2_csv = out_dir / "grid_phase2.csv"
    with p2_csv.open("w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=["region", "threshold", "dice"])
        w_.writeheader()
        w_.writerows(phase2_rows)

    print()
    print("Phase 2 best thresholds:")
    for r in REGIONS:
        print(f"  {r}: {best_thresh[r]:.2f}  "
              f"dice={np.mean(phase2_acc[r][best_thresh[r]]):.4f}")
    print(f"  saved: {p2_csv}")
    print(f"  elapsed: {time.time()-t0:.1f}s")
    print()

    # ===== Phase 3: per-region min component size =====
    print("=" * 70)
    print(f"Phase 3: per-region min component size")
    print("=" * 70)
    t0 = time.time()

    phase3_acc = {r: {ms: [] for ms in args.min_sizes} for r in REGIONS}

    for ci, case_id in enumerate(cases, 1):
        try:
            probs = [load_prob_uint8(d / f"{case_id}.npz") for d in prob_dirs]
        except Exception:
            continue
        gt_path = ref_dir / case_id / f"{case_id}_seg.nii.gz"
        if not gt_path.exists():
            continue
        seg = np.asarray(nib.load(str(gt_path)).dataobj)
        gt = gt_regions(seg)

        ens = ensemble_uint8(probs, weights)
        for i, r in enumerate(REGIONS):
            base = ens[i] >= int(round(best_thresh[r] * 255))
            for ms in args.min_sizes:
                cleaned = remove_small_components(base, ms)
                phase3_acc[r][ms].append(dice_binary(cleaned, gt[r]))

        if ci % 100 == 0 or ci == len(cases):
            print(f"  [{ci}/{len(cases)}] {case_id}  ({time.time()-t0:.0f}s)")

    phase3_rows = []
    best_ms = {}
    for r in REGIONS:
        scored = {ms: float(np.mean(phase3_acc[r][ms])) if phase3_acc[r][ms] else 0.0
                  for ms in args.min_sizes}
        best_ms[r] = max(scored, key=scored.get)
        for ms in args.min_sizes:
            phase3_rows.append({"region": r, "min_size": ms, "dice": scored[ms]})

    p3_csv = out_dir / "grid_phase3.csv"
    with p3_csv.open("w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=["region", "min_size", "dice"])
        w_.writeheader()
        w_.writerows(phase3_rows)

    print()
    print("Phase 3 best min sizes:")
    for r in REGIONS:
        print(f"  {r}: {best_ms[r]:>5d}  dice={np.mean(phase3_acc[r][best_ms[r]]):.4f}")
    print(f"  saved: {p3_csv}")
    print(f"  elapsed: {time.time()-t0:.1f}s")
    print()

    # ===== Final =====
    final_dice = {r: float(np.mean(phase3_acc[r][best_ms[r]])) for r in REGIONS}
    final_dice["mean"] = float(np.mean(list(final_dice.values())))

    best_config = {
        "models": dict(zip(args.model_names, [str(d) for d in prob_dirs])),
        "weights": dict(zip(args.model_names, weights)),
        "alpha": best_alpha,
        "thresholds": {r: float(best_thresh[r]) for r in REGIONS},
        "min_sizes": {r: int(best_ms[r]) for r in REGIONS},
        "final_dice": final_dice,
        "n_cases": len(cases),
    }

    with (out_dir / "best_config.json").open("w") as f:
        json.dump(best_config, f, indent=2)

    summary = []
    summary.append("=" * 70)
    summary.append("GRID SEARCH FINAL RESULTS")
    summary.append("=" * 70)
    summary.append(f"N cases:    {len(cases)}")
    summary.append(f"Models:     {', '.join(args.model_names)}")
    summary.append("")
    summary.append("Best Ensemble Weights:")
    for m, w in best_config["weights"].items():
        summary.append(f"  {m:12s}: {w:.2f}")
    summary.append("")
    summary.append("Best Thresholds:")
    for r in REGIONS:
        summary.append(f"  {r}: {best_thresh[r]:.2f}")
    summary.append("")
    summary.append("Best Min Component Sizes:")
    for r in REGIONS:
        summary.append(f"  {r}: {best_ms[r]}")
    summary.append("")
    summary.append("Final Dice:")
    for r in REGIONS:
        summary.append(f"  {r}:   {final_dice[r]:.4f}")
    summary.append(f"  mean: {final_dice['mean']:.4f}")
    summary.append("=" * 70)

    summary_text = "\n".join(summary)
    with (out_dir / "summary.txt").open("w") as f:
        f.write(summary_text + "\n")

    print()
    print(summary_text)
    print()
    print(f"All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
