#!/usr/bin/env python3
"""
PARALLEL Grid Search for BraTS ensemble.
Use joblib to parallelize per-case computation across all CPUs.

Speedup: ~8-10x vs single-core grid_search.py
"""

import argparse
import csv
import json
import time
from pathlib import Path
from functools import partial

import numpy as np
import nibabel as nib
from joblib import Parallel, delayed
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
    pred = pred.astype(bool); gt = gt.astype(bool)
    ps = pred.sum(); gs = gt.sum()
    if ps == 0 and gs == 0: return 1.0
    if ps == 0 or gs == 0:  return 0.0
    return float(2.0 * np.logical_and(pred, gt).sum() / (ps + gs))


def remove_small_components(mask, min_size):
    if min_size <= 0 or not mask.any(): return mask
    s = generate_binary_structure(3, 2)
    lab, n = label(mask, structure=s)
    if n == 0: return mask
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return (sizes >= min_size)[lab]


def load_prob_uint8(path):
    z = np.load(path); p = z["prob"]
    if p.dtype != np.uint8:
        if p.dtype in (np.float32, np.float64, np.float16):
            p = (np.clip(p * 255.0 if p.max() <= 1.0 else p, 0, 255)).astype(np.uint8)
        else:
            p = p.astype(np.uint8)
    return p


def ensemble_uint8(probs, weights):
    out = np.zeros_like(probs[0], dtype=np.float32)
    for p, w in zip(probs, weights):
        out += p.astype(np.float32) * float(w)
    return np.clip(out, 0, 255).astype(np.uint8)


def process_case_phase1(cid, prob_dirs, ref_dir, alphas):
    """Phase 1: 加载 case + 算所有 alpha 的 dice (默认 thr=0.5)."""
    try:
        probs = [load_prob_uint8(d / f"{cid}.npz") for d in prob_dirs]
    except Exception:
        return None
    gt_path = ref_dir / cid / f"{cid}_seg.nii.gz"
    if not gt_path.exists():
        return None
    seg = np.asarray(nib.load(str(gt_path)).dataobj)
    gt = gt_regions(seg)
    
    result = {}
    for alpha in alphas:
        if len(probs) == 2:
            w = [alpha, 1.0 - alpha]
        else:
            w = [1.0 / len(probs)] * len(probs)
        ens = ensemble_uint8(probs, w)
        per = {}
        for i, r in enumerate(REGIONS):
            pred = ens[i] >= 128
            per[r] = dice_binary(pred, gt[r])
        per["mean"] = float(np.mean([per[r] for r in REGIONS]))
        result[alpha] = per
    return cid, result


def process_case_phase2(cid, prob_dirs, ref_dir, weights, thresholds):
    """Phase 2: per-region threshold sweep with fixed alpha."""
    try:
        probs = [load_prob_uint8(d / f"{cid}.npz") for d in prob_dirs]
    except Exception:
        return None
    gt_path = ref_dir / cid / f"{cid}_seg.nii.gz"
    if not gt_path.exists():
        return None
    seg = np.asarray(nib.load(str(gt_path)).dataobj)
    gt = gt_regions(seg)
    
    ens = ensemble_uint8(probs, weights)
    result = {r: {} for r in REGIONS}
    for i, r in enumerate(REGIONS):
        for t in thresholds:
            pred = ens[i] >= int(round(t * 255))
            result[r][t] = dice_binary(pred, gt[r])
    return cid, result


def process_case_phase3(cid, prob_dirs, ref_dir, weights, thresholds, min_sizes):
    """Phase 3: per-region min component size."""
    try:
        probs = [load_prob_uint8(d / f"{cid}.npz") for d in prob_dirs]
    except Exception:
        return None
    gt_path = ref_dir / cid / f"{cid}_seg.nii.gz"
    if not gt_path.exists():
        return None
    seg = np.asarray(nib.load(str(gt_path)).dataobj)
    gt = gt_regions(seg)
    
    ens = ensemble_uint8(probs, weights)
    result = {r: {} for r in REGIONS}
    for i, r in enumerate(REGIONS):
        base = ens[i] >= int(round(thresholds[r] * 255))
        for ms in min_sizes:
            cleaned = remove_small_components(base, ms)
            result[r][ms] = dice_binary(cleaned, gt[r])
    return cid, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob_dirs", "--oof_dirs", dest="prob_dirs", nargs="+", required=True)
    ap.add_argument("--ref_dir", "--data_root", dest="ref_dir", default="data/BraTS2021_TrainingData")
    ap.add_argument("--folds_json", default=None)
    ap.add_argument("--model_names", nargs="+", default=None)
    ap.add_argument("--out_dir", default="work_dir/grid_search")
    ap.add_argument("--max_cases", type=int, default=0)
    ap.add_argument("--n_jobs", type=int, default=-1, help="-1 = all cores")
    ap.add_argument("--skip_phase3", action="store_true", help="跳过 min_size 搜索 (节省时间)")
    ap.add_argument("--alphas", nargs="*", type=float,
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--thresholds", nargs="*", type=float,
                    default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    ap.add_argument("--min_sizes", nargs="*", type=int,
                    default=[0, 50, 100, 200, 500, 1000])
    args = ap.parse_args()

    if args.model_names is None:
        args.model_names = [Path(d).name.replace("_oof", "") for d in args.prob_dirs]

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = Path(args.ref_dir)
    prob_dirs = [Path(d) for d in args.prob_dirs]

    # List intersection
    sets = [set(f.stem for f in d.glob("*.npz") if not f.name.startswith("_tmp_")) for d in prob_dirs]
    cases = sorted(sets[0].intersection(*sets[1:]) if len(sets) > 1 else sets[0])
    if args.max_cases > 0:
        cases = cases[:args.max_cases]
    print(f"Total cases: {len(cases)}")
    print(f"Workers:     {args.n_jobs if args.n_jobs > 0 else 'all cores'}")
    print()

    # ========== Phase 1 ==========
    print("=" * 70)
    print("Phase 1: alpha sweep (parallel)")
    print("=" * 70)
    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(process_case_phase1)(c, prob_dirs, ref_dir, args.alphas) for c in cases
    )
    results = [r for r in results if r is not None]
    print(f"  processed {len(results)}/{len(cases)} cases")

    # Aggregate
    phase1_acc = {a: {r: [] for r in REGIONS + ["mean"]} for a in args.alphas}
    for cid, per_alpha in results:
        for a in args.alphas:
            for k in REGIONS + ["mean"]:
                phase1_acc[a][k].append(per_alpha[a][k])

    phase1_summary = []
    for a in args.alphas:
        row = {"alpha": a}
        for k in REGIONS + ["mean"]:
            row[f"dice_{k}"] = float(np.mean(phase1_acc[a][k])) if phase1_acc[a][k] else 0.0
        phase1_summary.append(row)

    with (out_dir / "grid_phase1.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alpha","dice_tc","dice_wt","dice_et","dice_mean"])
        w.writeheader(); w.writerows(phase1_summary)

    best_alpha_row = max(phase1_summary, key=lambda r: r["dice_mean"])
    best_alpha = best_alpha_row["alpha"]
    print(f"\n✓ Phase 1 best alpha = {best_alpha:.2f}  dice_mean = {best_alpha_row['dice_mean']:.4f}")
    print(f"  TC={best_alpha_row['dice_tc']:.4f} WT={best_alpha_row['dice_wt']:.4f} ET={best_alpha_row['dice_et']:.4f}")
    print(f"  elapsed: {time.time()-t0:.0f}s")
    print()

    # ========== Phase 2 ==========
    print("=" * 70)
    print(f"Phase 2: per-region threshold (alpha={best_alpha}, parallel)")
    print("=" * 70)
    t0 = time.time()
    weights = [best_alpha, 1.0 - best_alpha] if len(prob_dirs) == 2 else [1.0/len(prob_dirs)]*len(prob_dirs)
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(process_case_phase2)(c, prob_dirs, ref_dir, weights, args.thresholds) for c in cases
    )
    results = [r for r in results if r is not None]

    phase2_acc = {r: {t: [] for t in args.thresholds} for r in REGIONS}
    for cid, per in results:
        for r in REGIONS:
            for t in args.thresholds:
                phase2_acc[r][t].append(per[r][t])

    p2_rows = []
    best_thresh = {}
    for r in REGIONS:
        scored = {t: float(np.mean(phase2_acc[r][t])) for t in args.thresholds}
        best_thresh[r] = max(scored, key=scored.get)
        for t in args.thresholds:
            p2_rows.append({"region": r, "threshold": t, "dice": scored[t]})

    with (out_dir / "grid_phase2.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["region","threshold","dice"])
        w.writeheader(); w.writerows(p2_rows)

    print(f"\n✓ Phase 2 best thresholds:")
    for r in REGIONS:
        print(f"  {r}: {best_thresh[r]:.2f}  dice={np.mean(phase2_acc[r][best_thresh[r]]):.4f}")
    print(f"  elapsed: {time.time()-t0:.0f}s")
    print()

    # ========== Phase 3 (optional) ==========
    if args.skip_phase3:
        print("Phase 3: SKIPPED (--skip_phase3)")
        best_ms = {r: 0 for r in REGIONS}
        final_dice = {r: float(np.mean(phase2_acc[r][best_thresh[r]])) for r in REGIONS}
    else:
        print("=" * 70)
        print("Phase 3: per-region min component size (parallel)")
        print("=" * 70)
        t0 = time.time()
        results = Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(process_case_phase3)(c, prob_dirs, ref_dir, weights, best_thresh, args.min_sizes) for c in cases
        )
        results = [r for r in results if r is not None]

        phase3_acc = {r: {ms: [] for ms in args.min_sizes} for r in REGIONS}
        for cid, per in results:
            for r in REGIONS:
                for ms in args.min_sizes:
                    phase3_acc[r][ms].append(per[r][ms])

        p3_rows = []; best_ms = {}
        for r in REGIONS:
            scored = {ms: float(np.mean(phase3_acc[r][ms])) for ms in args.min_sizes}
            best_ms[r] = max(scored, key=scored.get)
            for ms in args.min_sizes:
                p3_rows.append({"region": r, "min_size": ms, "dice": scored[ms]})

        with (out_dir / "grid_phase3.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["region","min_size","dice"])
            w.writeheader(); w.writerows(p3_rows)

        print(f"\n✓ Phase 3 best min sizes:")
        for r in REGIONS:
            print(f"  {r}: {best_ms[r]:>5d}  dice={np.mean(phase3_acc[r][best_ms[r]]):.4f}")
        print(f"  elapsed: {time.time()-t0:.0f}s")
        final_dice = {r: float(np.mean(phase3_acc[r][best_ms[r]])) for r in REGIONS}

    final_dice["mean"] = float(np.mean([final_dice[r] for r in REGIONS]))

    best_config = {
        "models": dict(zip(args.model_names, [str(d) for d in prob_dirs])),
        "weights": dict(zip(args.model_names, weights)),
        "alpha": best_alpha,
        "thresholds": {r: float(best_thresh[r]) for r in REGIONS},
        "min_sizes": {r: int(best_ms[r]) for r in REGIONS},
        "final_dice": final_dice,
        "n_cases": len(cases),
        "skip_phase3": args.skip_phase3,
    }
    with (out_dir / "best_config.json").open("w") as f:
        json.dump(best_config, f, indent=2)

    summary = "\n".join([
        "="*70,
        "GRID SEARCH FINAL RESULTS",
        "="*70,
        f"N cases:    {len(cases)}",
        f"Best alpha: {best_alpha:.2f}",
        f"Weights:    {dict(zip(args.model_names, weights))}",
        f"Thresholds: {best_thresh}",
        f"Min sizes:  {best_ms}",
        f"Final Dice: TC={final_dice['tc']:.4f} WT={final_dice['wt']:.4f} "
        f"ET={final_dice['et']:.4f} mean={final_dice['mean']:.4f}",
        "="*70,
    ])
    (out_dir / "summary.txt").write_text(summary + "\n")
    print()
    print(summary)


if __name__ == "__main__":
    main()
