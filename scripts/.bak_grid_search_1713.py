#!/usr/bin/env python3
"""
Parallel coarse-to-fine grid search for BraTS ensemble.

Search evaluates the same postprocessing path used by final ensemble output:
weighted probability fusion -> postprocess_brats -> BraTS TC/WT/ET Dice.
"""

import argparse
import csv
import json
import time
from itertools import product
from pathlib import Path

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.postprocess import postprocess_brats


REGIONS = ["tc", "wt", "et"]


def gt_regions(seg):
    seg = seg.astype(np.uint8)
    return {
        "tc": np.logical_or(seg == 1, seg == 4),
        "wt": seg > 0,
        "et": seg == 4,
    }


def pred_regions(label):
    label = label.astype(np.uint8)
    return {
        "tc": np.logical_or(label == 1, label == 4),
        "wt": label > 0,
        "et": label == 4,
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
    return float(2.0 * np.logical_and(pred, gt).sum() / (ps + gs))


def dice_from_label(label, gt):
    pr = pred_regions(label)
    out = {r: dice_binary(pr[r], gt[r]) for r in REGIONS}
    out["mean"] = float(np.mean([out[r] for r in REGIONS]))
    return out


def load_prob_uint8(path):
    z = np.load(path)
    p = z["prob"]
    if p.dtype != np.uint8:
        if p.dtype in (np.float32, np.float64, np.float16):
            p = np.clip(p * 255.0 if p.max() <= 1.0 else p, 0, 255).astype(np.uint8)
        else:
            p = p.astype(np.uint8)
    if p.shape[0] != 3:
        raise ValueError(f"{path}: expected first dim 3, got {p.shape}")
    return p


def ensemble_uint8(probs, weights):
    out = np.zeros_like(probs[0], dtype=np.float32)
    for p, w in zip(probs, weights):
        out += p.astype(np.float32) * float(w)
    return np.clip(out, 0, 255).astype(np.uint8)



def largest_component_mask(mask):
    from scipy import ndimage
    labeled, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    max_label = int(np.argmax(sizes)) + 1
    return labeled == max_label


def remove_small_components(mask, min_size):
    from scipy import ndimage
    if min_size <= 0 or not mask.any():
        return mask
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return (sizes >= min_size)[labeled]


def masks_to_label(tc_mask, wt_mask, et_mask, min_sizes=None):
    out = np.zeros(tc_mask.shape, dtype=np.uint8)
    out[wt_mask] = 2
    out[tc_mask] = 1
    out[et_mask] = 4

    min_sizes = min_sizes or {}
    if min_sizes.get("tc", 0) > 0:
        tc_keep = remove_small_components(np.logical_or(out == 1, out == 4), min_sizes["tc"])
        out = np.where(tc_keep | (out == 2), out, 0).astype(np.uint8)
    if min_sizes.get("wt", 0) > 0:
        wt_keep = remove_small_components(out > 0, min_sizes["wt"])
        out = np.where(wt_keep, out, 0).astype(np.uint8)
    if min_sizes.get("et", 0) > 0:
        et_keep = remove_small_components(out == 4, min_sizes["et"])
        out = np.where((out == 4) & ~et_keep, 1, out).astype(np.uint8)

    keep = largest_component_mask(out > 0)
    return np.where(keep, out, 0).astype(np.uint8)


def score_masks(tc_mask, wt_mask, et_mask, gt, min_sizes=None):
    return dice_from_label(masks_to_label(tc_mask, wt_mask, et_mask, min_sizes), gt)

def score_fused_uint8(fused, gt, thresholds, min_sizes=None, et_threshold_voxels=200, et_min_prob=0.5):
    prob = fused.astype(np.float32) / 255.0
    label = postprocess_brats(
        prob,
        thresholds=thresholds,
        min_sizes=min_sizes,
        et_threshold_voxels=et_threshold_voxels,
        et_min_prob=et_min_prob,
    )
    return dice_from_label(label, gt)


def select_row(rows, selection_metric, mean_tolerance):
    best_mean = max(rows, key=lambda r: r["dice_mean"])
    if selection_metric == "et_constrained":
        min_mean = best_mean["dice_mean"] - mean_tolerance
        candidates = [r for r in rows if r["dice_mean"] >= min_mean]
        return max(candidates, key=lambda r: (r["dice_et"], r["dice_mean"])), best_mean
    return best_mean, best_mean


def refine_values(center, radius, step, lo=0.0, hi=1.0):
    n0 = int(round((center - radius) / step))
    n1 = int(round((center + radius) / step))
    vals = []
    for i in range(n0, n1 + 1):
        v = round(i * step, 6)
        if lo <= v <= hi:
            vals.append(v)
    return sorted(set(vals))


def aggregate_rows(per_case_results, key_names):
    acc = {}
    for _, per_key in per_case_results:
        for key, vals in per_key.items():
            acc.setdefault(key, {r: [] for r in REGIONS + ["mean"]})
            for r in REGIONS + ["mean"]:
                acc[key][r].append(vals[r])

    rows = []
    for key, vals in acc.items():
        row = {}
        if not isinstance(key, tuple):
            key = (key,)
        for name, value in zip(key_names, key):
            row[name] = value
        for r in REGIONS + ["mean"]:
            row[f"dice_{r}"] = float(np.mean(vals[r])) if vals[r] else 0.0
        rows.append(row)
    return rows


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def process_case_alpha(cid, prob_dirs, ref_dir, alphas, et_threshold_voxels, et_min_prob):
    try:
        probs = [load_prob_uint8(d / f"{cid}.npz") for d in prob_dirs]
    except Exception:
        return None
    gt_path = ref_dir / cid / f"{cid}_seg.nii.gz"
    if not gt_path.exists():
        return None
    gt = gt_regions(np.asarray(nib.load(str(gt_path)).dataobj))

    result = {}
    thresholds = {r: 0.5 for r in REGIONS}
    for alpha in alphas:
        weights = [alpha, 1.0 - alpha] if len(probs) == 2 else [1.0 / len(probs)] * len(probs)
        fused = ensemble_uint8(probs, weights)
        result[alpha] = score_fused_uint8(
            fused, gt, thresholds,
            et_threshold_voxels=et_threshold_voxels,
            et_min_prob=et_min_prob,
        )
    return cid, result


def process_case_thresholds(cid, prob_dirs, ref_dir, weights, triples, et_threshold_voxels, et_min_prob):
    """用 postprocess_brats 评估每个 threshold triple (与 final_eval 一致)."""
    try:
        probs = [load_prob_uint8(d / f"{cid}.npz") for d in prob_dirs]
    except Exception:
        return None
    gt_path = ref_dir / cid / f"{cid}_seg.nii.gz"
    if not gt_path.exists():
        return None
    gt = gt_regions(np.asarray(nib.load(str(gt_path)).dataobj))
    fused = ensemble_uint8(probs, weights).astype(np.float32) / 255.0

    result = {}
    for tc_t, wt_t, et_t in triples:
        thresholds = {"tc": tc_t, "wt": wt_t, "et": et_t}
        result[(tc_t, wt_t, et_t)] = score_fused_uint8(
            ensemble_uint8(probs, weights), gt, thresholds,
            et_threshold_voxels=et_threshold_voxels,
            et_min_prob=et_min_prob,
        )
    return cid, result


def process_case_min_sizes(cid, prob_dirs, ref_dir, weights, thresholds, size_triples, et_threshold_voxels, et_min_prob):
    try:
        probs = [load_prob_uint8(d / f"{cid}.npz") for d in prob_dirs]
    except Exception:
        return None
    gt_path = ref_dir / cid / f"{cid}_seg.nii.gz"
    if not gt_path.exists():
        return None
    gt = gt_regions(np.asarray(nib.load(str(gt_path)).dataobj))
    fused = ensemble_uint8(probs, weights).astype(np.float32) / 255.0
    tc_prob, wt_prob, et_prob = fused

    wt_mask = wt_prob >= thresholds["wt"]
    raw_tc = tc_prob >= thresholds["tc"]
    tc_mask = raw_tc & wt_mask
    raw_et = et_prob >= thresholds["et"]
    et_mask = raw_et & raw_tc
    et_voxels = int(et_mask.sum())
    if et_voxels < et_threshold_voxels and float(et_prob.max()) < et_min_prob:
        et_mask = np.zeros_like(et_mask)

    result = {}
    for tc_ms, wt_ms, et_ms in size_triples:
        min_sizes = {"tc": tc_ms, "wt": wt_ms, "et": et_ms}
        result[(tc_ms, wt_ms, et_ms)] = score_masks(tc_mask, wt_mask, et_mask, gt, min_sizes)
    return cid, result


def run_parallel(items, n_jobs):
    return [x for x in items if x is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob_dirs", "--oof_dirs", dest="prob_dirs", nargs="+", required=True)
    ap.add_argument("--ref_dir", "--data_root", dest="ref_dir", default="data/BraTS2021_TrainingData")
    ap.add_argument("--folds_json", default=None)
    ap.add_argument("--model_names", nargs="*", default=None)
    ap.add_argument("--out_dir", default="work_dir/grid_search")
    ap.add_argument("--max_cases", type=int, default=0)
    ap.add_argument("--n_jobs", type=int, default=-1, help="-1 = all cores")
    ap.add_argument("--skip_phase3", action="store_true", help="Skip min-size search")
    ap.add_argument("--alphas", nargs="*", type=float,
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--thresholds", nargs="*", type=float,
                    default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    ap.add_argument("--min_sizes", nargs="*", type=int,
                    default=[0, 50, 100, 200, 500, 1000])
    ap.add_argument("--selection_metric", choices=["mean", "et_constrained"], default="mean")
    ap.add_argument("--mean_tolerance", type=float, default=0.0)
    ap.add_argument("--no_refine", action="store_true", help="Disable coarse-to-fine refinement")
    ap.add_argument("--refine_alpha_radius", type=float, default=0.06)
    ap.add_argument("--refine_alpha_step", type=float, default=0.01)
    ap.add_argument("--refine_threshold_radius", type=float, default=0.04)
    ap.add_argument("--refine_threshold_step", type=float, default=0.01)
    ap.add_argument("--et_threshold_voxels", type=int, default=200)
    ap.add_argument("--et_min_prob", type=float, default=0.5)
    args = ap.parse_args()

    if args.model_names is None:
        args.model_names = [Path(d).name.replace("_oof", "") for d in args.prob_dirs]
    if len(args.model_names) != len(args.prob_dirs):
        raise SystemExit("--model_names length must match --prob_dirs")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = Path(args.ref_dir)
    prob_dirs = [Path(d) for d in args.prob_dirs]

    sets = [set(f.stem for f in d.glob("*.npz") if not f.name.startswith("_tmp_")) for d in prob_dirs]
    cases = sorted(sets[0].intersection(*sets[1:]) if len(sets) > 1 else sets[0])
    if args.max_cases > 0:
        cases = cases[:args.max_cases]

    print("=" * 70)
    print("Grid Search Configuration")
    print("=" * 70)
    print(f"cases:       {len(cases)}")
    print(f"workers:     {args.n_jobs if args.n_jobs > 0 else 'all cores'}")
    print(f"selection:   {args.selection_metric} (mean_tolerance={args.mean_tolerance})")
    print(f"postprocess: et_threshold_voxels={args.et_threshold_voxels}, et_min_prob={args.et_min_prob}")
    print(f"refine:      {not args.no_refine}")
    print()

    # Phase 1: alpha coarse search with final postprocess at threshold 0.5.
    print("=" * 70)
    print("Phase 1: alpha coarse search (postprocess-consistent)")
    print("=" * 70)
    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(process_case_alpha)(c, prob_dirs, ref_dir, args.alphas, args.et_threshold_voxels, args.et_min_prob)
        for c in cases
    )
    rows = aggregate_rows(run_parallel(results, args.n_jobs), ["alpha"])
    rows.sort(key=lambda r: r["alpha"])
    write_csv(out_dir / "grid_phase1.csv", rows, ["alpha", "dice_tc", "dice_wt", "dice_et", "dice_mean"])
    chosen_alpha_row, best_alpha_mean_row = select_row(rows, args.selection_metric, args.mean_tolerance)

    if not args.no_refine:
        refined_alphas = refine_values(chosen_alpha_row["alpha"], args.refine_alpha_radius, args.refine_alpha_step)
        refined_alphas = [a for a in refined_alphas if a not in set(args.alphas)]
        if refined_alphas:
            print(f"Refining alpha around {chosen_alpha_row['alpha']:.3f}: {refined_alphas}")
            refined = Parallel(n_jobs=args.n_jobs, verbose=10)(
                delayed(process_case_alpha)(c, prob_dirs, ref_dir, refined_alphas, args.et_threshold_voxels, args.et_min_prob)
                for c in cases
            )
            ref_rows = aggregate_rows(run_parallel(refined, args.n_jobs), ["alpha"])
            rows = sorted(rows + ref_rows, key=lambda r: r["alpha"])
            write_csv(out_dir / "grid_phase1_refined.csv", rows, ["alpha", "dice_tc", "dice_wt", "dice_et", "dice_mean"])
            chosen_alpha_row, best_alpha_mean_row = select_row(rows, args.selection_metric, args.mean_tolerance)

    best_alpha = float(chosen_alpha_row["alpha"])
    weights = [best_alpha, 1.0 - best_alpha] if len(prob_dirs) == 2 else [1.0 / len(prob_dirs)] * len(prob_dirs)
    print(f"selected alpha={best_alpha:.3f} mean={chosen_alpha_row['dice_mean']:.4f} ET={chosen_alpha_row['dice_et']:.4f}")
    print(f"phase1 best mean={best_alpha_mean_row['dice_mean']:.4f} ET={best_alpha_mean_row['dice_et']:.4f}")
    print(f"elapsed: {time.time() - t0:.0f}s")
    print()

    # Phase 2: threshold triples, evaluated through final postprocess.
    print("=" * 70)
    print("Phase 2: threshold triple coarse search (postprocess-consistent)")
    print("=" * 70)
    t0 = time.time()
    triples = list(product(args.thresholds, args.thresholds, args.thresholds))
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(process_case_thresholds)(c, prob_dirs, ref_dir, weights, triples, args.et_threshold_voxels, args.et_min_prob)
        for c in cases
    )
    rows = aggregate_rows(run_parallel(results, args.n_jobs), ["tc", "wt", "et"])
    rows.sort(key=lambda r: (r["tc"], r["wt"], r["et"]))
    write_csv(out_dir / "grid_phase2.csv", rows,
              ["tc", "wt", "et", "dice_tc", "dice_wt", "dice_et", "dice_mean"])
    chosen_thr_row, best_thr_mean_row = select_row(rows, args.selection_metric, args.mean_tolerance)

    if not args.no_refine:
        tc_vals = refine_values(chosen_thr_row["tc"], args.refine_threshold_radius, args.refine_threshold_step)
        wt_vals = refine_values(chosen_thr_row["wt"], args.refine_threshold_radius, args.refine_threshold_step)
        et_vals = refine_values(chosen_thr_row["et"], args.refine_threshold_radius, args.refine_threshold_step)
        refined_triples = list(product(tc_vals, wt_vals, et_vals))
        coarse_set = {(r["tc"], r["wt"], r["et"]) for r in rows}
        refined_triples = [tri for tri in refined_triples if tri not in coarse_set]
        if refined_triples:
            print(f"Refining thresholds around tc={chosen_thr_row['tc']:.3f}, wt={chosen_thr_row['wt']:.3f}, et={chosen_thr_row['et']:.3f}")
            results = Parallel(n_jobs=args.n_jobs, verbose=10)(
                delayed(process_case_thresholds)(c, prob_dirs, ref_dir, weights, refined_triples, args.et_threshold_voxels, args.et_min_prob)
                for c in cases
            )
            ref_rows = aggregate_rows(run_parallel(results, args.n_jobs), ["tc", "wt", "et"])
            rows = sorted(rows + ref_rows, key=lambda r: (r["tc"], r["wt"], r["et"]))
            write_csv(out_dir / "grid_phase2_refined.csv", rows,
                      ["tc", "wt", "et", "dice_tc", "dice_wt", "dice_et", "dice_mean"])
            chosen_thr_row, best_thr_mean_row = select_row(rows, args.selection_metric, args.mean_tolerance)

    best_thresh = {"tc": float(chosen_thr_row["tc"]), "wt": float(chosen_thr_row["wt"]), "et": float(chosen_thr_row["et"])}
    print(f"selected thresholds={best_thresh} mean={chosen_thr_row['dice_mean']:.4f} ET={chosen_thr_row['dice_et']:.4f}")
    print(f"phase2 best mean={best_thr_mean_row['dice_mean']:.4f} ET={best_thr_mean_row['dice_et']:.4f}")
    print(f"elapsed: {time.time() - t0:.0f}s")
    print()

    # Phase 3: min-size triples through final postprocess.
    if args.skip_phase3:
        print("Phase 3: SKIPPED")
        best_ms = {r: 0 for r in REGIONS}
        final_dice = {r: float(chosen_thr_row[f"dice_{r}"]) for r in REGIONS}
        final_dice["mean"] = float(chosen_thr_row["dice_mean"])
        chosen_ms_row = None
        best_ms_mean_row = None
    else:
        print("=" * 70)
        print("Phase 3: min-size triple search (postprocess-consistent)")
        print("=" * 70)
        t0 = time.time()
        size_triples = list(product(args.min_sizes, args.min_sizes, args.min_sizes))
        results = Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(process_case_min_sizes)(c, prob_dirs, ref_dir, weights, best_thresh, size_triples,
                                            args.et_threshold_voxels, args.et_min_prob)
            for c in cases
        )
        rows = aggregate_rows(run_parallel(results, args.n_jobs), ["tc", "wt", "et"])
        rows.sort(key=lambda r: (r["tc"], r["wt"], r["et"]))
        write_csv(out_dir / "grid_phase3.csv", rows,
                  ["tc", "wt", "et", "dice_tc", "dice_wt", "dice_et", "dice_mean"])
        chosen_ms_row, best_ms_mean_row = select_row(rows, args.selection_metric, args.mean_tolerance)
        best_ms = {"tc": int(chosen_ms_row["tc"]), "wt": int(chosen_ms_row["wt"]), "et": int(chosen_ms_row["et"])}
        final_dice = {r: float(chosen_ms_row[f"dice_{r}"]) for r in REGIONS}
        final_dice["mean"] = float(chosen_ms_row["dice_mean"])
        print(f"selected min_sizes={best_ms} mean={final_dice['mean']:.4f} ET={final_dice['et']:.4f}")
        print(f"phase3 best mean={best_ms_mean_row['dice_mean']:.4f} ET={best_ms_mean_row['dice_et']:.4f}")
        print(f"elapsed: {time.time() - t0:.0f}s")
        print()

    best_config = {
        "models": dict(zip(args.model_names, [str(d) for d in prob_dirs])),
        "weights": dict(zip(args.model_names, weights)),
        "alpha": best_alpha,
        "selection_metric": args.selection_metric,
        "mean_tolerance": args.mean_tolerance,
        "postprocess_consistent": True,
        "coarse_to_fine": not args.no_refine,
        "et_threshold_voxels": args.et_threshold_voxels,
        "et_min_prob": args.et_min_prob,
        "phase1_best_mean": best_alpha_mean_row,
        "phase2_best_mean": best_thr_mean_row,
        "phase3_best_mean": best_ms_mean_row,
        "thresholds": best_thresh,
        "min_sizes": best_ms,
        "final_dice": final_dice,
        "n_cases": len(cases),
        "skip_phase3": args.skip_phase3,
    }
    with (out_dir / "best_config.json").open("w") as f:
        json.dump(best_config, f, indent=2)

    summary = "\n".join([
        "=" * 70,
        "GRID SEARCH FINAL RESULTS",
        "=" * 70,
        f"N cases:    {len(cases)}",
        f"Weights:    {dict(zip(args.model_names, weights))}",
        f"Thresholds: {best_thresh}",
        f"Min sizes:  {best_ms}",
        f"Final Dice: TC={final_dice['tc']:.4f} WT={final_dice['wt']:.4f} ET={final_dice['et']:.4f} mean={final_dice['mean']:.4f}",
        "=" * 70,
    ])
    (out_dir / "summary.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
