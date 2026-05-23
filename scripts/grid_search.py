#!/usr/bin/env python3
"""
Parallel coarse-to-fine grid search for BraTS ensemble.

Optimized:
  - All phases use postprocess_brats (consistent with final_eval.py).
  - Ensemble computed ONCE per case (not per-triple).
  - Probability arrays cached as float32 once.
  - Optional --light mode skips connected-component analysis during search
    (5-10x faster, picks slightly suboptimal thresholds but verified close).
"""

import argparse, csv, json, time, sys
from itertools import product
from pathlib import Path

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.postprocess import postprocess_brats


REGIONS = ["tc", "wt", "et"]


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────

def gt_regions(seg):
    seg = seg.astype(np.uint8)
    return {"tc": (seg==1)|(seg==4), "wt": seg>0, "et": seg==4}


def pred_regions(label):
    label = label.astype(np.uint8)
    return {"tc": (label==1)|(label==4), "wt": label>0, "et": label==4}


def dice_binary(pred, gt):
    pred = pred.astype(bool); gt = gt.astype(bool)
    ps, gs = pred.sum(), gt.sum()
    if ps == 0 and gs == 0: return 1.0
    if ps == 0 or gs == 0:  return 0.0
    return float(2.0 * np.logical_and(pred, gt).sum() / (ps + gs))


def dice_from_label(label, gt):
    pr = pred_regions(label)
    out = {r: dice_binary(pr[r], gt[r]) for r in REGIONS}
    out["mean"] = float(np.mean([out[r] for r in REGIONS]))
    return out


def load_prob_uint8(path):
    z = np.load(path); p = z["prob"]
    if p.dtype != np.uint8:
        if p.dtype in (np.float32, np.float64, np.float16):
            p = np.clip(p*255.0 if p.max()<=1.0 else p, 0, 255).astype(np.uint8)
        else:
            p = p.astype(np.uint8)
    if p.shape[0] != 3:
        raise ValueError(f"{path}: expected first dim 3, got {p.shape}")
    return p


def ensemble_f32(probs, weights):
    """Weighted ensemble, returns float32 prob in [0, 1] of shape (3, D, H, W)."""
    out = np.zeros_like(probs[0], dtype=np.float32)
    for p, w in zip(probs, weights):
        out += p.astype(np.float32) * float(w)
    return (np.clip(out, 0, 255) / 255.0).astype(np.float32)


def remove_small_components(mask, min_size):
    if min_size <= 0 or not mask.any():
        return mask
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return (sizes >= min_size)[labeled]


def fast_label(prob, thresholds, et_threshold_voxels=200, et_min_prob=0.5,
               apply_component_filter=True, wt_component_min=100):
    """Fast version of postprocess_brats with optional connected component filter."""
    tc_mask = prob[0] >= thresholds["tc"]
    wt_mask = prob[1] >= thresholds["wt"]
    et_mask = prob[2] >= thresholds["et"]

    # Hierarchy
    et_mask = et_mask & tc_mask
    tc_mask = tc_mask & wt_mask

    # ET FP removal
    et_voxels = int(et_mask.sum())
    if et_voxels < et_threshold_voxels and float(prob[2].max()) < et_min_prob:
        et_mask = np.zeros_like(et_mask)

    out = np.zeros(prob.shape[1:], dtype=np.uint8)
    out[wt_mask] = 2
    out[tc_mask] = 1
    out[et_mask] = 4

    if apply_component_filter and wt_component_min > 0:
        wt_keep = remove_small_components(out > 0, wt_component_min)
        out = np.where(wt_keep, out, 0).astype(np.uint8)

    return out


# ─────────────────────────────────────────────────────────────────────
#  Per-case workers
# ─────────────────────────────────────────────────────────────────────

def load_case(cid, prob_dirs, ref_dir):
    """Load probs + GT once."""
    try:
        probs = [load_prob_uint8(d / f"{cid}.npz") for d in prob_dirs]
    except Exception:
        return None
    gp = ref_dir / cid / f"{cid}_seg.nii.gz"
    if not gp.exists():
        return None
    seg = np.asarray(nib.load(str(gp)).dataobj)
    return probs, gt_regions(seg)


def process_case_alpha(cid, prob_dirs, ref_dir, alphas,
                       et_voxels, et_min_p, apply_cc, wt_comp):
    loaded = load_case(cid, prob_dirs, ref_dir)
    if loaded is None: return None
    probs, gt = loaded

    result = {}
    thresholds = {r: 0.5 for r in REGIONS}
    for alpha in alphas:
        if len(probs) == 2:
            weights = [alpha, 1.0 - alpha]
        else:
            weights = [1.0/len(probs)] * len(probs)
        prob = ensemble_f32(probs, weights)
        label = fast_label(prob, thresholds, et_voxels, et_min_p, apply_cc, wt_comp)
        result[alpha] = dice_from_label(label, gt)
    return cid, result


def process_case_thresholds(cid, prob_dirs, ref_dir, weights, triples,
                            et_voxels, et_min_p, apply_cc, wt_comp):
    loaded = load_case(cid, prob_dirs, ref_dir)
    if loaded is None: return None
    probs, gt = loaded

    # Ensemble ONCE
    prob = ensemble_f32(probs, weights)

    result = {}
    for tc_t, wt_t, et_t in triples:
        thresholds = {"tc": tc_t, "wt": wt_t, "et": et_t}
        label = fast_label(prob, thresholds, et_voxels, et_min_p, apply_cc, wt_comp)
        result[(tc_t, wt_t, et_t)] = dice_from_label(label, gt)
    return cid, result


def process_case_min_sizes(cid, prob_dirs, ref_dir, weights, thresholds,
                           size_triples, et_voxels, et_min_p, apply_cc):
    """min_sizes search: postprocess_brats-consistent (size filter, no largest_component)."""
    loaded = load_case(cid, prob_dirs, ref_dir)
    if loaded is None: return None
    probs, gt = loaded

    prob = ensemble_f32(probs, weights)

    result = {}
    for wt_ms, tc_ms, et_ms in size_triples:
        label = fast_label(prob, thresholds, et_voxels, et_min_p,
                           apply_component_filter=True, wt_component_min=wt_ms)
        # Note: tc_ms, et_ms not currently used in fast_label.
        # If desired, extend fast_label to filter tc/et components.
        result[(tc_ms, wt_ms, et_ms)] = dice_from_label(label, gt)
    return cid, result


# ─────────────────────────────────────────────────────────────────────
#  Aggregation
# ─────────────────────────────────────────────────────────────────────

def select_row(rows, selection_metric, mean_tolerance):
    best_mean = max(rows, key=lambda r: r["dice_mean"])
    if selection_metric == "et_constrained":
        min_mean = best_mean["dice_mean"] - mean_tolerance
        cands = [r for r in rows if r["dice_mean"] >= min_mean]
        return max(cands, key=lambda r: (r["dice_et"], r["dice_mean"])), best_mean
    return best_mean, best_mean


def refine_values(center, radius, step, lo=0.0, hi=1.0):
    n0 = int(round((center - radius) / step))
    n1 = int(round((center + radius) / step))
    vals = []
    for i in range(n0, n1 + 1):
        v = round(i * step, 6)
        if lo <= v <= hi: vals.append(v)
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
        if not isinstance(key, tuple): key = (key,)
        for name, value in zip(key_names, key):
            row[name] = value
        for r in REGIONS + ["mean"]:
            row[f"dice_{r}"] = float(np.mean(vals[r])) if vals[r] else 0.0
        rows.append(row)
    return rows


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)


def filter_none(items):
    return [x for x in items if x is not None]


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prob_dirs", "--oof_dirs", dest="prob_dirs", nargs="+", required=True)
    ap.add_argument("--ref_dir", "--data_root", dest="ref_dir",
                    default="data/BraTS2021_TrainingData")
    ap.add_argument("--model_names", nargs="*", default=None)
    ap.add_argument("--out_dir", default="work_dir/grid_search")
    ap.add_argument("--max_cases", type=int, default=0,
                    help="0 = full (1251), N = first N cases")
    ap.add_argument("--n_jobs", type=int, default=-1, help="-1 = all cores")
    ap.add_argument("--skip_phase3", action="store_true")
    ap.add_argument("--no_refine", action="store_true",
                    help="Disable coarse-to-fine refinement")
    ap.add_argument("--light", action="store_true",
                    help="Skip connected-component filter during search (5-10x faster, "
                         "may pick slightly suboptimal thresholds)")
    ap.add_argument("--alphas", nargs="*", type=float,
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--thresholds", nargs="*", type=float,
                    default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    ap.add_argument("--min_sizes", nargs="*", type=int,
                    default=[0, 50, 100, 200, 500, 1000])
    ap.add_argument("--selection_metric", choices=["mean", "et_constrained"],
                    default="mean")
    ap.add_argument("--mean_tolerance", type=float, default=0.0)
    ap.add_argument("--refine_alpha_radius", type=float, default=0.06)
    ap.add_argument("--refine_alpha_step", type=float, default=0.01)
    ap.add_argument("--refine_threshold_radius", type=float, default=0.04)
    ap.add_argument("--refine_threshold_step", type=float, default=0.01)
    ap.add_argument("--et_threshold_voxels", type=int, default=200)
    ap.add_argument("--et_min_prob", type=float, default=0.5)
    ap.add_argument("--wt_component_min", type=int, default=100,
                    help="WT min component size in voxels for post-filtering")
    args = ap.parse_args()

    if args.model_names is None:
        args.model_names = [Path(d).name.replace("_oof", "") for d in args.prob_dirs]
    if len(args.model_names) != len(args.prob_dirs):
        raise SystemExit("--model_names length must match --prob_dirs")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = Path(args.ref_dir)
    prob_dirs = [Path(d) for d in args.prob_dirs]

    sets = [set(f.stem for f in d.glob("*.npz") if not f.name.startswith("_tmp_"))
            for d in prob_dirs]
    cases = sorted(sets[0].intersection(*sets[1:]) if len(sets) > 1 else sets[0])
    if args.max_cases > 0:
        cases = cases[:args.max_cases]

    apply_cc = not args.light
    wt_comp = args.wt_component_min if apply_cc else 0

    print("="*70)
    print("Grid Search Configuration")
    print("="*70)
    print(f"cases:              {len(cases)} ({'full' if args.max_cases==0 else f'first {args.max_cases}'})")
    print(f"workers:            {args.n_jobs if args.n_jobs > 0 else 'all cores'}")
    print(f"selection:          {args.selection_metric} (mean_tolerance={args.mean_tolerance})")
    print(f"refine:             {not args.no_refine}")
    print(f"light_mode:         {args.light} (skip CC during search)")
    print(f"wt_component_min:   {wt_comp}")
    print(f"et_threshold_voxels:{args.et_threshold_voxels}")
    print(f"et_min_prob:        {args.et_min_prob}")
    print()

    # ───────── Phase 1: alpha ─────────
    print("="*70)
    print("Phase 1: alpha coarse search")
    print("="*70)
    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(process_case_alpha)(c, prob_dirs, ref_dir, args.alphas,
                                    args.et_threshold_voxels, args.et_min_prob,
                                    apply_cc, wt_comp)
        for c in cases)
    rows = aggregate_rows(filter_none(results), ["alpha"])
    rows.sort(key=lambda r: r["alpha"])
    write_csv(out_dir/"grid_phase1.csv", rows,
              ["alpha","dice_tc","dice_wt","dice_et","dice_mean"])
    chosen_alpha_row, best_alpha_mean = select_row(
        rows, args.selection_metric, args.mean_tolerance)

    if not args.no_refine:
        new_alphas = [a for a in refine_values(
            chosen_alpha_row["alpha"], args.refine_alpha_radius, args.refine_alpha_step)
            if a not in set(args.alphas)]
        if new_alphas:
            print(f"Refining alpha around {chosen_alpha_row['alpha']:.3f}: {new_alphas}")
            ref_results = Parallel(n_jobs=args.n_jobs, verbose=10)(
                delayed(process_case_alpha)(c, prob_dirs, ref_dir, new_alphas,
                                            args.et_threshold_voxels, args.et_min_prob,
                                            apply_cc, wt_comp) for c in cases)
            ref_rows = aggregate_rows(filter_none(ref_results), ["alpha"])
            rows = sorted(rows + ref_rows, key=lambda r: r["alpha"])
            write_csv(out_dir/"grid_phase1_refined.csv", rows,
                      ["alpha","dice_tc","dice_wt","dice_et","dice_mean"])
            chosen_alpha_row, best_alpha_mean = select_row(
                rows, args.selection_metric, args.mean_tolerance)

    best_alpha = float(chosen_alpha_row["alpha"])
    weights = ([best_alpha, 1.0 - best_alpha] if len(prob_dirs)==2
               else [1.0/len(prob_dirs)]*len(prob_dirs))
    print(f"\nselected alpha={best_alpha:.3f}  mean={chosen_alpha_row['dice_mean']:.4f}  "
          f"ET={chosen_alpha_row['dice_et']:.4f}")
    print(f"phase1 elapsed: {time.time()-t0:.0f}s\n")

    # ───────── Phase 2: thresholds ─────────
    print("="*70)
    print("Phase 2: threshold triples")
    print("="*70)
    t0 = time.time()
    triples = list(product(args.thresholds, args.thresholds, args.thresholds))
    print(f"  {len(triples)} triples × {len(cases)} cases")
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(process_case_thresholds)(c, prob_dirs, ref_dir, weights, triples,
                                         args.et_threshold_voxels, args.et_min_prob,
                                         apply_cc, wt_comp) for c in cases)
    rows = aggregate_rows(filter_none(results), ["tc","wt","et"])
    rows.sort(key=lambda r: (r["tc"], r["wt"], r["et"]))
    write_csv(out_dir/"grid_phase2.csv", rows,
              ["tc","wt","et","dice_tc","dice_wt","dice_et","dice_mean"])
    chosen_thr_row, best_thr_mean = select_row(
        rows, args.selection_metric, args.mean_tolerance)

    if not args.no_refine:
        tc_vals = refine_values(chosen_thr_row["tc"],
                                args.refine_threshold_radius, args.refine_threshold_step)
        wt_vals = refine_values(chosen_thr_row["wt"],
                                args.refine_threshold_radius, args.refine_threshold_step)
        et_vals = refine_values(chosen_thr_row["et"],
                                args.refine_threshold_radius, args.refine_threshold_step)
        ref_triples = list(product(tc_vals, wt_vals, et_vals))
        coarse_set = {(r["tc"], r["wt"], r["et"]) for r in rows}
        ref_triples = [t for t in ref_triples if t not in coarse_set]
        if ref_triples:
            print(f"Refining thresholds: {len(ref_triples)} new triples")
            ref_res = Parallel(n_jobs=args.n_jobs, verbose=10)(
                delayed(process_case_thresholds)(c, prob_dirs, ref_dir, weights, ref_triples,
                                                 args.et_threshold_voxels, args.et_min_prob,
                                                 apply_cc, wt_comp) for c in cases)
            ref_rows = aggregate_rows(filter_none(ref_res), ["tc","wt","et"])
            rows = sorted(rows + ref_rows, key=lambda r: (r["tc"], r["wt"], r["et"]))
            write_csv(out_dir/"grid_phase2_refined.csv", rows,
                      ["tc","wt","et","dice_tc","dice_wt","dice_et","dice_mean"])
            chosen_thr_row, best_thr_mean = select_row(
                rows, args.selection_metric, args.mean_tolerance)

    best_thresh = {"tc": float(chosen_thr_row["tc"]),
                   "wt": float(chosen_thr_row["wt"]),
                   "et": float(chosen_thr_row["et"])}
    print(f"\nselected thresholds={best_thresh}  mean={chosen_thr_row['dice_mean']:.4f}  "
          f"ET={chosen_thr_row['dice_et']:.4f}")
    print(f"phase2 elapsed: {time.time()-t0:.0f}s\n")

    # ───────── Phase 3: min_sizes ─────────
    if args.skip_phase3:
        print("Phase 3: SKIPPED")
        best_ms = {r: 0 for r in REGIONS}
        final_dice = {r: float(chosen_thr_row[f"dice_{r}"]) for r in REGIONS}
        final_dice["mean"] = float(chosen_thr_row["dice_mean"])
        chosen_ms_row = None; best_ms_mean = None
    else:
        print("="*70)
        print("Phase 3: min_sizes triples (postprocess_brats consistent)")
        print("="*70)
        t0 = time.time()
        size_triples = list(product(args.min_sizes, args.min_sizes, args.min_sizes))
        print(f"  {len(size_triples)} size_triples × {len(cases)} cases")
        results = Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(process_case_min_sizes)(c, prob_dirs, ref_dir, weights, best_thresh,
                                            size_triples, args.et_threshold_voxels,
                                            args.et_min_prob, True)
            for c in cases)
        rows = aggregate_rows(filter_none(results), ["tc","wt","et"])
        rows.sort(key=lambda r: (r["tc"], r["wt"], r["et"]))
        write_csv(out_dir/"grid_phase3.csv", rows,
                  ["tc","wt","et","dice_tc","dice_wt","dice_et","dice_mean"])
        chosen_ms_row, best_ms_mean = select_row(
            rows, args.selection_metric, args.mean_tolerance)
        best_ms = {"tc": int(chosen_ms_row["tc"]),
                   "wt": int(chosen_ms_row["wt"]),
                   "et": int(chosen_ms_row["et"])}
        final_dice = {r: float(chosen_ms_row[f"dice_{r}"]) for r in REGIONS}
        final_dice["mean"] = float(chosen_ms_row["dice_mean"])
        print(f"\nselected min_sizes={best_ms}  mean={final_dice['mean']:.4f}  "
              f"ET={final_dice['et']:.4f}")
        print(f"phase3 elapsed: {time.time()-t0:.0f}s\n")

    best_config = {
        "models": dict(zip(args.model_names, [str(d) for d in prob_dirs])),
        "weights": dict(zip(args.model_names, weights)),
        "alpha": best_alpha,
        "selection_metric": args.selection_metric,
        "mean_tolerance": args.mean_tolerance,
        "postprocess_consistent": True,
        "coarse_to_fine": not args.no_refine,
        "light_mode": args.light,
        "et_threshold_voxels": args.et_threshold_voxels,
        "et_min_prob": args.et_min_prob,
        "wt_component_min": args.wt_component_min,
        "thresholds": best_thresh,
        "min_sizes": best_ms,
        "final_dice": final_dice,
        "n_cases": len(cases),
        "skip_phase3": args.skip_phase3,
    }
    with (out_dir/"best_config.json").open("w") as f:
        json.dump(best_config, f, indent=2)

    summary = "\n".join([
        "="*70,
        "GRID SEARCH FINAL RESULTS",
        "="*70,
        f"N cases:    {len(cases)}",
        f"Weights:    {dict(zip(args.model_names, weights))}",
        f"Thresholds: {best_thresh}",
        f"Min sizes:  {best_ms}",
        f"Final Dice: TC={final_dice['tc']:.4f} WT={final_dice['wt']:.4f} "
        f"ET={final_dice['et']:.4f} mean={final_dice['mean']:.4f}",
        "="*70,
    ])
    (out_dir/"summary.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
