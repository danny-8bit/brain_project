#!/usr/bin/env python3
"""
Final evaluation: Use best_config from grid_search to evaluate on ALL cases.
Outputs Dice + HD95 per case + summary.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import (
    binary_erosion, distance_transform_edt,
    label, generate_binary_structure,
)


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


def hd95_binary(pred, gt, spacing=(1.0, 1.0, 1.0), empty_penalty=373.12866):
    pred = pred.astype(bool); gt = gt.astype(bool)
    if not pred.any() and not gt.any(): return 0.0
    if not pred.any() or not gt.any():  return float(empty_penalty)
    s = generate_binary_structure(3, 1)
    pb = np.logical_xor(pred, binary_erosion(pred, structure=s, border_value=0))
    gb = np.logical_xor(gt, binary_erosion(gt, structure=s, border_value=0))
    if not pb.any() or not gb.any(): return float(empty_penalty)
    dt_g = distance_transform_edt(~gb, sampling=spacing)
    dt_p = distance_transform_edt(~pb, sampling=spacing)
    d = np.concatenate([dt_g[pb], dt_p[gb]])
    return float(np.percentile(d, 95))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob_dirs", nargs="+", required=True)
    ap.add_argument("--ref_dir", default="data/BraTS2021_TrainingData")
    ap.add_argument("--config", required=True, help="best_config.json from grid_search")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--model_names", nargs="+", default=None)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    if args.model_names is None:
        args.model_names = [Path(d).name.replace("_oof", "") for d in args.prob_dirs]
    weights = [cfg["weights"][m] for m in args.model_names]
    thresholds = cfg["thresholds"]
    min_sizes = cfg["min_sizes"]

    print("=" * 70)
    print("Final Evaluation Configuration")
    print("=" * 70)
    print(f"  weights:    {dict(zip(args.model_names, weights))}")
    print(f"  thresholds: {thresholds}")
    print(f"  min_sizes:  {min_sizes}")
    print()

    ref = Path(args.ref_dir)
    prob_dirs = [Path(d) for d in args.prob_dirs]
    sets = [set(f.stem for f in d.glob("*.npz")) for d in prob_dirs]
    cases = sorted(sets[0].intersection(*sets[1:]) if len(sets) > 1 else sets[0])
    print(f"Total cases: {len(cases)}")
    print()

    rows = []
    t0 = time.time()
    for ci, cid in enumerate(cases, 1):
        try:
            probs = [load_prob_uint8(d / f"{cid}.npz") for d in prob_dirs]
        except Exception as e:
            print(f"  [skip] {cid}: load fail ({e})")
            continue
        gt_path = ref / cid / f"{cid}_seg.nii.gz"
        if not gt_path.exists():
            continue
        img = nib.load(str(gt_path))
        seg = np.asarray(img.dataobj)
        spacing = img.header.get_zooms()[:3]
        gt = gt_regions(seg)

        ens = ensemble_uint8(probs, weights)
        row = {"case": cid}
        d_list, h_list = [], []
        for i, r in enumerate(REGIONS):
            pred = ens[i] >= int(round(thresholds[r] * 255))
            pred = remove_small_components(pred, min_sizes[r])
            d = dice_binary(pred, gt[r])
            h = hd95_binary(pred, gt[r], spacing=spacing)
            row[f"dice_{r}"] = d
            row[f"hd95_{r}"] = h
            d_list.append(d); h_list.append(h)
        row["dice_mean"] = float(np.mean(d_list))
        row["hd95_mean"] = float(np.mean(h_list))
        rows.append(row)

        if ci % 50 == 0 or ci == len(cases):
            print(f"  [{ci}/{len(cases)}] {cid}  ({time.time()-t0:.0f}s)")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case", "dice_tc", "dice_wt", "dice_et", "dice_mean",
                  "hd95_tc", "hd95_wt", "hd95_et", "hd95_mean"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)

    print()
    print("=" * 70)
    print("FINAL ENSEMBLE RESULTS (full dataset)")
    print("=" * 70)
    for k in fieldnames[1:]:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        print(f"  {k:12s}: mean={vals.mean():.4f}  std={vals.std():.4f}")
    print(f"  n_cases: {len(rows)}")
    print(f"  saved:   {out_csv}")


if __name__ == "__main__":
    main()
