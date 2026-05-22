#!/usr/bin/env python3
"""
PARALLEL Dice + HD95 evaluation.
Speedup: ~8-10x vs single-core eval_npz_dice_hd95.py
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import nibabel as nib
from joblib import Parallel, delayed
from scipy.ndimage import (
    binary_erosion, distance_transform_edt,
    generate_binary_structure,
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


def hd95_binary(pred, gt, spacing=(1.0,1.0,1.0), empty_penalty=373.12866):
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


def load_prob_uint8(path):
    z = np.load(path); p = z["prob"]
    if p.dtype != np.uint8:
        if p.dtype in (np.float32, np.float64, np.float16):
            p = (np.clip(p * 255.0 if p.max() <= 1.0 else p, 0, 255)).astype(np.uint8)
        else:
            p = p.astype(np.uint8)
    return p


def process_case(npz_path, data_root, threshold):
    cid = Path(npz_path).stem
    try:
        prob = load_prob_uint8(npz_path)
    except Exception:
        return None
    gt_path = Path(data_root) / cid / f"{cid}_seg.nii.gz"
    if not gt_path.exists():
        return None
    img = nib.load(str(gt_path))
    seg = np.asarray(img.dataobj)
    spacing = img.header.get_zooms()[:3]
    gt = gt_regions(seg)
    
    thr_u8 = int(round(threshold * 255))
    row = {"case": cid}
    d_list, h_list = [], []
    for i, r in enumerate(REGIONS):
        pred = prob[i] >= thr_u8
        d = dice_binary(pred, gt[r])
        h = hd95_binary(pred, gt[r], spacing=spacing)
        row[f"dice_{r}"] = d
        row[f"hd95_{r}"] = h
        d_list.append(d); h_list.append(h)
    row["dice_mean"] = float(np.mean(d_list))
    row["hd95_mean"] = float(np.mean(h_list))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs_dir", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max_cases", type=int, default=0)
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()

    files = sorted(Path(args.probs_dir).glob("*.npz"))
    files = [f for f in files if not f.name.startswith("_tmp_")]
    if args.max_cases > 0:
        files = files[:args.max_cases]

    print(f"Probs: {args.probs_dir}")
    print(f"Cases: {len(files)}, workers: {args.n_jobs}")

    t0 = time.time()
    rows = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(process_case)(f, args.data_root, args.threshold) for f in files
    )
    rows = [r for r in rows if r is not None]
    print(f"  processed {len(rows)}/{len(files)} cases in {time.time()-t0:.0f}s")

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case","dice_tc","dice_wt","dice_et","dice_mean",
                  "hd95_tc","hd95_wt","hd95_et","hd95_mean"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)

    print("\n" + "="*60)
    for k in fieldnames[1:]:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        print(f"  {k:12s}: mean={vals.mean():.4f}  std={vals.std():.4f}")
    print(f"  saved: {args.out_csv}")


if __name__ == "__main__":
    main()
