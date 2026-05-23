#!/usr/bin/env python3
"""
Parallel final eval using postprocess_brats — consistent with grid_search.py.

Reads best_config.json from grid_search, applies the SAME postprocessing pipeline
(weighted fusion → postprocess_brats → BraTS TC/WT/ET Dice + HD95).

Usage:
  # 全量 1251 (论文用)
  python scripts/final_eval.py \
      --prob_dirs /dev/shm/brats_probs/segresnet_oof /dev/shm/brats_probs/swinunetr_oof \
      --config work_dir/grid_search/best_config.json \
      --out_csv result/final_ensemble_1251.csv

  # 子采样 (快速验证)
  python scripts/final_eval.py ... --max_cases 250
"""

import argparse, csv, json, time, sys
from pathlib import Path
import numpy as np
import nibabel as nib
from joblib import Parallel, delayed
from scipy.ndimage import (
    binary_erosion, distance_transform_edt, generate_binary_structure,
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.postprocess import postprocess_brats


REGIONS = ["tc", "wt", "et"]


def gt_regions(seg):
    seg = seg.astype(np.uint8)
    return {"tc":(seg==1)|(seg==4), "wt":seg>0, "et":seg==4}


def pred_regions(label):
    label = label.astype(np.uint8)
    return {"tc":(label==1)|(label==4), "wt":label>0, "et":label==4}


def dice_binary(pred, gt):
    pred = pred.astype(bool); gt = gt.astype(bool)
    ps = pred.sum(); gs = gt.sum()
    if ps == 0 and gs == 0: return 1.0
    if ps == 0 or gs == 0:  return 0.0
    return float(2.0 * np.logical_and(pred, gt).sum() / (ps + gs))


def hd95_binary(pred, gt, spacing=(1,1,1), empty_penalty=373.12866):
    pred = pred.astype(bool); gt = gt.astype(bool)
    if not pred.any() and not gt.any(): return 0.0
    if not pred.any() or not gt.any():  return float(empty_penalty)
    s = generate_binary_structure(3, 1)
    pb = np.logical_xor(pred, binary_erosion(pred, structure=s, border_value=0))
    gb = np.logical_xor(gt, binary_erosion(gt, structure=s, border_value=0))
    if not pb.any() or not gb.any(): return float(empty_penalty)
    dt_g = distance_transform_edt(~gb, sampling=spacing)
    dt_p = distance_transform_edt(~pb, sampling=spacing)
    return float(np.percentile(np.concatenate([dt_g[pb], dt_p[gb]]), 95))


def load_prob(p):
    z = np.load(p); pr = z["prob"]
    if pr.dtype != np.uint8:
        pr = (pr*255 if pr.max()<=1 else pr).clip(0,255).astype(np.uint8)
    return pr


def ensemble(probs, weights):
    out = np.zeros_like(probs[0], dtype=np.float32)
    for p, w in zip(probs, weights):
        out += p.astype(np.float32) * float(w)
    return np.clip(out, 0, 255).astype(np.uint8)


def process(cid, prob_dirs, ref_dir, weights, thresholds, min_sizes,
            et_threshold_voxels, et_min_prob, compute_hd95):
    try:
        probs = [load_prob(d / f"{cid}.npz") for d in prob_dirs]
    except Exception:
        return None
    gp = ref_dir / cid / f"{cid}_seg.nii.gz"
    if not gp.exists(): return None
    img = nib.load(str(gp))
    seg = np.asarray(img.dataobj)
    spacing = img.header.get_zooms()[:3]
    gt = gt_regions(seg)

    # Ensemble + postprocess (与 grid_search.py 完全一致!)
    fused_u8 = ensemble(probs, weights)
    prob_f32 = fused_u8.astype(np.float32) / 255.0
    label = postprocess_brats(
        prob_f32,
        thresholds=thresholds,
        min_sizes=min_sizes,
        et_threshold_voxels=et_threshold_voxels,
        et_min_prob=et_min_prob,
    )
    pr = pred_regions(label)

    row = {"case": cid}
    d_list, h_list = [], []
    for r in REGIONS:
        d = dice_binary(pr[r], gt[r])
        row[f"dice_{r}"] = d
        d_list.append(d)
        if compute_hd95:
            h = hd95_binary(pr[r], gt[r], spacing=spacing)
            row[f"hd95_{r}"] = h
            h_list.append(h)
    row["dice_mean"] = float(np.mean(d_list))
    if compute_hd95:
        row["hd95_mean"] = float(np.mean(h_list))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prob_dirs", nargs="+", required=True)
    ap.add_argument("--ref_dir", default="data/BraTS2021_TrainingData")
    ap.add_argument("--config", required=True, help="grid_search 输出的 best_config.json")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--model_names", nargs="+", default=None)
    ap.add_argument("--max_cases", type=int, default=0, help="0 = 全量, N = 子采样前 N")
    ap.add_argument("--n_jobs", type=int, default=-1, help="-1 = all cores")
    ap.add_argument("--no_hd95", action="store_true", help="跳过 HD95 (更快)")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    if args.model_names is None:
        args.model_names = [Path(d).name.replace("_oof","") for d in args.prob_dirs]
    weights = [cfg["weights"][m] for m in args.model_names]
    thresholds = cfg["thresholds"]
    min_sizes = cfg.get("min_sizes", {"tc":0, "wt":0, "et":0})
    et_voxels = cfg.get("et_threshold_voxels", 200)
    et_min_p = cfg.get("et_min_prob", 0.5)

    print("="*70)
    print("FINAL EVAL (postprocess_brats, consistent with grid_search.py)")
    print("="*70)
    print(f"weights:               {dict(zip(args.model_names, weights))}")
    print(f"thresholds:            {thresholds}")
    print(f"min_sizes:             {min_sizes}")
    print(f"et_threshold_voxels:   {et_voxels}")
    print(f"et_min_prob:           {et_min_p}")
    print(f"compute_hd95:          {not args.no_hd95}")
    print()

    ref_dir = Path(args.ref_dir)
    prob_dirs = [Path(d) for d in args.prob_dirs]
    sets = [set(f.stem for f in d.glob("*.npz") if not f.name.startswith("_tmp_")) for d in prob_dirs]
    cases = sorted(sets[0].intersection(*sets[1:]) if len(sets) > 1 else sets[0])
    if args.max_cases > 0:
        cases = cases[:args.max_cases]
    print(f"Total cases: {len(cases)}")

    t0 = time.time()
    rows = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(process)(c, prob_dirs, ref_dir, weights, thresholds, min_sizes,
                         et_voxels, et_min_p, not args.no_hd95) for c in cases
    )
    rows = [r for r in rows if r is not None]
    print(f"\nprocessed {len(rows)}/{len(cases)} cases in {time.time()-t0:.0f}s")

    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case","dice_tc","dice_wt","dice_et","dice_mean"]
    if not args.no_hd95:
        fields += ["hd95_tc","hd95_wt","hd95_et","hd95_mean"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    print()
    print("="*70)
    print("FINAL ENSEMBLE RESULTS")
    print("="*70)
    for k in fields[1:]:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        print(f"  {k:12s}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"median={np.median(vals):.4f}")
    print(f"  n_cases: {len(rows)}")
    print(f"  saved:   {out}")


if __name__ == "__main__":
    main()
