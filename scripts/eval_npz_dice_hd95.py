import argparse
import csv
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


REGIONS = ["tc", "wt", "et"]


def gt_regions(seg):
    """
    BraTS labels:
      0 = background
      1 = NCR/NET
      2 = edema
      4 = enhancing tumor

    Regions:
      WT = labels 1,2,4
      TC = labels 1,4
      ET = label 4
    """
    seg = seg.astype(np.uint8)
    return {
        "tc": np.logical_or(seg == 1, seg == 4),
        "wt": seg > 0,
        "et": seg == 4,
    }


def dice_binary(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    pred_sum = pred.sum()
    gt_sum = gt.sum()

    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0

    inter = np.logical_and(pred, gt).sum()
    return float(2.0 * inter / (pred_sum + gt_sum))


def hd95_binary(pred, gt, spacing=(1.0, 1.0, 1.0), empty_penalty=373.12866):
    """
    Symmetric 95th percentile Hausdorff distance.

    Empty handling:
      both empty -> 0.0
      one empty  -> empty_penalty
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float(empty_penalty)

    structure = generate_binary_structure(3, 1)

    pred_border = np.logical_xor(
        pred,
        binary_erosion(pred, structure=structure, border_value=0),
    )
    gt_border = np.logical_xor(
        gt,
        binary_erosion(gt, structure=structure, border_value=0),
    )

    if not pred_border.any() and not gt_border.any():
        return 0.0
    if not pred_border.any() or not gt_border.any():
        return float(empty_penalty)

    # distance to nearest surface voxel
    dt_gt = distance_transform_edt(~gt_border, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)

    d_pred_to_gt = dt_gt[pred_border]
    d_gt_to_pred = dt_pred[gt_border]

    distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    if distances.size == 0:
        return float(empty_penalty)

    return float(np.percentile(distances, 95))


def load_pred_regions(npz_path, threshold=128, enforce_hierarchy=False):
    z = np.load(npz_path)
    prob = z["prob"]

    if prob.shape[0] != 3:
        raise ValueError(f"{npz_path}: expected prob shape (3,H,W,D), got {prob.shape}")

    # uint8 prob: 0~255
    pred_tc = prob[0] >= threshold
    pred_wt = prob[1] >= threshold
    pred_et = prob[2] >= threshold

    if enforce_hierarchy:
        # ET should be inside TC; TC should be inside WT.
        pred_tc = np.logical_or(pred_tc, pred_et)
        pred_wt = np.logical_or(pred_wt, pred_tc)

    return {
        "tc": pred_tc,
        "wt": pred_wt,
        "et": pred_et,
    }


def evaluate_one(npz_path, data_root, threshold=128, enforce_hierarchy=False, empty_penalty=373.12866):
    case_id = npz_path.stem
    gt_path = data_root / case_id / f"{case_id}_seg.nii.gz"

    if not gt_path.exists():
        raise FileNotFoundError(f"GT not found: {gt_path}")

    gt_img = nib.load(str(gt_path))
    seg = np.asarray(gt_img.dataobj)
    spacing = gt_img.header.get_zooms()[:3]

    pred = load_pred_regions(npz_path, threshold=threshold, enforce_hierarchy=enforce_hierarchy)
    gt = gt_regions(seg)

    # shape check
    for r in REGIONS:
        if pred[r].shape != gt[r].shape:
            raise ValueError(
                f"{case_id} {r}: pred shape {pred[r].shape} != gt shape {gt[r].shape}"
            )

    row = {"case": case_id}

    dice_vals = []
    hd_vals = []

    for r in REGIONS:
        d = dice_binary(pred[r], gt[r])
        h = hd95_binary(pred[r], gt[r], spacing=spacing, empty_penalty=empty_penalty)

        row[f"dice_{r}"] = d
        row[f"hd95_{r}"] = h

        dice_vals.append(d)
        hd_vals.append(h)

    row["dice_mean"] = float(np.mean(dice_vals))
    row["hd95_mean"] = float(np.mean(hd_vals))

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs_dir", required=True, help="Directory containing *.npz probability files")
    ap.add_argument("--data_root", default="data/BraTS2021_TrainingData")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--threshold", type=int, default=128, help="uint8 threshold, default 128")
    ap.add_argument("--enforce_hierarchy", action="store_true", help="force ET⊂TC⊂WT")
    ap.add_argument("--empty_penalty", type=float, default=373.12866)
    args = ap.parse_args()

    probs_dir = Path(args.probs_dir)
    data_root = Path(args.data_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(probs_dir.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No npz files found in {probs_dir}")

    print(f"Found {len(files)} npz files in {probs_dir}")
    print(f"Data root: {data_root}")
    print(f"Threshold: {args.threshold}")
    print(f"Enforce hierarchy: {args.enforce_hierarchy}")
    print()

    rows = []
    for i, f in enumerate(files, 1):
        try:
            row = evaluate_one(
                f,
                data_root=data_root,
                threshold=args.threshold,
                enforce_hierarchy=args.enforce_hierarchy,
                empty_penalty=args.empty_penalty,
            )
            rows.append(row)
        except Exception as e:
            print(f"[ERROR] {f.name}: {type(e).__name__}: {e}")
            continue

        if i % 20 == 0 or i == len(files):
            print(
                f"[{i}/{len(files)}] "
                f"last={row['case']} "
                f"dice={row['dice_mean']:.4f} "
                f"hd95={row['hd95_mean']:.2f}"
            )

    if not rows:
        raise RuntimeError("No valid cases evaluated")

    fieldnames = [
        "case",
        "dice_tc", "dice_wt", "dice_et", "dice_mean",
        "hd95_tc", "hd95_wt", "hd95_et", "hd95_mean",
    ]

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Saved per-case CSV: {out_csv}")
    print()
    print("===== Summary =====")

    for key in [
        "dice_tc", "dice_wt", "dice_et", "dice_mean",
        "hd95_tc", "hd95_wt", "hd95_et", "hd95_mean",
    ]:
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        print(f"{key:10s}: mean={vals.mean():.4f}  std={vals.std():.4f}")

    print()
    print("n_cases:", len(rows))


if __name__ == "__main__":
    main()
