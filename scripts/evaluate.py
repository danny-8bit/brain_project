"""在验证集上评估融合后的 nii.gz 结果"""

import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import load_datalist
from src.metrics import BratsMetrics


def label_to_regions(label):
    """BraTS label -> 3 region binary masks (TC, WT, ET)"""
    tc = np.isin(label, [1, 4])
    wt = np.isin(label, [1, 2, 4])
    et = (label == 4)
    return np.stack([tc, wt, et], axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--folds_json", default="./data/folds.json")
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()

    val_files = load_datalist(args.folds_json, args.fold, "val")
    metrics = BratsMetrics()

    for item in tqdm(val_files, ncols=100):
        case_id = item["id"]
        gt = np.asarray(nib.load(item["label"]).dataobj).astype(np.uint8)
        pred = np.asarray(nib.load(Path(args.pred_dir) / f"{case_id}.nii.gz").dataobj).astype(np.uint8)

        gt_r = torch.from_numpy(label_to_regions(gt))[None]
        pred_r = torch.from_numpy(label_to_regions(pred))[None]
        metrics(y_pred=pred_r, y=gt_r)

    r = metrics.aggregate()
    print("Validation results:")
    for k, v in r.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()