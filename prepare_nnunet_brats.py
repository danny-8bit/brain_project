"""把 BraTS2021 数据转成 nnU-Net v2 格式"""
import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np


def convert_label_4_to_3(src, dst):
    img = nib.load(str(src))
    arr = np.asanyarray(img.dataobj).astype(np.uint8)
    arr[arr == 4] = 3
    out = nib.Nifti1Image(arr, img.affine, img.header)
    out.header.set_data_dtype(np.uint8)
    nib.save(out, str(dst))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst_raw", required=True)
    parser.add_argument("--dataset_id", type=int, default=1)
    parser.add_argument("--dataset_name", type=str, default="BraTSGlioma")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst_raw) / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    (dst / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dst / "labelsTr").mkdir(parents=True, exist_ok=True)

    n = 0
    for case_dir in sorted([p for p in src.iterdir() if p.is_dir()]):
        cid = case_dir.name
        t1 = case_dir / f"{cid}_t1.nii.gz"
        t1ce = case_dir / f"{cid}_t1ce.nii.gz"
        t2 = case_dir / f"{cid}_t2.nii.gz"
        flair = case_dir / f"{cid}_flair.nii.gz"
        seg = case_dir / f"{cid}_seg.nii.gz"

        if not all(p.exists() for p in [t1, t1ce, t2, flair, seg]):
            print(f"[skip] {cid}")
            continue

        shutil.copy(t1, dst / "imagesTr" / f"{cid}_0000.nii.gz")
        shutil.copy(t1ce, dst / "imagesTr" / f"{cid}_0001.nii.gz")
        shutil.copy(t2, dst / "imagesTr" / f"{cid}_0002.nii.gz")
        shutil.copy(flair, dst / "imagesTr" / f"{cid}_0003.nii.gz")
        convert_label_4_to_3(seg, dst / "labelsTr" / f"{cid}.nii.gz")
        n += 1
        if n % 50 == 0:
            print(f"  converted {n} cases...")

    meta = {
        "channel_names": {"0": "T1", "1": "T1ce", "2": "T2", "3": "FLAIR"},
        "labels": {"background": 0, "NCR_NET": 1, "ED": 2, "ET": 3},
        "numTraining": n,
        "file_ending": ".nii.gz",
    }
    with open(dst / "dataset.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. {n} cases -> {dst}")

if __name__ == "__main__":
    main()