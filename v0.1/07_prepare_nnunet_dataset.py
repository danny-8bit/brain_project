import os
import json
import shutil
import nibabel as nib
import numpy as np


DATA_DIR = "./data/BraTS2021"
SPLIT_JSON = "./splits_holdout_seed42.json"

NNUNET_RAW = os.environ.get("nnUNet_raw", "./nnUNet_raw")

DATASET_ID = 501
DATASET_NAME = "Dataset501_BraTSGlioma"

OUT_DATASET_DIR = os.path.join(NNUNET_RAW, DATASET_NAME)

IMAGES_TR = os.path.join(OUT_DATASET_DIR, "imagesTr")
LABELS_TR = os.path.join(OUT_DATASET_DIR, "labelsTr")

VAL_IMAGE_DIR = "./result/nnunet_val_images"

MODALITIES = [
    ("flair", "0000"),
    ("t1ce", "0001"),
    ("t1", "0002"),
    ("t2", "0003"),
]


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def copy_or_link(src, dst):
    if os.path.exists(dst):
        return

    src_abs = os.path.abspath(src)

    try:
        os.symlink(src_abs, dst)
    except Exception:
        shutil.copy2(src_abs, dst)


def remap_label_4_to_3(src_label, dst_label):
    if os.path.exists(dst_label):
        return

    img = nib.load(src_label)
    data = img.get_fdata().astype(np.uint8)

    data[data == 4] = 3

    out = nib.Nifti1Image(data.astype(np.uint8), img.affine, img.header)
    nib.save(out, dst_label)


def load_split():
    if not os.path.exists(SPLIT_JSON):
        raise FileNotFoundError(
            f"找不到 {SPLIT_JSON}，请先运行 00_make_splits.py"
        )

    with open(SPLIT_JSON, "r", encoding="utf-8") as f:
        split = json.load(f)

    train_ids = split["train"]
    val_ids = split["val"]

    print(f"训练病例数: {len(train_ids)}")
    print(f"验证病例数: {len(val_ids)}")

    return train_ids, val_ids


def prepare_training_cases(train_ids):
    print("\n准备 nnU-Net imagesTr / labelsTr ...")

    for case_id in train_ids:
        case_dir = os.path.join(DATA_DIR, case_id)

        for modality, suffix in MODALITIES:
            src = os.path.join(case_dir, f"{case_id}_{modality}.nii.gz")
            dst = os.path.join(IMAGES_TR, f"{case_id}_{suffix}.nii.gz")

            if not os.path.exists(src):
                raise FileNotFoundError(src)

            copy_or_link(src, dst)

        src_label = os.path.join(case_dir, f"{case_id}_seg.nii.gz")
        dst_label = os.path.join(LABELS_TR, f"{case_id}.nii.gz")

        if not os.path.exists(src_label):
            raise FileNotFoundError(src_label)

        remap_label_4_to_3(src_label, dst_label)


def prepare_val_images(val_ids):
    print("\n准备 nnU-Net validation prediction images ...")

    safe_mkdir(VAL_IMAGE_DIR)

    for case_id in val_ids:
        case_dir = os.path.join(DATA_DIR, case_id)

        for modality, suffix in MODALITIES:
            src = os.path.join(case_dir, f"{case_id}_{modality}.nii.gz")
            dst = os.path.join(VAL_IMAGE_DIR, f"{case_id}_{suffix}.nii.gz")

            if not os.path.exists(src):
                raise FileNotFoundError(src)

            copy_or_link(src, dst)


def write_dataset_json(num_training):
    dataset_json = {
        "channel_names": {
            "0": "FLAIR",
            "1": "T1ce",
            "2": "T1",
            "3": "T2"
        },
        "labels": {
            "background": 0,
            "NCR_NET": 1,
            "ED": 2,
            "ET": 3
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz"
    }

    out_path = os.path.join(OUT_DATASET_DIR, "dataset.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=4, ensure_ascii=False)

    print(f"\n已写入 dataset.json: {out_path}")


def main():
    safe_mkdir(OUT_DATASET_DIR)
    safe_mkdir(IMAGES_TR)
    safe_mkdir(LABELS_TR)
    safe_mkdir("./result")

    train_ids, val_ids = load_split()

    prepare_training_cases(train_ids)
    prepare_val_images(val_ids)
    write_dataset_json(num_training=len(train_ids))

    print("\n✅ nnU-Net v2 数据准备完成")
    print(f"nnUNet_raw dataset: {OUT_DATASET_DIR}")
    print(f"validation image dir: {VAL_IMAGE_DIR}")


if __name__ == "__main__":
    main()