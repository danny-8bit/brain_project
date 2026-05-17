import os
import shutil
import json
import numpy as np
import nibabel as nib
from pathlib import Path

def convert_brats_to_nnunetv2(brats_dir, nnunet_raw_dir, dataset_id=137, dataset_name="BraTS21"):
    """
    将原生 BraTS 数据集转换为 nnU-Net v2 要求的严格目录和命名格式，并生成 dataset.json。
    """
    task_name = f"Dataset{dataset_id:03d}_{dataset_name}"
    target_base = os.path.join(nnunet_raw_dir, task_name)
    
    imagesTr = os.path.join(target_base, "imagesTr")
    labelsTr = os.path.join(target_base, "labelsTr")
    imagesTs = os.path.join(target_base, "imagesTs")
    
    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)
    os.makedirs(imagesTs, exist_ok=True)
    
    print(f"创建 nnU-Net 任务目录: {target_base}")
    
    # 遍历 BraTS 目录下的病例
    cases = sorted([d for d in os.listdir(brats_dir) if d.startswith("BraTS2021_") and os.path.isdir(os.path.join(brats_dir, d))])
    
    print(f"找到 {len(cases)} 个病例，正在转换格式...")
    
    for case in cases:
        case_path = os.path.join(brats_dir, case)
        # 模态映射后缀: 0000=T1, 0001=T1ce, 0002=T2, 0003=FLAIR
        modalities = {
            "t1": "0000",
            "t1ce": "0001",
            "t2": "0002",
            "flair": "0003"
        }
        
        # 转移影像
        for mod, suffix in modalities.items():
            src_img = os.path.join(case_path, f"{case}_{mod}.nii.gz")
            tgt_img = os.path.join(imagesTr, f"{case}_{suffix}.nii.gz")
            if os.path.exists(src_img):
                shutil.copy(src_img, tgt_img)
        
        # 转换并转移标签
        src_label = os.path.join(case_path, f"{case}_seg.nii.gz")
        tgt_label = os.path.join(labelsTr, f"{case}.nii.gz")
        if os.path.exists(src_label):
            # BraTS 原生标签为: 0(背景), 1(坏死/NCR), 2(水肿/ED), 4(增强/ET)
            # nnU-Net 要求标签连续，故需将 4 转为 3
            img = nib.load(src_label)
            data = img.get_fdata()
            # 标签转换
            data[data == 4] = 3
            new_img = nib.Nifti1Image(data.astype(np.uint8), img.affine, img.header)
            nib.save(new_img, tgt_label)

    # ------------------ 设置 dataset.json ------------------
    dataset_json = {
        "channel_names": {
            "0": "T1",
            "1": "T1CE",
            "2": "T2",
            "3": "FLAIR"
        },
        "labels": {
            "background": 0,
            "NCR_NET": 1,
            "ED": 2,
            "ET": 3
        },
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "reference": "MICCAI BraTS 2021",
        "release": "2021",
        "description": "BraTS dataset converted for nnU-Net v2"
    }

    with open(os.path.join(target_base, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=4)
        
    print(f"转换完成！数据集配置已保存至 {os.path.join(target_base, 'dataset.json')}")

if __name__ == "__main__":
    WORKSPACE = "/home/cdy/workspace/brain_project"
    BRATS_PATH = os.path.join(WORKSPACE, "data", "BraTS2021")
    NNUNET_RAW = os.path.join(WORKSPACE, "nnUNet_raw")
    
    # 执行格式转换
    convert_brats_to_nnunetv2(BRATS_PATH, NNUNET_RAW, dataset_id=137)
