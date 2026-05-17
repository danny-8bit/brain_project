import os
import json
import argparse
import numpy as np


def load_case_ids(split_json, cache_dir):
    if os.path.exists(split_json):
        with open(split_json, "r", encoding="utf-8") as f:
            split = json.load(f)
        case_ids = split["val"]
    else:
        label_files = [
            f for f in os.listdir(cache_dir)
            if f.endswith("_label.npy")
        ]
        case_ids = [
            f.replace("_label.npy", "")
            for f in label_files
        ]

    return sorted(case_ids)


def label_cache_path(cache_dir, case_id):
    return os.path.join(cache_dir, f"{case_id}_label.npy")


def output_cache_path(cache_dir, case_id, model_name):
    return os.path.join(cache_dir, f"{case_id}_{model_name}.npy")


def find_pred_file(pred_dir, case_id):
    candidates = [
        os.path.join(pred_dir, f"{case_id}.npz"),
        os.path.join(pred_dir, f"{case_id}.npy"),
        os.path.join(pred_dir, f"{case_id}.nii.gz"),
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        f"找不到 {case_id} 的预测文件，尝试过:\n" + "\n".join(candidates)
    )


def load_npz_array(path):
    z = np.load(path)

    for key in ["probabilities", "softmax", "arr_0", "data", "pred"]:
        if key in z:
            return z[key]

    keys = list(z.keys())
    raise KeyError(f"{path} 中找不到概率数组，可用 keys={keys}")


def hard_seg_to_region_prob(seg):
    """
    支持两种标签:
    1. BraTS 原始: 0,1,2,4
    2. nnU-Net remap 后: 0,1,2,3
    """
    seg = seg.astype(np.uint8)

    if np.any(seg == 4):
        et = seg == 4
    else:
        et = seg == 3

    tc = np.logical_or(seg == 1, et)
    wt = seg > 0

    region = np.stack(
        [
            tc.astype(np.float32),
            wt.astype(np.float32),
            et.astype(np.float32)
        ],
        axis=0
    )

    return region


def softmax_to_region_prob(arr):
    """
    输入:
        4-class softmax: [4,D,H,W]
            0 background
            1 NCR/NET
            2 ED
            3 ET

    输出:
        [3,D,H,W]
            0 TC = class1 + class3
            1 WT = class1 + class2 + class3
            2 ET = class3
    """

    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 4 and arr.shape[0] == 3:
        # 已经是 TC/WT/ET
        return np.clip(arr.astype(np.float32), 0.0, 1.0)

    if arr.ndim == 4 and arr.shape[0] == 4:
        p_bg = arr[0]
        p_1 = arr[1]
        p_2 = arr[2]
        p_et = arr[3]

        p_tc = p_1 + p_et
        p_wt = p_1 + p_2 + p_et

        region = np.stack(
            [
                p_tc,
                p_wt,
                p_et
            ],
            axis=0
        )

        return np.clip(region.astype(np.float32), 0.0, 1.0)

    if arr.ndim == 3:
        return hard_seg_to_region_prob(arr)

    raise ValueError(f"无法识别预测数组 shape={arr.shape}")


def load_prediction(path):
    if path.endswith(".npz"):
        arr = load_npz_array(path)
        return softmax_to_region_prob(arr)

    if path.endswith(".npy"):
        arr = np.load(path)
        return softmax_to_region_prob(arr)

    if path.endswith(".nii.gz"):
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("读取 .nii.gz 需要安装 nibabel: pip install nibabel")

        seg = nib.load(path).get_fdata().astype(np.uint8)
        return hard_seg_to_region_prob(seg)

    raise ValueError(f"不支持的预测文件类型: {path}")


def resize_region_prob_if_needed(region_prob, target_shape, allow_resize):
    """
    region_prob: [3,D,H,W]
    target_shape: [3,D,H,W]
    """

    if region_prob.shape == target_shape:
        return region_prob

    if region_prob.shape[0] != 3:
        raise ValueError(f"region_prob channel 应为 3，实际 shape={region_prob.shape}")

    if region_prob.shape[1:] == target_shape[1:]:
        return region_prob

    if not allow_resize:
        raise ValueError(
            f"预测 shape {region_prob.shape} 和 label shape {target_shape} 不一致。\n"
            f"如果你确认只是空间尺寸差异，可以加 --allow_resize"
        )

    try:
        from scipy.ndimage import zoom
    except ImportError:
        raise ImportError("resize 需要 scipy: pip install scipy")

    zoom_factors = [
        target_shape[1] / region_prob.shape[1],
        target_shape[2] / region_prob.shape[2],
        target_shape[3] / region_prob.shape[3],
    ]

    resized = []

    for c in range(3):
        resized_c = zoom(
            region_prob[c],
            zoom=zoom_factors,
            order=1
        )
        resized.append(resized_c)

    resized = np.stack(resized, axis=0)

    return np.clip(resized.astype(np.float32), 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["nnunet", "mednext", "umamba", "custom"],
        help="写入缓存时使用的模型名"
    )

    parser.add_argument(
        "--pred_dir",
        type=str,
        required=True,
        help="nnU-Net / MedNeXt 预测输出目录"
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./result/prob_cache_holdout_v1",
        help="v1/v2 概率缓存目录"
    )

    parser.add_argument(
        "--split_json",
        type=str,
        default="./splits_holdout_seed42.json",
        help="验证集划分文件"
    )

    parser.add_argument(
        "--allow_resize",
        action="store_true",
        help="如果预测 shape 和 label shape 不一致，允许线性 resize"
    )

    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    case_ids = load_case_ids(args.split_json, args.cache_dir)

    print(f"模型名: {args.model_name}")
    print(f"预测目录: {args.pred_dir}")
    print(f"缓存目录: {args.cache_dir}")
    print(f"病例数: {len(case_ids)}")

    success = 0
    failed = 0

    for case_id in case_ids:
        try:
            label_path = label_cache_path(args.cache_dir, case_id)

            if not os.path.exists(label_path):
                print(f"跳过 {case_id}: 找不到 label cache {label_path}")
                failed += 1
                continue

            label = np.load(label_path)

            if label.ndim == 5 and label.shape[0] == 1:
                label = label[0]

            pred_path = find_pred_file(args.pred_dir, case_id)
            region_prob = load_prediction(pred_path)

            region_prob = resize_region_prob_if_needed(
                region_prob=region_prob,
                target_shape=label.shape,
                allow_resize=args.allow_resize
            )

            out_path = output_cache_path(
                args.cache_dir,
                case_id,
                args.model_name
            )

            np.save(out_path, region_prob.astype(np.float16))
            success += 1

        except Exception as e:
            print(f"❌ {case_id} 转换失败: {e}")
            failed += 1

    print("\n转换完成")
    print(f"成功: {success}")
    print(f"失败: {failed}")


if __name__ == "__main__":
    main()