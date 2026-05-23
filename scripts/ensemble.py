"""
模型融合：读取多个模型输出的概率图 .npz，加权平均后做后处理，
保存最终标签 nii.gz 文件。
"""

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.postprocess import postprocess_brats


def parse_weights(s):
    return [float(x) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob_dirs", nargs="+", required=True,
                        help="多个 predict 输出目录")
    parser.add_argument("--weights", type=parse_weights, default=None,
                        help="逗号分隔，例如 0.5,0.3,0.2")
    parser.add_argument("--ref_dir", required=True,
                        help="原始数据目录，用于读取 affine/header 还原")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--config", default=None,
                        help="grid_search best_config.json; overrides weights/thresholds/min_sizes")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--et_threshold_voxels", type=int, default=200)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text()) if args.config else {}

    prob_dirs = [Path(p) for p in args.prob_dirs]
    if cfg.get("weights"):
        model_names = [d.name.replace("_oof", "") for d in prob_dirs]
        weights = [cfg["weights"].get(name) for name in model_names]
        if any(w is None for w in weights):
            weights = list(cfg["weights"].values())
    elif args.weights is None:
        weights = [1.0 / len(prob_dirs)] * len(prob_dirs)
    else:
        weights = args.weights
    s = sum(weights)
    weights = [w / s for w in weights]
    thresholds = cfg.get("thresholds")
    min_sizes = cfg.get("min_sizes")
    et_threshold_voxels = cfg.get("et_threshold_voxels", args.et_threshold_voxels)
    et_min_prob = cfg.get("et_min_prob", 0.5)
    print(f"Weights: {weights}")

    # 取第一个目录的 case 列表
    cases = sorted([p.stem for p in prob_dirs[0].glob("*.npz")])
    print(f"Cases: {len(cases)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_root = Path(args.ref_dir)

    for case_id in tqdm(cases, desc="Ensemble", ncols=100):
        probs = []
        for d in prob_dirs:
            arr_raw = np.load(d / f"{case_id}.npz")["prob"]
            if arr_raw.dtype == np.uint8:
                arr = arr_raw.astype(np.float32) / 255.0
            else:
                arr = arr_raw.astype(np.float32)
            probs.append(arr)

        # 加权平均
        fused = np.zeros_like(probs[0])
        for w, p in zip(weights, probs):
            fused += w * p

        # 后处理
        label = postprocess_brats(
            fused,
            threshold=args.threshold,
            thresholds=thresholds,
            min_sizes=min_sizes,
            et_threshold_voxels=et_threshold_voxels,
            et_min_prob=et_min_prob,
        )

        # 还原到原始空间：这里假设 predict 已经是原始空间
        # 读取参考 nii.gz 取 affine/header
        ref_case_dir = ref_root / case_id
        ref_nii = list(ref_case_dir.glob(f"{case_id}_t1.nii*"))[0]
        ref_img = nib.load(str(ref_nii))

        # 注意：如果训练时 spacing 改变了，需要重采样回原 spacing。
        # 简单起见此处假设原始数据本身就是 1mm 各向同性 (BraTS 标准)。
        out = nib.Nifti1Image(label.astype(np.uint8), ref_img.affine, ref_img.header)
        nib.save(out, str(out_dir / f"{case_id}.nii.gz"))


if __name__ == "__main__":
    main()