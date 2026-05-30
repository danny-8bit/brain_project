"""
单模型推理：对一个 fold 的验证集（或测试集）输出概率图 .npz。
支持 TTA。
[v2] 新增：per-case Dice CSV 输出 + 推理时间记录（论文图用）。
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from monai.data import Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import load_datalist, scan_brats_cases
from src.data.transforms import get_val_transforms, get_test_transforms
from src.models.builder import build_model
from src.tta import sliding_window_tta
from src.metrics import BratsMetrics
from src.utils import load_yaml, merge_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="./configs/base.yaml")
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--fold", type=int, default=None,
                        help="预测某 fold 的验证集；None 则用 --test_dir")
    parser.add_argument("--test_dir", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    # ===== 新增参数 =====
    parser.add_argument("--thr", type=float, default=0.5,
                        help="二值化阈值，用于计算 per-case Dice（默认 0.5）")
    parser.add_argument("--model_name", default="model",
                        help="模型名称，写入 CSV（如 segresnet / swinunetr）")
    parser.add_argument("--save_csv", action="store_true",
                        help="保存 per-case Dice CSV（仅当 --fold 模式有效）")
    args = parser.parse_args()

    cfg = merge_dict(load_yaml(args.base_config), load_yaml(args.model_config))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # 数据
    if args.fold is not None:
        files = load_datalist(cfg["folds_json"], args.fold, "val")
        transform = get_val_transforms()
        has_label = True
    else:
        files = scan_brats_cases(args.test_dir)
        transform = get_test_transforms()
        has_label = False

    ds = Dataset(data=files, transform=transform)
    loader = DataLoader(
        ds, batch_size=1,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        shuffle=False,   # 关键：保持顺序，便于用 idx 取 case_id
    )

    # 模型
    model_kwargs = {k: v for k, v in cfg.items() if k in (
        "init_filters",
        "dropout",
        "feature_size",
        "use_checkpoint",
        "num_res_units",
        "hidden_size",
        "mlp_dim",
        "num_heads",
        "pos_embed",
        "deepmedic_filters",
        "transformer_layers",
        "mlp_ratio",
    )}
    model_kwargs.update(cfg.get("model_kwargs", {}))

    model = build_model(
        cfg["model"], in_channels=4, out_channels=3,
        roi_size=cfg["roi_size"],
        **model_kwargs,
    ).to(device)

    state = torch.load(args.ckpt, map_location=device)
    key = "ema" if (args.use_ema and "ema" in state) else "model"
    model.load_state_dict(state[key])
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== 新增：per-case 结果累积 =====
    per_case_results = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc="Predict", ncols=100)):
            image = batch["image"].to(device)

            # 从原始 files 列表稳定取 case_id（比 image_meta_dict 兼容性更好）
            # 兼容多种数据格式取 case_id
            entry = files[idx]
            fpath = None
            for _k in ["image", "t1ce", "t1", "t2", "flair"]:
                if _k in entry:
                    _v = entry[_k]
                    fpath = _v[0] if isinstance(_v, (list, tuple)) else _v
                    break
            if fpath is None:
                for _kk, _vv in entry.items():
                    if "label" in _kk.lower() or "seg" in _kk.lower():
                        continue
                    fpath = _vv[0] if isinstance(_vv, (list, tuple)) else _vv
                    break
            case_id = Path(fpath).parent.name

            # ===== 推理 + 计时 =====
            t0 = time.time()
            prob = sliding_window_tta(
                image,
                model,
                roi_size=cfg["roi_size"],
                sw_batch_size=cfg["sw_batch_size"],
                overlap=0.5,
                mode="gaussian",
                flips=args.tta,
                sigmoid=True,
            )
            infer_sec = time.time() - t0

            # ===== 保存概率图（原有逻辑）=====
            prob_np = (prob[0].cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
            np.savez_compressed(out_dir / f"{case_id}.npz", prob=prob_np)

            # ===== 新增：计算 per-case Dice =====
            if has_label and args.save_csv and "label" in batch:
                # label shape: (1, 3, D, H, W) - 已经是 3 通道 [TC, WT, ET]
                label_np = batch["label"][0].numpy()
                pred_binary = (prob[0].cpu().numpy() > args.thr).astype(np.uint8)

                m = BratsMetrics()
                m([pred_binary], [label_np])
                r = m.aggregate()

                per_case_results.append({
                    "case_id": case_id,
                    "model": args.model_name,
                    "fold": args.fold,
                    "use_ema": args.use_ema,
                    "use_tta": args.tta,
                    "thr": args.thr,
                    "dice_tc": round(r["dice_tc"], 6),
                    "dice_wt": round(r["dice_wt"], 6),
                    "dice_et": round(r["dice_et"], 6),
                    "dice_mean": round(r["dice_mean"], 6),
                    "infer_sec": round(infer_sec, 2),
                })

    # ===== 保存 CSV =====
    if per_case_results:
        df = pd.DataFrame(per_case_results)
        csv_path = out_dir / "per_case_dice.csv"
        df.to_csv(csv_path, index=False)

        print(f"\n✓ Per-case CSV → {csv_path}")
        print(f"  Cases: {len(df)}")
        print(f"  Mean Dice: TC={df['dice_tc'].mean():.4f}  "
              f"WT={df['dice_wt'].mean():.4f}  "
              f"ET={df['dice_et'].mean():.4f}  "
              f"Overall={df['dice_mean'].mean():.4f}")
        print(f"  Min/Max Mean Dice: {df['dice_mean'].min():.4f} / {df['dice_mean'].max():.4f}")
        print(f"  Avg infer time: {df['infer_sec'].mean():.1f}s")


if __name__ == "__main__":
    main()
