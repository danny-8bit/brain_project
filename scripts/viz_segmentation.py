#!/usr/bin/env python3
"""
F5 论文核心图：定性分割结果对比
生成 N 个 case 的横向对比：
  T1ce | FLAIR | GT | SegResNet | SwinUNETR | Ensemble

颜色：
  - 绿色 = ED (周围水肿)
  - 黄色 = NCR (坏死核心)
  - 红色 = ET (增强肿瘤)

用法：
  # 指定 case
  python scripts/viz_segmentation.py \
      --case_ids BraTS2021_00001,BraTS2021_00007,BraTS2021_00015 \
      --fold 0 --output paper_figs/F5_qualitative.png

  # 自动从 CSV 选 N 个代表性 case（低/中/高 Dice）
  python scripts/viz_segmentation.py \
      --csv paper_data/all_per_case_dice.csv \
      --auto 5 --output paper_figs/F5_qualitative.png

  # 仅看 GT（测试用，无需预测结果）
  python scripts/viz_segmentation.py \
      --case_ids BraTS2021_00001 \
      --gt_only --output paper_figs/test.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ===== 配色（论文级别）=====
# BraTS 三区域分层颜色：ED 绿 / NCR 黄 / ET 红
COLOR_MAP = {
    1: (0.13, 0.82, 0.31, 0.55),   # ED only - 绿色
    2: (1.00, 0.85, 0.00, 0.65),   # NCR     - 黄色
    3: (0.95, 0.10, 0.10, 0.75),   # ET      - 红色
}

LABEL_NAMES = {1: "ED", 2: "NCR", 3: "ET"}


# ===== 数据加载 =====

def load_nifti(path: Path) -> np.ndarray:
    """加载 NIfTI，返回 (H, W, D)"""
    return nib.load(str(path)).get_fdata()


def load_case_data(data_dir: Path, case_id: str):
    """加载一个 case 的所有模态 + GT"""
    case_dir = data_dir / case_id
    if not case_dir.exists():
        raise FileNotFoundError(f"Case not found: {case_dir}")

    return {
        "t1ce":  load_nifti(case_dir / f"{case_id}_t1ce.nii.gz"),
        "flair": load_nifti(case_dir / f"{case_id}_flair.nii.gz"),
        "gt":    load_nifti(case_dir / f"{case_id}_seg.nii.gz"),
    }


def load_prob(probs_dir: Path, model: str, fold: int, case_id: str):
    """加载预测概率图，返回 (3, H, W, D) 的 float32"""
    npz_path = probs_dir / model / f"fold_{fold}" / f"{case_id}.npz"
    if not npz_path.exists():
        # 尝试 OOF 目录
        npz_path = probs_dir / f"{model}_oof" / f"{case_id}.npz"
        if not npz_path.exists():
            return None
    data = np.load(npz_path)
    prob = data["prob"]
    if prob.dtype == np.uint8:
        prob = prob.astype(np.float32) / 255.0
    return prob


# ===== 标签转换 =====

def gt_to_regions(gt: np.ndarray) -> np.ndarray:
    """
    BraTS GT (0/1/2/4) → 可视化分层标签 (0/1/2/3)
      0 背景
      1 ED only  (绿)
      2 NCR      (黄)
      3 ET       (红)
    """
    out = np.zeros_like(gt, dtype=np.uint8)
    out[gt == 2] = 1   # ED
    out[gt == 1] = 2   # NCR
    out[gt == 4] = 3   # ET
    return out


def prob_to_regions(prob: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """
    模型输出 (3, H, W, D) [TC, WT, ET] → 可视化标签 (0/1/2/3)
    """
    tc = prob[0] > thr
    wt = prob[1] > thr
    et = prob[2] > thr

    out = np.zeros(prob.shape[1:], dtype=np.uint8)
    out[wt & ~tc] = 1   # ED only
    out[tc & ~et] = 2   # NCR
    out[et] = 3         # ET
    return out


# ===== 切片选择 =====

def find_best_slice(seg: np.ndarray) -> int:
    """选肿瘤最大的 axial slice"""
    areas = (seg > 0).sum(axis=(0, 1))
    if areas.sum() == 0:
        return seg.shape[2] // 2
    return int(np.argmax(areas))


# ===== 绘图 =====

def overlay_on_mri(ax, mri: np.ndarray, seg: np.ndarray, title: str = ""):
    """
    在 MRI 灰度图上叠加彩色分割掩码
    mri: (H, W) 2D slice
    seg: (H, W) 2D label slice (0/1/2/3)
    """
    # MRI 归一化（用 1~99 百分位避免极值）
    valid = mri[mri > 0]
    if len(valid) > 0:
        vmin, vmax = np.percentile(valid, [1, 99])
    else:
        vmin, vmax = 0, 1

    # 显示 MRI（转置 + origin='lower' = 放射学惯例）
    ax.imshow(mri.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax,
              interpolation='bilinear')

    # 构造 RGBA 叠加层
    H, W = seg.shape
    overlay = np.zeros((W, H, 4))   # 转置匹配
    for label, color in COLOR_MAP.items():
        mask = (seg == label).T
        overlay[mask] = color

    ax.imshow(overlay, origin='lower', interpolation='nearest')

    if title:
        ax.set_title(title, fontsize=11, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ===== Case 选择 =====

def select_cases_from_csv(csv_path: Path, n: int) -> list:
    """从 per_case_dice CSV 选 N 个代表性 case（按 Dice 分位数均匀采样）"""
    import pandas as pd
    df = pd.read_csv(csv_path)
    # 取每个 case 在所有 fold/model 上的平均 Dice
    case_mean = df.groupby("case_id")["dice_mean"].mean().reset_index()
    case_mean = case_mean.sort_values("dice_mean").reset_index(drop=True)

    if n == 1:
        idx = [len(case_mean) // 2]
    else:
        # 均匀分位采样
        quantiles = np.linspace(0.1, 0.9, n)
        idx = (quantiles * (len(case_mean) - 1)).astype(int)

    selected = case_mean.iloc[idx]
    print(f"自动选择 {n} 个 case（按 Dice 分位）:")
    for _, row in selected.iterrows():
        print(f"  {row['case_id']}  Dice={row['dice_mean']:.4f}")
    return selected["case_id"].tolist()


# ===== 主函数 =====

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)
    parser.add_argument("--data_dir", default="./data/BraTS2021_TrainingData",
                        help="BraTS 训练数据根目录")
    parser.add_argument("--case_ids", default=None,
                        help="逗号分隔的 case_id，如 BraTS2021_00001,BraTS2021_00007")
    parser.add_argument("--csv", default=None,
                        help="per_case_dice.csv 路径（用于自动选 case）")
    parser.add_argument("--auto", type=int, default=0,
                        help="自动选 N 个代表性 case")
    parser.add_argument("--probs_dir", default="/dev/shm/brats_probs",
                        help="概率图根目录")
    parser.add_argument("--models", default="segresnet,swinunetr",
                        help="逗号分隔的模型名")
    parser.add_argument("--fold", type=int, default=0,
                        help="加载哪个 fold 的预测")
    parser.add_argument("--output", default="paper_figs/F5_qualitative.png",
                        help="输出 PNG 路径")
    parser.add_argument("--thr", type=float, default=0.5,
                        help="二值化阈值")
    parser.add_argument("--slice", type=int, default=-1,
                        help="指定切片号；-1 = 自动选肿瘤最大切片")
    parser.add_argument("--gt_only", action="store_true",
                        help="仅显示 GT（用于测试，无需预测文件）")
    parser.add_argument("--dpi", type=int, default=200,
                        help="输出 DPI")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    probs_dir = Path(args.probs_dir)

    # 选择 case
    if args.case_ids:
        case_ids = [c.strip() for c in args.case_ids.split(",")]
    elif args.csv and args.auto > 0:
        case_ids = select_cases_from_csv(Path(args.csv), args.auto)
    else:
        sys.exit("请用 --case_ids 或 --csv + --auto 指定 case")

    models = [m.strip() for m in args.models.split(",")]

    # 列定义
    if args.gt_only:
        col_titles = ["T1ce", "FLAIR", "GT"]
    else:
        col_titles = ["T1ce", "FLAIR", "GT"] + models
        # 检查是否有 ensemble
        ensemble_path = probs_dir / "fused"
        if ensemble_path.exists():
            col_titles.append("Ensemble")

    n_rows = len(case_ids)
    n_cols = len(col_titles)

    # 创建画布
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.2 * n_cols, 2.5 * n_rows),
        gridspec_kw={'wspace': 0.02, 'hspace': 0.08}
    )

    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    # 顶行加列标题
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=13, fontweight='bold', pad=8)

    # 逐 case 绘制
    for i, case_id in enumerate(case_ids):
        print(f"[{i+1}/{n_rows}] 处理 {case_id} ...")

        try:
            d = load_case_data(data_dir, case_id)
        except FileNotFoundError as e:
            print(f"  ⚠ {e}")
            continue

        gt_regions = gt_to_regions(d["gt"])

        # 切片选择
        if args.slice >= 0:
            slc = args.slice
        else:
            slc = find_best_slice(gt_regions)

        # 收集所有预测结果（在选切片前完整加载）
        pred_regions = {}
        if not args.gt_only:
            for m in models:
                prob = load_prob(probs_dir, m, args.fold, case_id)
                if prob is not None:
                    pred_regions[m] = prob_to_regions(prob, args.thr)
                else:
                    print(f"  ⚠ {m} fold {args.fold} 预测未找到")

            # Ensemble (如果有)
            if "Ensemble" in col_titles:
                prob_ens = load_prob(probs_dir, "fused", args.fold, case_id)
                if prob_ens is not None:
                    pred_regions["Ensemble"] = prob_to_regions(prob_ens, args.thr)

        # 绘制每一列
        col = 0
        # T1ce
        overlay_on_mri(axes[i, col], d["t1ce"][:, :, slc],
                       np.zeros_like(gt_regions[:, :, slc]))
        col += 1
        # FLAIR
        overlay_on_mri(axes[i, col], d["flair"][:, :, slc],
                       np.zeros_like(gt_regions[:, :, slc]))
        col += 1
        # GT
        overlay_on_mri(axes[i, col], d["t1ce"][:, :, slc],
                       gt_regions[:, :, slc])
        col += 1
        # 模型预测（仅非 gt_only 模式）
        if not args.gt_only:
            for m in models:
                if m in pred_regions:
                    overlay_on_mri(axes[i, col], d["t1ce"][:, :, slc],
                                   pred_regions[m][:, :, slc])
                else:
                    axes[i, col].text(0.5, 0.5, "N/A",
                                      transform=axes[i, col].transAxes,
                                      ha='center', va='center', fontsize=14,
                                      color='gray')
                    axes[i, col].set_xticks([])
                    axes[i, col].set_yticks([])
                col += 1
            # Ensemble
            if "Ensemble" in col_titles:
                if "Ensemble" in pred_regions:
                    overlay_on_mri(axes[i, col], d["t1ce"][:, :, slc],
                                   pred_regions["Ensemble"][:, :, slc])
                else:
                    axes[i, col].text(0.5, 0.5, "N/A",
                                      transform=axes[i, col].transAxes,
                                      ha='center', va='center', fontsize=14,
                                      color='gray')
                    axes[i, col].set_xticks([])
                    axes[i, col].set_yticks([])

        # 左侧加 case_id 标签
        axes[i, 0].set_ylabel(case_id.replace("BraTS2021_", "Case "),
                              fontsize=11, fontweight='bold')

    # 全局图例
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_MAP[1][:3], alpha=COLOR_MAP[1][3],
                       label='ED (Edema)', edgecolor='none'),
        mpatches.Patch(facecolor=COLOR_MAP[2][:3], alpha=COLOR_MAP[2][3],
                       label='NCR (Necrosis)', edgecolor='none'),
        mpatches.Patch(facecolor=COLOR_MAP[3][:3], alpha=COLOR_MAP[3][3],
                       label='ET (Enhancing)', edgecolor='none'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, -0.01))

    # 输出
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_path, dpi=args.dpi, bbox_inches='tight',
                pad_inches=0.15, facecolor='white')
    print(f"\n✓ 已保存: {out_path}")
    print(f"  大小: {out_path.stat().st_size / 1024:.1f} KB")

    # 同时输出 PDF 用于论文
    pdf_path = out_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.15,
                facecolor='white')
    print(f"  PDF: {pdf_path}")


if __name__ == "__main__":
    main()
