#!/usr/bin/env python3
"""
F6 论文图：Dice 箱线图（按区域 × 模型分组）

从 per_case_dice.csv 生成分组箱线图，展示：
  - 每个模型在 TC/WT/ET/Mean 上的 Dice 分布
  - 中位数 / 均值 / 异常值
  - 模型之间的统计显著性（paired Wilcoxon）

用法:
  python scripts/plot_dice_boxplot.py \
      --csv paper_data/csv/all_per_case_dice.csv \
      --output paper_figs/F6_dice_boxplot.png

  # 小提琴图（更平滑）
  python scripts/plot_dice_boxplot.py --violin

  # 叠加散点
  python scripts/plot_dice_boxplot.py --strip

  # 过滤 EMA + TTA
  python scripts/plot_dice_boxplot.py --filter "use_ema=True,use_tta=True"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ===== 配色 =====
MODEL_COLORS = {
    "segresnet": "#2E86AB",
    "swinunetr": "#E63946",
    "mednext":   "#06A77D",
    "fused":     "#9B59B6",
    "ensemble":  "#9B59B6",
}

REGION_LABELS = ["WT", "TC", "ET", "Mean"]
REGION_COLS = ["dice_wt", "dice_tc", "dice_et", "dice_mean"]


def parse_filter(s: str) -> dict:
    """'use_ema=True,fold=0' → {'use_ema': True, 'fold': 0}"""
    if not s:
        return {}
    out = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        k = k.strip()
        v = v.strip()
        # 自动类型转换
        if v.lower() == "true":   v = True
        elif v.lower() == "false": v = False
        elif v.lstrip("-").isdigit(): v = int(v)
        else:
            try: v = float(v)
            except ValueError: pass
        out[k] = v
    return out


def apply_filter(df: pd.DataFrame, filt: dict) -> pd.DataFrame:
    for k, v in filt.items():
        if k not in df.columns:
            print(f"  ⚠ 过滤键 '{k}' 不在 CSV 中，忽略")
            continue
        before = len(df)
        df = df[df[k] == v]
        print(f"  过滤 {k}={v}: {before} → {len(df)} 行")
    return df


def compute_significance(df: pd.DataFrame, models: list, region_col: str) -> dict:
    """
    对每对模型做 paired Wilcoxon signed-rank 检验
    返回 {(m1, m2): (stat, p_value)}
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return {}

    results = {}
    # 把数据 pivot 成 case × model 矩阵
    pivot = df.pivot_table(index="case_id", columns="model",
                           values=region_col, aggfunc="mean")

    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            if m1 not in pivot.columns or m2 not in pivot.columns:
                continue
            paired = pivot[[m1, m2]].dropna()
            if len(paired) < 5:
                continue
            try:
                stat, p = wilcoxon(paired[m1], paired[m2])
                results[(m1, m2)] = (stat, p)
            except Exception:
                pass
    return results


def sig_marker(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    parser.add_argument("--csv", default="paper_data/csv/all_per_case_dice.csv",
                        help="per-case Dice CSV 路径")
    parser.add_argument("--output", default="paper_figs/F6_dice_boxplot.png")
    parser.add_argument("--models", default=None,
                        help="逗号分隔，限定模型；不指定则用 CSV 中全部")
    parser.add_argument("--filter", default=None,
                        help="过滤条件，如 'use_ema=True,use_tta=True'")
    parser.add_argument("--violin", action="store_true",
                        help="用小提琴图代替箱线图")
    parser.add_argument("--strip", action="store_true",
                        help="叠加散点（小提琴图自动开启）")
    parser.add_argument("--ymin", type=float, default=0.5,
                        help="Y 轴下限（默认 0.5）")
    parser.add_argument("--ymax", type=float, default=1.005,
                        help="Y 轴上限")
    parser.add_argument("--no_sig", action="store_true",
                        help="不画显著性标记")
    parser.add_argument("--no_pdf", action="store_true")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"✗ CSV 不存在: {csv_path}\n"
                 f"  请先运行推理生成 per-case CSV，或检查路径")

    # ===== 加载 + 过滤 =====
    df = pd.read_csv(csv_path)
    print(f"✓ 加载 CSV: {len(df)} 行")

    if args.filter:
        df = apply_filter(df, parse_filter(args.filter))

    if args.models:
        models = [m.strip() for m in args.models.split(",")]
        df = df[df["model"].isin(models)]
    else:
        models = sorted(df["model"].unique())

    if len(df) == 0:
        sys.exit("✗ 过滤后无数据")

    print(f"  模型: {models}")
    print(f"  case 数: {df['case_id'].nunique()}")

    # ===== 准备绘图数据 =====
    # data[region_idx][model_idx] = list of dice values
    n_regions = len(REGION_LABELS)
    n_models = len(models)

    data = [[[] for _ in models] for _ in REGION_LABELS]
    for ri, col in enumerate(REGION_COLS):
        for mi, m in enumerate(models):
            data[ri][mi] = df.loc[df["model"] == m, col].dropna().values

    # ===== 绘图 =====
    fig, ax = plt.subplots(figsize=(max(8, n_regions * 2.5), 6))

    # 计算每个箱子的 x 位置
    group_width = 0.8
    box_width = group_width / n_models
    positions = []
    for ri in range(n_regions):
        for mi in range(n_models):
            x = ri + (mi - (n_models - 1) / 2) * box_width
            positions.append(x)

    # 收集要画的盒子参数
    flat_data = []
    flat_colors = []
    for ri in range(n_regions):
        for mi in range(n_models):
            flat_data.append(data[ri][mi])
            flat_colors.append(MODEL_COLORS.get(models[mi], "#888888"))

    # 画箱线图或小提琴图
    if args.violin:
        parts = ax.violinplot(flat_data, positions=positions,
                              widths=box_width * 0.9,
                              showmeans=False, showmedians=True,
                              showextrema=True)
        for pc, color in zip(parts['bodies'], flat_colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.65)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.8)
        for partname in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
            if partname in parts:
                parts[partname].set_color('black')
                parts[partname].set_linewidth(1.0)
    else:
        bp = ax.boxplot(flat_data, positions=positions,
                        widths=box_width * 0.8,
                        patch_artist=True,
                        showfliers=True,
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='white',
                                       markeredgecolor='black', markersize=6),
                        medianprops=dict(color='black', linewidth=1.5),
                        flierprops=dict(marker='o', markerfacecolor='red',
                                        markersize=4, markeredgecolor='darkred',
                                        alpha=0.6),
                        boxprops=dict(linewidth=0.8),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8))
        for patch, color in zip(bp['boxes'], flat_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

    # 叠加散点
    if args.strip or args.violin:
        np.random.seed(0)
        for pos, vals, color in zip(positions, flat_data, flat_colors):
            jitter = np.random.uniform(-box_width * 0.15, box_width * 0.15, len(vals))
            ax.scatter(np.full_like(vals, pos, dtype=float) + jitter, vals,
                       s=8, color=color, alpha=0.35, zorder=2,
                       edgecolors='none')

    # 均值文字标注
    for pos, vals in zip(positions, flat_data):
        if len(vals) > 0:
            mean_val = np.mean(vals)
            ax.text(pos, args.ymax - 0.005, f'{mean_val:.3f}',
                    ha='center', va='top', fontsize=7,
                    color='black', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15',
                              facecolor='white', edgecolor='none', alpha=0.7))

    # X 轴
    ax.set_xticks(range(n_regions))
    ax.set_xticklabels(REGION_LABELS, fontsize=12, fontweight='bold')
    ax.set_xlabel("Region", fontsize=12)
    ax.set_ylabel("Dice Score", fontsize=12)
    ax.set_ylim(args.ymin, args.ymax)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    # 区域之间加竖虚线
    for ri in range(1, n_regions):
        ax.axvline(ri - 0.5, color='gray', linestyle=':', alpha=0.4, linewidth=0.7)

    # 图例
    legend_elements = [
        Patch(facecolor=MODEL_COLORS.get(m, "#888"), alpha=0.75,
              edgecolor='black', linewidth=0.8, label=m)
        for m in models
    ]
    ax.legend(handles=legend_elements, loc='lower right',
              fontsize=10, frameon=True, ncol=min(len(models), 3))

    # 标题
    n_cases = df['case_id'].nunique()
    plot_type = "Violin Plot" if args.violin else "Box Plot"
    ax.set_title(f"Dice Score Distribution by Region — {plot_type}  "
                 f"(n={n_cases} cases)",
                 fontsize=13, fontweight='bold', pad=10)

    # ===== 统计显著性 =====
    if not args.no_sig and n_models >= 2:
        print("\n=== 统计显著性 (paired Wilcoxon) ===")
        for ri, (col, label) in enumerate(zip(REGION_COLS, REGION_LABELS)):
            results = compute_significance(df, models, col)
            if not results:
                continue
            # 找最显著的一对，画在该区域上方
            y_top = max(np.percentile(d, 100) if len(d) else 0
                        for d in data[ri]) + 0.015
            y_offset = 0
            for (m1, m2), (stat, p) in results.items():
                marker = sig_marker(p)
                print(f"  [{label}] {m1} vs {m2}: p={p:.4f} {marker}")
                if marker == "ns":
                    continue
                mi1, mi2 = models.index(m1), models.index(m2)
                x1 = ri + (mi1 - (n_models - 1) / 2) * box_width
                x2 = ri + (mi2 - (n_models - 1) / 2) * box_width
                y = y_top + y_offset
                if y > args.ymax - 0.01:
                    continue
                ax.plot([x1, x1, x2, x2], [y - 0.005, y, y, y - 0.005],
                        color='black', linewidth=0.8)
                ax.text((x1 + x2) / 2, y + 0.001, marker,
                        ha='center', va='bottom', fontsize=10,
                        fontweight='bold')
                y_offset += 0.025

    # ===== 保存 =====
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=args.dpi, bbox_inches='tight',
                pad_inches=0.15, facecolor='white')
    print(f"\n✓ 已保存: {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")

    if not args.no_pdf:
        pdf_path = out_path.with_suffix('.pdf')
        plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.15,
                    facecolor='white')
        print(f"  PDF: {pdf_path}")

    # ===== 摘要表 =====
    print("\n=== 摘要（mean ± std）===")
    print(f"{'Model':<14}{'WT':<18}{'TC':<18}{'ET':<18}{'Mean':<18}")
    for m in models:
        sub = df[df["model"] == m]
        row = f"{m:<14}"
        for col in REGION_COLS:
            mu = sub[col].mean()
            sd = sub[col].std()
            row += f"{mu:.4f}±{sd:.4f}   "
        print(row)


if __name__ == "__main__":
    main()
