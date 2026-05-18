#!/usr/bin/env python3
"""
F9 论文图：网格搜索热图（融合权重与阈值优化）

可视化集成融合超参数搜索的 OOF Dice 景观。
高维空间用 2D 切片矩阵展示（其他维度固定在最优）。

支持输入格式：
  - JSON: [{"w_seg":0.5, "thr_wt":0.5, "dice_mean":0.91}, ...]
  - CSV:  列名 = 维度名 + 目标值列
  - NPZ:  键 'records' 存 list of dict

用法:
  python scripts/plot_grid_heatmap.py \
      --input paper_data/ensemble/grid_search_results.json \
      --output paper_figs/F9_grid_heatmap.png

  # 指定要画的维度对（默认自动选 4 对）
  python scripts/plot_grid_heatmap.py \
      --pairs "w_seg-thr_wt,thr_wt-thr_et"

  # 生成 demo 数据测试脚本
  python scripts/plot_grid_heatmap.py --demo \
      --output paper_figs/F9_demo.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ===== 配色 =====
HEATMAP_CMAP = "viridis"   # 也可选 'plasma' 'magma' 'cividis'


# ===== 数据加载 =====
def load_grid_data(path: Path) -> pd.DataFrame:
    """支持 JSON / CSV / NPZ 三种格式"""
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "records" in data:
            data = data["records"]
        return pd.DataFrame(data)
    elif suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in (".npz", ".npy"):
        npz = np.load(path, allow_pickle=True)
        if "records" in npz.files:
            return pd.DataFrame(npz["records"].tolist())
        # 其他形式：把每个 key 当列
        return pd.DataFrame({k: npz[k] for k in npz.files})
    else:
        raise ValueError(f"不支持的格式: {suffix}")


def generate_demo_data() -> pd.DataFrame:
    """生成假的网格搜索数据用于测试"""
    np.random.seed(42)
    records = []

    # 真实最优在这里
    true_opt = {"w_seg": 0.45, "thr_wt": 0.50, "thr_tc": 0.50, "thr_et": 0.40}
    peak = 0.918

    for w in np.arange(0.0, 1.01, 0.1):
        for tw in np.arange(0.40, 0.61, 0.05):
            for tt in np.arange(0.40, 0.61, 0.05):
                for te in np.arange(0.30, 0.51, 0.05):
                    # 距离最优的"高斯衰减"
                    d2 = ((w - true_opt["w_seg"]) / 0.3) ** 2 + \
                         ((tw - true_opt["thr_wt"]) / 0.15) ** 2 + \
                         ((tt - true_opt["thr_tc"]) / 0.15) ** 2 + \
                         ((te - true_opt["thr_et"]) / 0.15) ** 2
                    dice = peak - 0.05 * d2 + np.random.normal(0, 0.002)
                    dice = np.clip(dice, 0.7, 0.95)
                    records.append({
                        "w_seg": round(w, 3),
                        "thr_wt": round(tw, 3),
                        "thr_tc": round(tt, 3),
                        "thr_et": round(te, 3),
                        "dice_mean": round(dice, 5),
                        "dice_wt": round(dice + 0.03, 5),
                        "dice_tc": round(dice - 0.005, 5),
                        "dice_et": round(dice - 0.04, 5),
                    })
    return pd.DataFrame(records)


# ===== 绘图 =====
def plot_heatmap_pair(ax, df: pd.DataFrame, dim_x: str, dim_y: str,
                      value_col: str, optimum: dict,
                      annotate: bool = True, vmin: float = None, vmax: float = None):
    """画一个 2D 热图，固定其他维度在最优"""
    # 找出 df 中除了 x, y, value 以外的所有维度
    other_dims = [c for c in df.columns
                  if c not in [dim_x, dim_y, value_col]
                  and c in optimum
                  and not c.startswith("dice_")]

    # 过滤：其他维度都在最优值
    sub = df.copy()
    for d in other_dims:
        # 找最接近最优值的实际取值（避免浮点不匹配）
        unique_vals = sorted(df[d].unique())
        opt_val = optimum[d]
        closest = min(unique_vals, key=lambda x: abs(x - opt_val))
        sub = sub[np.isclose(sub[d], closest, atol=1e-6)]

    if len(sub) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha='center', va='center', fontsize=14)
        return None

    # Pivot 成 2D 矩阵
    pivot = sub.pivot_table(index=dim_y, columns=dim_x,
                            values=value_col, aggfunc="mean")

    # 排序索引（Y 轴反向）
    pivot = pivot.sort_index(ascending=False)
    pivot = pivot[sorted(pivot.columns)]

    # 画热图
    im = ax.imshow(pivot.values, cmap=HEATMAP_CMAP,
                   aspect='auto', vmin=vmin, vmax=vmax,
                   interpolation='nearest')

    # X / Y 轴
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns],
                       fontsize=8, rotation=0)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in pivot.index], fontsize=8)
    ax.set_xlabel(dim_x, fontsize=10)
    ax.set_ylabel(dim_y, fontsize=10)

    # 单元格数值标注
    if annotate and pivot.size <= 200:
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if np.isnan(v):
                    continue
                # 根据背景明暗选择文字颜色
                normalized = (v - (vmin or pivot.values.min())) / \
                             ((vmax or pivot.values.max()) - (vmin or pivot.values.min()) + 1e-9)
                color = 'white' if normalized < 0.5 else 'black'
                ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                        fontsize=6.5, color=color)

    # 标记最优点（矩形边框 + 角上小星）
    if dim_x in optimum and dim_y in optimum:
        from matplotlib.patches import Rectangle
        x_opt = optimum[dim_x]
        y_opt = optimum[dim_y]
        x_idx = np.argmin(np.abs(pivot.columns.values - x_opt))
        y_idx = np.argmin(np.abs(pivot.index.values - y_opt))

        # 红色双层矩形边框（外白内红，提高对比度）
        outer = Rectangle((x_idx - 0.5, y_idx - 0.5), 1, 1,
                          fill=False, edgecolor='white',
                          linewidth=3.5, zorder=9)
        inner = Rectangle((x_idx - 0.5, y_idx - 0.5), 1, 1,
                          fill=False, edgecolor='#E63946',
                          linewidth=2.0, zorder=10)
        ax.add_patch(outer)
        ax.add_patch(inner)

        # 右上角小星星（不遮文字）
        ax.plot(x_idx + 0.36, y_idx - 0.36, marker='*',
                markersize=11, markerfacecolor='#E63946',
                markeredgecolor='white', markeredgewidth=0.8,
                zorder=11, clip_on=False)

    # 其他维度固定值文字
    if other_dims:
        fixed_str = ", ".join(f"{d}={optimum[d]:.2f}" for d in other_dims)
        ax.set_title(f"{dim_y} vs {dim_x}\n(fixed: {fixed_str})",
                     fontsize=10, pad=6)
    else:
        ax.set_title(f"{dim_y} vs {dim_x}", fontsize=11, pad=6)

    return im


def auto_pick_pairs(df: pd.DataFrame, value_col: str) -> list:
    """自动挑选要画的维度对"""
    dims = [c for c in df.columns
            if c != value_col and not c.startswith("dice_")
            and df[c].nunique() > 1]   # 排除常数列

    if "w_seg" in dims:
        # 优先 w_seg 与各阈值的组合
        thr_dims = [d for d in dims if d.startswith("thr")]
        pairs = [("w_seg", d) for d in thr_dims[:2]]
        # 再加阈值之间的组合
        if len(thr_dims) >= 2:
            pairs.append((thr_dims[0], thr_dims[1]))
        if len(thr_dims) >= 3:
            pairs.append((thr_dims[1], thr_dims[2]))
        return pairs[:4]
    else:
        # 两两组合
        pairs = []
        for i, d1 in enumerate(dims):
            for d2 in dims[i+1:]:
                pairs.append((d1, d2))
        return pairs[:4]


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    parser.add_argument("--input", default="paper_data/ensemble/grid_search_results.json",
                        help="网格搜索结果文件 (JSON/CSV/NPZ)")
    parser.add_argument("--output", default="paper_figs/F9_grid_heatmap.png")
    parser.add_argument("--value", default="dice_mean",
                        help="目标值列名（默认 dice_mean）")
    parser.add_argument("--pairs", default=None,
                        help="逗号分隔的维度对，如 'w_seg-thr_wt,thr_wt-thr_et'")
    parser.add_argument("--no_annotate", action="store_true",
                        help="不在 cell 上写数值")
    parser.add_argument("--demo", action="store_true",
                        help="生成 demo 数据测试")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--no_pdf", action="store_true")
    args = parser.parse_args()

    # 加载数据
    if args.demo:
        print("✓ 使用 demo 数据")
        df = generate_demo_data()
    else:
        in_path = Path(args.input)
        if not in_path.exists():
            sys.exit(f"✗ 输入文件不存在: {in_path}\n"
                     f"  提示: 用 --demo 测试，或先跑 grid_search.py")
        df = load_grid_data(in_path)
        print(f"✓ 加载 {len(df)} 行")

    if args.value not in df.columns:
        sys.exit(f"✗ 目标列 '{args.value}' 不在数据中: {list(df.columns)}")

    # 找全局最优
    opt_row = df.loc[df[args.value].idxmax()]
    optimum = {k: opt_row[k] for k in df.columns
               if not k.startswith("dice_")}
    print(f"\n=== 全局最优 ({args.value} = {opt_row[args.value]:.4f}) ===")
    for k, v in optimum.items():
        if isinstance(v, (int, float)):
            print(f"  {k:15s} = {v:.4f}")

    # 决定要画的维度对
    if args.pairs:
        pairs = [tuple(p.split("-")) for p in args.pairs.split(",")]
    else:
        pairs = auto_pick_pairs(df, args.value)

    if not pairs:
        sys.exit("✗ 没有可画的维度对")

    print(f"\n绘制 {len(pairs)} 个 2D 切片:")
    for x, y in pairs:
        print(f"  {y} vs {x}")

    # ===== 画图 =====
    n = len(pairs)
    n_cols = 2 if n > 1 else 1
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6.5 * n_cols, 5.5 * n_rows))
    if n == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    # 用全局 vmin/vmax 保证 colorbar 一致
    vmin, vmax = df[args.value].min(), df[args.value].max()

    last_im = None
    for i, (dim_x, dim_y) in enumerate(pairs):
        r, c = i // n_cols, i % n_cols
        last_im = plot_heatmap_pair(
            axes[r, c], df, dim_x, dim_y, args.value, optimum,
            annotate=not args.no_annotate,
            vmin=vmin, vmax=vmax
        )

    # 隐藏多余 subplot
    for i in range(n, n_rows * n_cols):
        r, c = i // n_cols, i % n_cols
        axes[r, c].axis('off')

    # 全局 colorbar
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(),
                            shrink=0.7, aspect=30, pad=0.02,
                            label=args.value)
        cbar.ax.tick_params(labelsize=9)

    # 全局标题
    fig.suptitle(
        f"Grid Search Landscape  (★ = optimum: {args.value} = {opt_row[args.value]:.4f})",
        fontsize=13, fontweight='bold', y=1.0
    )

    # ===== 保存 =====
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=args.dpi, bbox_inches='tight',
                pad_inches=0.15, facecolor='white')
    print(f"\n✓ 已保存: {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")

    if not args.no_pdf:
        pdf_path = out_path.with_suffix('.pdf')
        plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.15,
                    facecolor='white')
        print(f"  PDF: {pdf_path}")

    # ===== Top-K 摘要表 =====
    print("\n=== Top-10 配置 ===")
    top = df.nlargest(10, args.value)
    show_cols = [c for c in df.columns if not c.startswith("dice_")
                 or c == args.value]
    print(top[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
