#!/usr/bin/env python3
"""
F4 论文训练曲线图

从 logs/train_{model}_fold{k}.log 解析:
  - 训练 loss / 学习率（每 epoch）
  - 验证 Dice (mean/TC/WT/ET)（每 N epoch）
  - Best dice 标记
  - SWA 开始位置（如有）

输出 2x2 网格：训练 Loss / 平均 Val Dice / 三区域 Dice / 学习率
所有 fold 取均值，阴影显示 ±1 std。

用法:
  # 默认（自动检测 logs/ 下所有模型）
  python scripts/plot_training_curves.py

  # 指定模型对比
  python scripts/plot_training_curves.py --models segresnet,swinunetr

  # 单模型单 fold（训练中快速看）
  python scripts/plot_training_curves.py --models segresnet --folds 0
"""

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ===== 正则：匹配训练日志中的关键行 =====
RE_EPOCH = re.compile(
    r"Epoch\s+(\d+)\s*/\s*(\d+)\s*\|\s*loss[=:]([\d.eE+-]+)"
    r".*?lr[=:]([\d.eE+-]+)"
    r"(?:.*?time[=:]([\d.]+))?",
    re.IGNORECASE
)

RE_VAL = re.compile(
    r"\[?Val\s+ep[=:]?(\d+)\]?\s*"
    r".*?mean[=:]([\d.]+)"
    r".*?TC[=:]([\d.]+)"
    r".*?WT[=:]([\d.]+)"
    r".*?ET[=:]([\d.]+)",
    re.IGNORECASE
)

RE_BEST = re.compile(
    r"Best\s+dice[=:]([\d.]+)\s+at\s+epoch\s+(\d+)",
    re.IGNORECASE
)

RE_SWA = re.compile(
    r"(?:SWA\s+start|swa_start).*?epoch\s+(\d+)",
    re.IGNORECASE
)


# ===== 配色（学术风）=====
MODEL_COLORS = {
    "segresnet": "#2E86AB",   # 深蓝
    "swinunetr": "#E63946",   # 砖红
    "mednext":   "#06A77D",   # 翠绿
    "fused":     "#9B59B6",   # 紫
}

REGION_COLORS = {
    "wt": "#1f77b4",   # 蓝
    "tc": "#ff7f0e",   # 橙
    "et": "#2ca02c",   # 绿
}


def parse_log(log_path: Path) -> dict:
    """
    解析单个 log 文件，返回 dict：
      train: {epoch: {loss, lr, time}}
      val:   {epoch: {mean, tc, wt, et}}
      best:  (epoch, dice)
      swa:   epoch or None
    """
    train = {}
    val = {}
    best = None
    swa = None

    if not log_path.exists():
        return None

    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = RE_EPOCH.search(line)
            if m:
                ep = int(m.group(1))
                train[ep] = {
                    "loss": float(m.group(3)),
                    "lr": float(m.group(4)),
                    "time": float(m.group(5)) if m.group(5) else 0.0,
                }
                continue

            m = RE_VAL.search(line)
            if m:
                ep = int(m.group(1))
                val[ep] = {
                    "mean": float(m.group(2)),
                    "tc": float(m.group(3)),
                    "wt": float(m.group(4)),
                    "et": float(m.group(5)),
                }
                continue

            m = RE_BEST.search(line)
            if m:
                d, ep = float(m.group(1)), int(m.group(2))
                if best is None or d > best[1]:
                    best = (ep, d)
                continue

            m = RE_SWA.search(line)
            if m and swa is None:
                swa = int(m.group(1))

    return {"train": train, "val": val, "best": best, "swa": swa}


def find_log(logs_dir: Path, work_dir: Path, model: str, fold: int) -> Path:
    """智能查找日志位置"""
    candidates = [
        logs_dir / f"train_{model}_fold{fold}.log",
        logs_dir / f"train_{model}_fold_{fold}.log",
        work_dir / model / f"fold_{fold}" / "train.log",
        work_dir / model / f"fold{fold}" / "train.log",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def aggregate_folds(fold_data_list: list, key_chain: list) -> tuple:
    """
    跨 fold 聚合：返回 (epochs, mean, std)
    key_chain: e.g. ["train", "loss"] 或 ["val", "mean"]
    """
    all_series = []
    for d in fold_data_list:
        if d is None:
            continue
        source = d[key_chain[0]]
        # 提取 epoch -> value
        series = {ep: v[key_chain[1]] for ep, v in source.items()
                  if key_chain[1] in v}
        if series:
            all_series.append(series)

    if not all_series:
        return np.array([]), np.array([]), np.array([])

    # 取所有 fold 都有的 epoch（不要 NaN）
    all_epochs = sorted(set().union(*[s.keys() for s in all_series]))

    means, stds = [], []
    for ep in all_epochs:
        vals = [s[ep] for s in all_series if ep in s]
        means.append(np.mean(vals))
        stds.append(np.std(vals) if len(vals) > 1 else 0.0)

    return np.array(all_epochs), np.array(means), np.array(stds)


def plot_curve(ax, epochs, mean, std, label, color, linewidth=2, alpha_fill=0.2):
    """画一条带阴影带的曲线"""
    if len(epochs) == 0:
        return
    ax.plot(epochs, mean, color=color, linewidth=linewidth, label=label)
    if (std > 0).any():
        ax.fill_between(epochs, mean - std, mean + std,
                        color=color, alpha=alpha_fill)


def smooth(x, window=5):
    """简单移动平均平滑"""
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode='valid')


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)
    parser.add_argument("--logs_dir", default="./logs",
                        help="日志目录")
    parser.add_argument("--work_dir", default="./work_dir",
                        help="work_dir 目录（备用日志位置）")
    parser.add_argument("--models", default="segresnet,swinunetr",
                        help="逗号分隔模型名")
    parser.add_argument("--folds", default="0,1,2,3,4",
                        help="逗号分隔 fold 编号")
    parser.add_argument("--output", default="paper_figs/F4_training_curves.png",
                        help="输出路径")
    parser.add_argument("--smooth_loss", type=int, default=5,
                        help="loss 曲线移动平均窗口（0=不平滑）")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--no_pdf", action="store_true")
    parser.add_argument("--no_swa_marker", action="store_true",
                        help="不画 SWA 起点竖线")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    work_dir = Path(args.work_dir)
    models = [m.strip() for m in args.models.split(",")]
    folds = [int(f) for f in args.folds.split(",")]

    # ===== 加载所有数据 =====
    all_data = {}   # all_data[model] = [parsed_fold_0, parsed_fold_1, ...]
    for model in models:
        all_data[model] = []
        for fold in folds:
            log = find_log(logs_dir, work_dir, model, fold)
            if log is None:
                print(f"  ⚠ 跳过 {model} fold {fold}（无日志）")
                all_data[model].append(None)
                continue
            data = parse_log(log)
            n_ep = len(data["train"]) if data else 0
            n_val = len(data["val"]) if data else 0
            print(f"  ✓ {model} fold {fold}: {n_ep} train ep, {n_val} val ep "
                  f"[{log}]")
            all_data[model].append(data)

    # 检查是否有任何数据
    has_data = any(any(d is not None for d in folds_data)
                   for folds_data in all_data.values())
    if not has_data:
        sys.exit("✗ 没找到任何日志文件，请检查路径")

    # ===== 画图 =====
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    plt.subplots_adjust(hspace=0.30, wspace=0.25)

    # ----- [0,0] 训练 Loss -----
    ax = axes[0, 0]
    for model in models:
        ep, mu, sd = aggregate_folds(all_data[model], ["train", "loss"])
        if len(ep) == 0:
            continue
        # 平滑
        if args.smooth_loss > 1 and len(mu) > args.smooth_loss:
            k = args.smooth_loss
            mu_s = np.convolve(mu, np.ones(k) / k, mode='valid')
            sd_s = np.convolve(sd, np.ones(k) / k, mode='valid')
            ep_s = ep[k - 1:]
            plot_curve(ax, ep_s, mu_s, sd_s, model,
                       MODEL_COLORS.get(model, "gray"))
        else:
            plot_curve(ax, ep, mu, sd, model,
                       MODEL_COLORS.get(model, "gray"))

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Training Loss", fontsize=11)
    ax.set_title("(a) Training Loss", fontsize=12, fontweight='bold')
    ax.legend(loc="upper right", fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # ----- [0,1] Mean Val Dice -----
    ax = axes[0, 1]
    for model in models:
        ep, mu, sd = aggregate_folds(all_data[model], ["val", "mean"])
        plot_curve(ax, ep, mu, sd, model,
                   MODEL_COLORS.get(model, "gray"))

        # SWA marker
        if not args.no_swa_marker:
            swa_epochs = [d["swa"] for d in all_data[model]
                          if d and d["swa"]]
            if swa_epochs:
                swa_ep = int(np.mean(swa_epochs))
                ax.axvline(swa_ep, color=MODEL_COLORS.get(model, "gray"),
                           linestyle='--', alpha=0.5, linewidth=1)
                ax.text(swa_ep, 0.02, f' SWA',
                        transform=ax.get_xaxis_transform(),
                        color=MODEL_COLORS.get(model, "gray"),
                        fontsize=8, rotation=90, va='bottom')

        # Best marker
        bests = [d["best"] for d in all_data[model] if d and d["best"]]
        if bests:
            best_ep = int(np.mean([b[0] for b in bests]))
            best_d = np.mean([b[1] for b in bests])
            ax.scatter([best_ep], [best_d], s=80, marker='*',
                       color=MODEL_COLORS.get(model, "gray"),
                       edgecolor='black', linewidth=0.8, zorder=10)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Mean Dice", fontsize=11)
    ax.set_title("(b) Validation Mean Dice", fontsize=12, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3)

    # ----- [1,0] Per-region Dice -----
    ax = axes[1, 0]
    region_legend = []
    for i, model in enumerate(models):
        for region in ["wt", "tc", "et"]:
            ep, mu, sd = aggregate_folds(all_data[model], ["val", region])
            if len(ep) == 0:
                continue
            color = REGION_COLORS[region]
            ls = ['-', '--', ':'][i % 3]
            ax.plot(ep, mu, color=color, linestyle=ls, linewidth=1.8,
                    label=f"{model}-{region.upper()}")
            if (sd > 0).any():
                ax.fill_between(ep, mu - sd, mu + sd, color=color, alpha=0.12)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation Dice", fontsize=11)
    ax.set_title("(c) Per-region Validation Dice", fontsize=12, fontweight='bold')
    ax.legend(loc="lower right", fontsize=8, ncol=2, frameon=True)
    ax.grid(True, alpha=0.3)

    # ----- [1,1] Learning Rate -----
    ax = axes[1, 1]
    for model in models:
        ep, mu, sd = aggregate_folds(all_data[model], ["train", "lr"])
        if len(ep) == 0:
            continue
        ax.plot(ep, mu, color=MODEL_COLORS.get(model, "gray"),
                linewidth=2, label=model)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Learning Rate", fontsize=11)
    ax.set_title("(d) Learning Rate Schedule", fontsize=12, fontweight='bold')
    ax.legend(loc="upper right", fontsize=10, frameon=True)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # ===== 全局标题 =====
    n_folds_total = sum(1 for m in models for d in all_data[m] if d is not None)
    fig.suptitle(
        f"Training Curves  ({n_folds_total} runs across {len(models)} model(s))",
        fontsize=13, fontweight='bold', y=0.995
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

    # ===== 输出摘要 =====
    print("\n=== 训练状态摘要 ===")
    for model in models:
        for fold, d in enumerate(all_data[model]):
            if d is None:
                continue
            n_train = len(d["train"])
            n_val = len(d["val"])
            best_str = f"best=Dice {d['best'][1]:.4f}@ep{d['best'][0]}" \
                       if d["best"] else "no best yet"
            print(f"  {model} fold {fold}: train {n_train}ep, val {n_val}ep, {best_str}")


if __name__ == "__main__":
    main()
