#!/usr/bin/env python3
"""F9 grid_search 曲线 (Phase 1 alpha sweep + Phase 2 threshold sweep)."""
import argparse, os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 论文版: 禁用 fill_between
import matplotlib.axes as _mpl_axes
_mpl_axes.Axes.fill_between = lambda *a, **k: None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_dir", default="work_dir/grid_search")
    ap.add_argument("--output", default="paper_figs/F9_grid_search.png")
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    p1 = pd.read_csv(Path(args.grid_dir) / "grid_phase1.csv")
    p2 = pd.read_csv(Path(args.grid_dir) / "grid_phase2.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Phase 1: alpha sweep
    ax = axes[0]
    for r, c in [("dice_tc","#2E86AB"), ("dice_wt","#06A77D"),
                 ("dice_et","#E63946"), ("dice_mean","black")]:
        lw = 2.5 if r == "dice_mean" else 1.5
        ls = "-" if r == "dice_mean" else "--"
        label = r.replace("dice_", "").upper()
        ax.plot(p1["alpha"], p1[r], marker="o", lw=lw, ls=ls,
                color=c, label=label)
    best_idx = p1["dice_mean"].idxmax()
    ax.axvline(p1.loc[best_idx, "alpha"], color="red", ls=":", alpha=0.6,
               label=f"Best α={p1.loc[best_idx,'alpha']:.2f}")
    ax.set_xlabel("α (weight of SegResNet)", fontsize=12)
    ax.set_ylabel("Dice", fontsize=12)
    ax.set_title("(a) Phase 1: Ensemble Weight Sweep", fontsize=13, fontweight='bold')
    ax.legend(loc="lower center", ncol=4, fontsize=10)
    ax.grid(alpha=0.3)

    # Phase 2: threshold sweep
    ax = axes[1]
    for r, c in [("tc","#2E86AB"), ("wt","#06A77D"), ("et","#E63946")]:
        sub = p2[p2["region"]==r]
        ax.plot(sub["threshold"], sub["dice"], marker="s", lw=2,
                color=c, label=r.upper())
        best = sub.loc[sub["dice"].idxmax()]
        ax.axvline(best["threshold"], color=c, ls=":", alpha=0.4)
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Dice", fontsize=12)
    ax.set_title("(b) Phase 2: Per-region Threshold Sweep", fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.savefig(args.output.replace(".png",".pdf"), bbox_inches="tight")
    print(f"✓ saved: {args.output}")

if __name__ == "__main__":
    main()
