#!/usr/bin/env python3
"""
把现有 result/*.csv 整合成 paper_data/csv/all_per_case_dice.csv 统一格式.

期望输出列:
  case_id, model, fold, dice_tc, dice_wt, dice_et, dice_mean,
  hd95_tc, hd95_wt, hd95_et, hd95_mean
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", default="result")
    ap.add_argument("--out_csv", default="paper_data/csv/all_per_case_dice.csv")
    ap.add_argument("--ensemble_csv", default="result/final_ensemble_1251.csv")
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    all_dfs = []

    # 1. 单 fold 评估 CSV: eval_{model}_fold{N}_dice_hd95.csv
    pattern = re.compile(r"eval_(\w+)_fold(\d+)_dice_hd95\.csv")
    for f in sorted(result_dir.glob("eval_*_fold*_dice_hd95.csv")):
        m = pattern.match(f.name)
        if not m: continue
        model_name, fold = m.group(1), int(m.group(2))
        df = pd.read_csv(f)
        df["model"] = model_name
        df["fold"] = fold
        df["config"] = f"{model_name}_fold{fold}"
        df = df.rename(columns={"case": "case_id"})
        all_dfs.append(df)
        print(f"  [+] {f.name}: {len(df)} cases, model={model_name}, fold={fold}")

    # 2. Ensemble (1251 cases)
    ens = Path(args.ensemble_csv)
    if ens.exists():
        df = pd.read_csv(ens)
        df["model"] = "ensemble"
        df["fold"] = -1
        df["config"] = "ensemble"
        df = df.rename(columns={"case": "case_id"})
        all_dfs.append(df)
        print(f"  [+] {ens.name}: {len(df)} cases, model=ensemble")

    if not all_dfs:
        print("✗ 没找到任何 CSV!")
        sys.exit(1)

    out = pd.concat(all_dfs, ignore_index=True)
    cols = ["case_id", "model", "fold", "config",
            "dice_tc", "dice_wt", "dice_et", "dice_mean",
            "hd95_tc", "hd95_wt", "hd95_et", "hd95_mean"]
    out = out[cols]

    out.to_csv(out_csv, index=False)
    print()
    print(f"✓ 保存 {len(out)} 行 → {out_csv}")
    print()
    print("各模型 Dice 概览:")
    summary = out.groupby("model")[["dice_tc","dice_wt","dice_et","dice_mean"]].agg(["mean","std"]).round(4)
    print(summary)
    print()
    print(f"各模型 HD95 概览:")
    summary = out.groupby("model")[["hd95_tc","hd95_wt","hd95_et","hd95_mean"]].agg(["mean","std"]).round(2)
    print(summary)


if __name__ == "__main__":
    main()
