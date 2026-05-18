#!/usr/bin/env python3
"""
F10 论文图：失败案例分析

从 per_case_dice.csv 中找出：
  1. Dice 最低的 N 个 case（失败案例）
  2. Dice 最高的 N 个 case（成功案例）
  3. 模型间方差最大的 N 个 case（分歧案例）

输出：
  - failure_report.md      失败案例分析报告
  - failure_cases.txt      失败案例 ID 列表（供 viz_segmentation.py 用）
  - 自动调用 viz_segmentation 生成 F10 图

用法:
  python scripts/find_failure_cases.py \
      --csv paper_data/csv/all_per_case_dice.csv \
      --n 5 --output_dir paper_figs/

  # 自动生成可视化
  python scripts/find_failure_cases.py --auto_viz
"""

import argparse
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def categorize_failure(row, all_df):
    """
    根据 dice 模式分类失败原因
    返回 (类别, 解释)
    """
    et = row.get("dice_et", 0)
    tc = row.get("dice_tc", 0)
    wt = row.get("dice_wt", 0)

    if et == 0 and tc > 0.7:
        return ("ET_missing", "ET 完全漏检（可能为小肿瘤或无增强）")
    if et < 0.3 and tc > 0.6:
        return ("ET_low", "ET 严重低估")
    if wt < 0.6:
        return ("WT_poor", "整体分割失败（可能解剖异常）")
    if tc < 0.5 and wt > 0.8:
        return ("TC_poor", "TC 难分（可能小核心区）")
    if all(v < 0.7 for v in [et, tc, wt]):
        return ("global_low", "全区域都差")
    return ("mixed", "混合误差")


def summarize_case(df_case: pd.DataFrame) -> dict:
    """对一个 case 跨模型/fold 汇总"""
    return {
        "case_id": df_case["case_id"].iloc[0],
        "n_models": df_case["model"].nunique(),
        "n_folds": df_case["fold"].nunique() if "fold" in df_case else 1,
        "dice_tc_mean": df_case["dice_tc"].mean(),
        "dice_wt_mean": df_case["dice_wt"].mean(),
        "dice_et_mean": df_case["dice_et"].mean(),
        "dice_mean":    df_case["dice_mean"].mean(),
        "dice_mean_std": df_case["dice_mean"].std() if len(df_case) > 1 else 0,
        # 详细分模型
        "by_model": {
            m: {
                "wt": g["dice_wt"].mean(),
                "tc": g["dice_tc"].mean(),
                "et": g["dice_et"].mean(),
                "mean": g["dice_mean"].mean(),
            }
            for m, g in df_case.groupby("model")
        }
    }


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    parser.add_argument("--csv", default="paper_data/csv/all_per_case_dice.csv")
    parser.add_argument("--n", type=int, default=5,
                        help="每类挑选 N 个 case")
    parser.add_argument("--output_dir", default="paper_figs",
                        help="输出目录")
    parser.add_argument("--filter", default=None,
                        help="过滤条件，如 'use_ema=True'")
    parser.add_argument("--model", default=None,
                        help="只看某个模型；不指定则汇总所有")
    parser.add_argument("--auto_viz", action="store_true",
                        help="自动调用 viz_segmentation.py 生成可视化")
    parser.add_argument("--data_dir", default="./data/BraTS2021_TrainingData",
                        help="原始数据目录（auto_viz 用）")
    parser.add_argument("--probs_dir", default="paper_data/probs",
                        help="概率图目录（auto_viz 用）")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"✗ CSV 不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"✓ 加载 CSV: {len(df)} 行, {df['case_id'].nunique()} 个 case")

    # 过滤
    if args.filter:
        for kv in args.filter.split(","):
            k, v = kv.split("=")
            k, v = k.strip(), v.strip()
            if v.lower() == "true":   v = True
            elif v.lower() == "false": v = False
            elif v.lstrip("-").isdigit(): v = int(v)
            else:
                try: v = float(v)
                except: pass
            if k in df.columns:
                df = df[df[k] == v]
                print(f"  过滤 {k}={v}: → {len(df)} 行")

    if args.model:
        df = df[df["model"] == args.model]
        print(f"  仅模型 {args.model}: → {len(df)} 行")

    if len(df) == 0:
        sys.exit("✗ 过滤后无数据")

    # ===== 按 case 聚合 =====
    summaries = []
    for case_id, g in df.groupby("case_id"):
        summaries.append(summarize_case(g))
    sum_df = pd.DataFrame(summaries)

    # ===== 找出各类 case =====
    sum_df_sorted_low = sum_df.nsmallest(args.n, "dice_mean")
    sum_df_sorted_high = sum_df.nlargest(args.n, "dice_mean")
    sum_df_sorted_var = sum_df.nlargest(args.n, "dice_mean_std")

    # 失败原因分类
    sum_df["failure_category"] = sum_df.apply(
        lambda r: categorize_failure(
            {"dice_et": r["dice_et_mean"],
             "dice_tc": r["dice_tc_mean"],
             "dice_wt": r["dice_wt_mean"]}, df)[0],
        axis=1
    )

    # ===== 输出报告 =====
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "failure_report.md"

    lines = []
    lines.append("# 失败案例分析报告\n")
    lines.append(f"**生成时间**: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}\n")
    lines.append(f"**数据源**: `{csv_path}`\n")
    lines.append(f"**总 case 数**: {len(sum_df)}\n")
    lines.append(f"**模型**: {sorted(df['model'].unique())}\n\n")

    # 整体分布
    lines.append("## 整体 Dice 分布\n")
    lines.append("| 分位 | Mean Dice |")
    lines.append("|---|---|")
    for q in [0.05, 0.25, 0.50, 0.75, 0.95]:
        lines.append(f"| {int(q*100)}% | {sum_df['dice_mean'].quantile(q):.4f} |")
    lines.append("")

    # 失败原因分布
    lines.append("## 失败原因分布（按 case 聚合后）\n")
    cat_counts = sum_df["failure_category"].value_counts()
    lines.append("| 类别 | 数量 | 占比 |")
    lines.append("|---|---:|---:|")
    for cat, n in cat_counts.items():
        lines.append(f"| {cat} | {n} | {n/len(sum_df)*100:.1f}% |")
    lines.append("")

    # === Top N 失败案例 ===
    lines.append(f"## 🔴 最差的 {args.n} 个 case（建议放论文 Discussion）\n")
    lines.append("| Case ID | WT | TC | ET | Mean | 原因 |")
    lines.append("|---|---:|---:|---:|---:|---|")
    failure_ids = []
    for _, row in sum_df_sorted_low.iterrows():
        cat, expl = categorize_failure({
            "dice_et": row["dice_et_mean"],
            "dice_tc": row["dice_tc_mean"],
            "dice_wt": row["dice_wt_mean"]
        }, df)
        lines.append(f"| `{row['case_id']}` "
                     f"| {row['dice_wt_mean']:.3f} "
                     f"| {row['dice_tc_mean']:.3f} "
                     f"| {row['dice_et_mean']:.3f} "
                     f"| **{row['dice_mean']:.3f}** "
                     f"| {expl} |")
        failure_ids.append(row["case_id"])
    lines.append("")

    # === Top N 成功案例 ===
    lines.append(f"## 🟢 最好的 {args.n} 个 case（对比展示）\n")
    lines.append("| Case ID | WT | TC | ET | Mean |")
    lines.append("|---|---:|---:|---:|---:|")
    success_ids = []
    for _, row in sum_df_sorted_high.iterrows():
        lines.append(f"| `{row['case_id']}` "
                     f"| {row['dice_wt_mean']:.3f} "
                     f"| {row['dice_tc_mean']:.3f} "
                     f"| {row['dice_et_mean']:.3f} "
                     f"| **{row['dice_mean']:.3f}** |")
        success_ids.append(row["case_id"])
    lines.append("")

    # === 模型分歧大的 case ===
    lines.append(f"## 🟡 模型分歧最大的 {args.n} 个 case\n")
    lines.append("| Case ID | Mean Dice | Std | 各模型 Mean |")
    lines.append("|---|---:|---:|---|")
    disagree_ids = []
    for _, row in sum_df_sorted_var.iterrows():
        by_m = row["by_model"]
        m_strs = ", ".join(f"{m}={info['mean']:.3f}"
                           for m, info in by_m.items())
        lines.append(f"| `{row['case_id']}` "
                     f"| {row['dice_mean']:.3f} "
                     f"| {row['dice_mean_std']:.3f} "
                     f"| {m_strs} |")
        disagree_ids.append(row["case_id"])
    lines.append("")

    # === 保存 case ID 列表给 viz 用 ===
    (out_dir / "failure_cases.txt").write_text("\n".join(failure_ids))
    (out_dir / "success_cases.txt").write_text("\n".join(success_ids))
    (out_dir / "disagree_cases.txt").write_text("\n".join(disagree_ids))

    # === Discussion 模板 ===
    lines.append("## 论文 Discussion 段落模板\n")
    lines.append("```text")
    lines.append(f"To investigate the limitations of our method, we analyzed the {args.n} ")
    lines.append(f"worst-performing cases (Fig. F10). The most common failure modes were:")
    for cat, n in cat_counts.head(3).items():
        lines.append(f"  - {cat}: {n} cases ({n/len(sum_df)*100:.1f}%)")
    lines.append(f"")
    lines.append(f"Notably, ET segmentation showed the highest variability "
                 f"(see Fig. F6), particularly in cases with small or absent enhancement, ")
    lines.append(f"consistent with prior literature on BraTS challenges.")
    lines.append("```")
    lines.append("")

    # === 写入文件 ===
    report_path.write_text("\n".join(lines))
    print(f"\n✓ 报告: {report_path}")
    print(f"✓ 失败 case 列表: {out_dir/'failure_cases.txt'} ({len(failure_ids)})")
    print(f"✓ 成功 case 列表: {out_dir/'success_cases.txt'} ({len(success_ids)})")
    print(f"✓ 分歧 case 列表: {out_dir/'disagree_cases.txt'} ({len(disagree_ids)})")

    # === 终端摘要 ===
    print(f"\n{'='*60}")
    print("失败案例 Top 3:")
    for _, row in sum_df_sorted_low.head(3).iterrows():
        print(f"  {row['case_id']}: Mean Dice = {row['dice_mean']:.4f}")
    print(f"\n成功案例 Top 3:")
    for _, row in sum_df_sorted_high.head(3).iterrows():
        print(f"  {row['case_id']}: Mean Dice = {row['dice_mean']:.4f}")

    # ===== 自动调 viz_segmentation =====
    if args.auto_viz:
        print(f"\n=== 自动生成可视化 ===")
        viz_script = Path(__file__).parent / "viz_segmentation.py"

        # 失败案例
        cmd1 = [
            "python", str(viz_script),
            "--case_ids", ",".join(failure_ids),
            "--data_dir", args.data_dir,
            "--probs_dir", args.probs_dir,
            "--output", str(out_dir / "F10_failure_cases.png"),
        ]
        print(f"运行: {' '.join(cmd1)}")
        subprocess.run(cmd1, check=False)

        # 成功对比
        cmd2 = [
            "python", str(viz_script),
            "--case_ids", ",".join(success_ids),
            "--data_dir", args.data_dir,
            "--probs_dir", args.probs_dir,
            "--output", str(out_dir / "F10_success_cases.png"),
        ]
        print(f"运行: {' '.join(cmd2)}")
        subprocess.run(cmd2, check=False)


if __name__ == "__main__":
    main()
