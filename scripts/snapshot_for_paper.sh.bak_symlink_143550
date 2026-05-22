#!/usr/bin/env bash
# snapshot_for_paper.sh
# 训练完成后立即执行：抢救 /dev/shm 数据 + 论文素材打包
#
# 用法:
#   bash scripts/snapshot_for_paper.sh                  # 默认: ./paper_data
#   bash scripts/snapshot_for_paper.sh ./my_snapshot    # 自定义路径
#   N_VIZ_CASES=50 bash scripts/snapshot_for_paper.sh   # 多备份一些 case
#
# 环境变量:
#   PROBS_DIR (默认 /dev/shm/brats_probs)
#   WORK_DIR  (默认 ./work_dir)
#   LOG_DIR   (默认 ./logs)
#   N_VIZ_CASES (默认 30，论文可视化保留的 case 数)

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SNAPSHOT_DIR="${1:-./paper_data}"
PROBS_DIR="${PROBS_DIR:-/dev/shm/brats_probs}"
WORK_DIR="${WORK_DIR:-./work_dir}"
LOG_DIR="${LOG_DIR:-./logs}"
N_VIZ_CASES="${N_VIZ_CASES:-30}"

# 颜色
R=$'\033[31m'; G=$'\033[32m'; Y=$'\033[33m'
C=$'\033[36m'; BOLD=$'\033[1m'; RST=$'\033[0m'

log() { echo "${C}[$(date +%H:%M:%S)]${RST} $*"; }
ok()  { echo "       ${G}✓${RST} $*"; }
skip() { echo "       ${Y}~${RST} $*（跳过：源不存在）"; }
fail() { echo "       ${R}✗${RST} $*"; }

echo
echo "${BOLD}${C}════════ 论文数据快照 ════════${RST}"
echo "  快照目录: ${BOLD}$SNAPSHOT_DIR${RST}"
echo "  概率源:   $PROBS_DIR"
echo "  工作目录: $WORK_DIR"
echo "  保留 case: $N_VIZ_CASES"
echo

mkdir -p "$SNAPSHOT_DIR"/{models,curves,logs,csv,probs,ensemble,figures}

# 导出给 Python 用
export SNAPSHOT_DIR PROBS_DIR WORK_DIR LOG_DIR N_VIZ_CASES

# ============================================================
# [1/7] 备份模型权重
# ============================================================
log "${BOLD}[1/7]${RST} 备份模型权重 (best.pt)..."
count=0
total_size=0
for ckpt in "$WORK_DIR"/*/fold_*/best.pt; do
    [[ -f "$ckpt" ]] || continue
    name=$(echo "$ckpt" | sed "s|$WORK_DIR/||; s|/best.pt||; s|/|_|g")
    cp "$ckpt" "$SNAPSHOT_DIR/models/${name}_best.pt"
    sz=$(stat -c%s "$ckpt")
    total_size=$((total_size + sz))
    count=$((count+1))
done
if [[ $count -gt 0 ]]; then
    ok "$count 个 best.pt ($(numfmt --to=iec $total_size))"
else
    skip "best.pt"
fi

# ============================================================
# [2/7] 备份 TensorBoard 日志
# ============================================================
log "${BOLD}[2/7]${RST} 备份 TensorBoard 日志..."
count=0
for tb_dir in "$WORK_DIR"/*/fold_*/tb; do
    [[ -d "$tb_dir" ]] || continue
    name=$(echo "$tb_dir" | sed "s|$WORK_DIR/||; s|/tb||; s|/|_|g")
    cp -r "$tb_dir" "$SNAPSHOT_DIR/curves/$name" 2>/dev/null
    count=$((count+1))
done
[[ $count -gt 0 ]] && ok "$count 个 TB 目录" || skip "TensorBoard 日志"

# ============================================================
# [3/7] 备份训练文本日志
# ============================================================
log "${BOLD}[3/7]${RST} 备份训练文本日志..."
n_logs=0
for f in "$LOG_DIR"/train_*.log "$LOG_DIR"/run_all.log "$LOG_DIR"/predict_*.log; do
    [[ -f "$f" ]] || continue
    cp "$f" "$SNAPSHOT_DIR/logs/" 2>/dev/null
    n_logs=$((n_logs+1))
done
[[ $n_logs -gt 0 ]] && ok "$n_logs 个日志文件" || skip "训练日志"

# ============================================================
# [4/7] 合并 per-case Dice CSV
# ============================================================
log "${BOLD}[4/7]${RST} 合并 per-case Dice CSV..."
python <<'PYEOF'
import glob, os, sys
try:
    import pandas as pd
except ImportError:
    print("       ✗ 缺 pandas: pip install pandas")
    sys.exit(0)

SNAP = os.environ["SNAPSHOT_DIR"]
PROBS = os.environ["PROBS_DIR"]

csv_files = sorted(glob.glob(f"{PROBS}/*/fold_*/per_case_dice.csv"))
if not csv_files:
    print(f"       ~ 没找到 CSV（推理还没完成？）")
    sys.exit(0)

dfs = [pd.read_csv(f) for f in csv_files]
df = pd.concat(dfs, ignore_index=True)
df.to_csv(f"{SNAP}/csv/all_per_case_dice.csv", index=False)
print(f"       ✓ 合并 {len(df)} 行 ({len(csv_files)} 个 CSV)")

# 按模型聚合
summary = df.groupby("model")[["dice_tc", "dice_wt", "dice_et", "dice_mean"]].agg(["mean", "std"]).round(4)
summary.to_csv(f"{SNAP}/csv/summary_by_model.csv")
print()
print("       === 模型性能（mean ± std）===")
for model in summary.index:
    row = summary.loc[model]
    print(f"       {model:12s} "
          f"TC={row[('dice_tc','mean')]:.4f}±{row[('dice_tc','std')]:.4f}  "
          f"WT={row[('dice_wt','mean')]:.4f}±{row[('dice_wt','std')]:.4f}  "
          f"ET={row[('dice_et','mean')]:.4f}±{row[('dice_et','std')]:.4f}  "
          f"Mean={row[('dice_mean','mean')]:.4f}±{row[('dice_mean','std')]:.4f}")
PYEOF

# ============================================================
# [5/7] 备份代表性概率图（按 Dice 分位采样）
# ============================================================
log "${BOLD}[5/7]${RST} 备份代表性概率图..."
python <<'PYEOF'
import os, glob, shutil, random, json, sys

SNAP = os.environ["SNAPSHOT_DIR"]
PROBS = os.environ["PROBS_DIR"]
N = int(os.environ["N_VIZ_CASES"])

if not os.path.isdir(PROBS):
    print(f"       ~ {PROBS} 不存在（推理还没完成？）")
    sys.exit(0)

# 优先按 Dice 分位采样
csv_path = os.path.join(SNAP, "csv/all_per_case_dice.csv")
selected_cases = []

if os.path.exists(csv_path):
    try:
        import pandas as pd
        import numpy as np
        df = pd.read_csv(csv_path)
        case_mean = df.groupby("case_id")["dice_mean"].mean().reset_index()
        case_mean = case_mean.sort_values("dice_mean").reset_index(drop=True)
        # 在 5%~95% 分位均匀采样
        quantiles = np.linspace(0.05, 0.95, N)
        idx = (quantiles * (len(case_mean) - 1)).astype(int)
        idx = np.unique(idx)
        selected_cases = case_mean.iloc[idx]["case_id"].tolist()
        print(f"       ✓ 按 Dice 分位采样 {len(selected_cases)} 个 case")
        print(f"         Dice 范围: {case_mean['dice_mean'].min():.4f} ~ "
              f"{case_mean['dice_mean'].max():.4f}")
    except Exception as e:
        print(f"       ⚠ CSV 分位采样失败 ({e})，改用随机")
        selected_cases = []

if not selected_cases:
    # 没有 CSV 或失败，随机选
    random.seed(42)
    all_npz = sorted(glob.glob(f"{PROBS}/*/fold_*/*.npz"))
    if not all_npz:
        print(f"       ~ {PROBS} 下没有 .npz 文件")
        sys.exit(0)
    all_cases = sorted(set(os.path.basename(f).replace(".npz", "") for f in all_npz))
    selected_cases = random.sample(all_cases, min(N, len(all_cases)))
    print(f"       ✓ 随机采样 {len(selected_cases)} 个 case")

# 备份这些 case 的所有概率图
copied = 0
total_bytes = 0
for case_id in selected_cases:
    for npz in glob.glob(f"{PROBS}/*/fold_*/{case_id}.npz"):
        rel = npz.replace(f"{PROBS}/", "")
        dst = os.path.join(SNAP, "probs", rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(npz, dst)
        copied += 1
        total_bytes += os.path.getsize(npz)
    # 也备份融合结果
    for npz in glob.glob(f"{PROBS}/fused/{case_id}.npz") + \
               glob.glob(f"{PROBS}/ensemble/{case_id}.npz") + \
               glob.glob(f"{PROBS}/*_oof/{case_id}.npz"):
        rel = npz.replace(f"{PROBS}/", "")
        dst = os.path.join(SNAP, "probs", rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(npz, dst)
        copied += 1
        total_bytes += os.path.getsize(npz)

print(f"       ✓ 复制 {copied} 个概率文件 ({total_bytes / 1024 / 1024:.1f} MB)")

# 保存选中 case 列表
with open(os.path.join(SNAP, "selected_cases.json"), "w") as f:
    json.dump(selected_cases, f, indent=2)
print(f"       ✓ selected_cases.json")
PYEOF

# ============================================================
# [6/7] 备份融合 + 网格搜索结果
# ============================================================
log "${BOLD}[6/7]${RST} 备份融合 + 网格搜索结果..."
n_ens=0
for f in "$WORK_DIR"/ensemble_results.json \
         "$WORK_DIR"/final_metrics.json \
         "$WORK_DIR"/grid_search_results.json \
         "$WORK_DIR"/grid_search_matrix.npz; do
    if [[ -f "$f" ]]; then
        cp "$f" "$SNAPSHOT_DIR/ensemble/"
        n_ens=$((n_ens+1))
    fi
done
[[ -d "$WORK_DIR/grid_search" ]] && cp -r "$WORK_DIR/grid_search" "$SNAPSHOT_DIR/ensemble/" && n_ens=$((n_ens+1))
[[ $n_ens -gt 0 ]] && ok "$n_ens 个融合/网格文件" || skip "融合结果"

# ============================================================
# [7/7] 生成 MANIFEST + 论文图命令脚本
# ============================================================
log "${BOLD}[7/7]${RST} 生成 MANIFEST 清单..."

# 计算每个子目录大小
size_models=$(du -sh "$SNAPSHOT_DIR/models" 2>/dev/null | awk '{print $1}')
size_curves=$(du -sh "$SNAPSHOT_DIR/curves" 2>/dev/null | awk '{print $1}')
size_logs=$(du -sh "$SNAPSHOT_DIR/logs" 2>/dev/null | awk '{print $1}')
size_csv=$(du -sh "$SNAPSHOT_DIR/csv" 2>/dev/null | awk '{print $1}')
size_probs=$(du -sh "$SNAPSHOT_DIR/probs" 2>/dev/null | awk '{print $1}')
size_ens=$(du -sh "$SNAPSHOT_DIR/ensemble" 2>/dev/null | awk '{print $1}')
size_total=$(du -sh "$SNAPSHOT_DIR" 2>/dev/null | awk '{print $1}')

cat > "$SNAPSHOT_DIR/MANIFEST.md" <<EOF
# 论文数据快照

**生成时间**: $(date '+%Y-%m-%d %H:%M:%S')
**源目录**:   $(pwd)
**总大小**:   $size_total

## 目录结构

| 子目录 | 内容 | 大小 |
|---|---|---|
| \`models/\`    | 所有 fold 的 best.pt        | $size_models |
| \`curves/\`    | TensorBoard 日志             | $size_curves |
| \`logs/\`      | 训练文本日志                 | $size_logs |
| \`csv/\`       | per-case Dice 合并表         | $size_csv |
| \`probs/\`     | 代表性 case 概率图           | $size_probs |
| \`ensemble/\`  | 融合 + 网格搜索结果          | $size_ens |
| \`figures/\`   | 论文图（生成后存这里）       | — |

## 论文图生成命令

\`\`\`bash
# F4 训练曲线
python scripts/plot_training_curves.py \\
    --logs_dir $SNAPSHOT_DIR/logs \\
    --models segresnet,swinunetr \\
    --output $SNAPSHOT_DIR/figures/F4_training_curves.png

# F5 定性分割结果（5 个代表性 case）
python scripts/viz_segmentation.py \\
    --csv $SNAPSHOT_DIR/csv/all_per_case_dice.csv \\
    --auto 5 \\
    --probs_dir $SNAPSHOT_DIR/probs \\
    --output $SNAPSHOT_DIR/figures/F5_qualitative.png

# F6 Dice 箱线图（待写脚本）
# F10 失败案例（待写脚本）
\`\`\`

## 关键发现

EOF

# 如果有 CSV，自动追加摘要
if [[ -f "$SNAPSHOT_DIR/csv/all_per_case_dice.csv" ]]; then
    python <<'PYAPP' >> "$SNAPSHOT_DIR/MANIFEST.md"
import os, pandas as pd
SNAP = os.environ["SNAPSHOT_DIR"]
df = pd.read_csv(f"{SNAP}/csv/all_per_case_dice.csv")
print(f"- 总 case 数: **{df['case_id'].nunique()}**")
print(f"- 总评估次数: **{len(df)}**")
print(f"- 模型: **{', '.join(df['model'].unique())}**")
print()
print("### 模型性能（mean Dice ± std）")
print()
print("| Model | TC | WT | ET | Mean |")
print("|---|---|---|---|---|")
for m, g in df.groupby("model"):
    print(f"| {m} "
          f"| {g['dice_tc'].mean():.4f}±{g['dice_tc'].std():.4f} "
          f"| {g['dice_wt'].mean():.4f}±{g['dice_wt'].std():.4f} "
          f"| {g['dice_et'].mean():.4f}±{g['dice_et'].std():.4f} "
          f"| **{g['dice_mean'].mean():.4f}±{g['dice_mean'].std():.4f}** |")
PYAPP
else
    echo "（推理未完成，无 CSV 摘要）" >> "$SNAPSHOT_DIR/MANIFEST.md"
fi

ok "MANIFEST.md 已生成"

# ============================================================
# 总结
# ============================================================
echo
echo "${BOLD}${G}════════ 快照完成 ════════${RST}"
echo
du -sh "$SNAPSHOT_DIR"/* 2>/dev/null | sort -k2
echo
echo "${BOLD}总大小:${RST} $size_total"
echo
echo "${BOLD}${C}下一步:${RST}"
echo "  ${BOLD}1.${RST} 查看清单:    cat $SNAPSHOT_DIR/MANIFEST.md"
echo "  ${BOLD}2.${RST} 生成论文图:   见 MANIFEST.md 里的命令"
echo "  ${BOLD}3.${RST} 下载到本地:   scp -r $SNAPSHOT_DIR/ user@local:~/"
echo
