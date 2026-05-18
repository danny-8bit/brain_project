#!/usr/bin/env bash
# 简洁状态面板，给 watch --color 用
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STATE_DIR="${STATE_DIR:-./run_state}"
LOG_DIR="${LOG_DIR:-./logs}"

R=$'\033[31m'; G=$'\033[32m'; Y=$'\033[33m'
C=$'\033[36m'; BOLD=$'\033[1m'; DIM=$'\033[2m'; RST=$'\033[0m'

fmt_time() {
    local s=$1
    if   [[ $s -lt 60 ]];   then echo "${s}s"
    elif [[ $s -lt 3600 ]]; then printf "%dm%02ds" $((s/60)) $((s%60))
    else printf "%dh%02dm" $((s/3600)) $((s%3600/60))
    fi
}

# === 任务统计 ===
n_done=0; n_run=0; n_fail=0; total_t=0
declare -a running_list=()
declare -a failed_list=()
declare -a done_list=()

if [[ -d "$STATE_DIR" ]]; then
    for f in "$STATE_DIR"/*.state; do
        [[ -f "$f" ]] || continue
        IFS='|' read -r ss t0 rc < "$f"
        name=$(basename "$f" .state)
        case "$ss" in
            done)
                n_done=$((n_done+1))
                total_t=$((total_t + t0))
                done_list+=("$name|$t0")
                ;;
            running)
                n_run=$((n_run+1))
                elapsed=$(( $(date +%s) - t0 ))
                running_list+=("$name|$elapsed")
                ;;
            failed)
                n_fail=$((n_fail+1))
                failed_list+=("$name|${rc:-?}")
                ;;
        esac
    done
fi

# === 头部 ===
echo "${BOLD}${C}━━━ BraTS Pipeline 状态 ━━━━━━━━━━━━━━━━━━━━━━━${RST}"
printf "${G}✓ 完成 %d${RST}   ${Y}▶ 运行 %d${RST}   ${R}✗ 失败 %d${RST}\n" \
    "$n_done" "$n_run" "$n_fail"
echo ""

# === 运行中（最重要）===
if [[ ${#running_list[@]} -gt 0 ]]; then
    echo "${BOLD}运行中:${RST}"
    for item in "${running_list[@]}"; do
        IFS='|' read -r name elapsed <<< "$item"
        short=${name#train_}; short=${short#predict_}

        # 尝试从日志读取 epoch 信息
        log_file="$LOG_DIR/${name}.log"
        progress=""
        if [[ -f "$log_file" ]]; then
            ep=$(grep -oE "Epoch [0-9]+/[0-9]+" "$log_file" 2>/dev/null | tail -1)
            if [[ -n "$ep" ]]; then
                progress=" ${DIM}|${RST} $ep"
            else
                pct=$(grep -oE "Loading dataset: *[0-9]+%" "$log_file" 2>/dev/null | tail -1 | grep -oE "[0-9]+%")
                [[ -n "$pct" ]] && progress=" ${DIM}|${RST} 加载 $pct"
            fi
        fi

        printf "  ${Y}▶${RST} %-22s ${DIM}%s${RST}%s\n" \
            "$short" "$(fmt_time $elapsed)" "$progress"
    done
    echo ""
fi

# === 失败 ===
if [[ ${#failed_list[@]} -gt 0 ]]; then
    echo "${BOLD}${R}失败任务:${RST}"
    for item in "${failed_list[@]}"; do
        IFS='|' read -r name rc <<< "$item"
        short=${name#train_}; short=${short#predict_}
        printf "  ${R}✗${RST} %-22s ${DIM}exit=%s${RST}\n" "$short" "$rc"
    done
    echo ""
fi

# === 完成（折叠显示）===
if [[ ${#done_list[@]} -gt 0 ]]; then
    echo "${BOLD}${G}已完成:${RST}"
    for item in "${done_list[@]}"; do
        IFS='|' read -r name t <<< "$item"
        short=${name#train_}; short=${short#predict_}
        printf "  ${G}✓${RST} %-22s ${DIM}%s${RST}\n" "$short" "$(fmt_time $t)"
    done
    echo ""
fi

# === 累计时间 ===
[[ $total_t -gt 0 ]] && echo "${DIM}累计训练: $(fmt_time $total_t)${RST}"
