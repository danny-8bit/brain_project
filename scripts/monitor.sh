#!/usr/bin/env bash
# 系统监控：GPU + 内存 + 磁盘（不含训练细节）
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STATE_DIR="${STATE_DIR:-./run_state}"
LOG_DIR="${LOG_DIR:-./logs}"
REFRESH="${1:-5}"

R=$'\033[31m'; G=$'\033[32m'; Y=$'\033[33m'
B=$'\033[34m'; C=$'\033[36m'
BOLD=$'\033[1m'; DIM=$'\033[2m'; RST=$'\033[0m'

trap 'clear; exit 0' INT TERM

fmt_time() {
    local s=$1
    if   [[ $s -lt 60 ]];   then echo "${s}s"
    elif [[ $s -lt 3600 ]]; then printf "%dm%02ds" $((s/60)) $((s%60))
    else printf "%dh%02dm" $((s/3600)) $((s%3600/60))
    fi
}

bar() {
    # 进度条 (用 / 共 20 格)
    local pct=$1
    local width=20
    local filled=$(( pct * width / 100 ))
    local empty=$(( width - filled ))
    local color=$G
    [[ $pct -gt 60 ]] && color=$Y
    [[ $pct -gt 85 ]] && color=$R
    printf "${color}"
    printf '█%.0s' $(seq 1 $filled) 2>/dev/null
    printf "${DIM}"
    printf '░%.0s' $(seq 1 $empty) 2>/dev/null
    printf "${RST}"
}

while true; do
    clear

    # ===== 标题 =====
    echo "${BOLD}${C}━━ 系统监控  $(date '+%H:%M:%S')  刷新${REFRESH}s ━━${RST}"
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        echo "${G}● ${RST}venv: $(basename $VIRTUAL_ENV) | py: $(python --version 2>&1 | awk '{print $2}')"
    fi
    echo ""

    # ===== GPU =====
    echo "${BOLD}GPU${RST}"
    gpu_info=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv,noheader,nounits 2>/dev/null | head -n 1)
    if [[ -n "$gpu_info" ]]; then
        IFS=',' read -r util mem_u mem_t temp pwr <<< "$gpu_info"
        util=$(echo "$util" | xargs); mem_u=$(echo "$mem_u" | xargs)
        mem_t=$(echo "$mem_t" | xargs); temp=$(echo "$temp" | xargs); pwr=$(echo "$pwr" | xargs)

        mem_pct=$(awk "BEGIN{printf \"%.0f\", $mem_u*100/$mem_t}")

        printf "  利用率  $(bar $util) ${G}%3s%%${RST}\n" "$util"
        printf "  显存    $(bar $mem_pct) %s/%sMB\n" "$mem_u" "$mem_t"
        printf "  ${DIM}温度 %s°C  功耗 %.0fW${RST}\n" "$temp" "$pwr"
    fi
    echo ""

    # ===== 内存（容器）=====
    echo "${BOLD}内存（容器）${RST}"
    if [[ -f /sys/fs/cgroup/memory/memory.usage_in_bytes ]]; then
        mused=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes)
        mlimit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
        mused_gb=$(awk "BEGIN{printf \"%.0f\", $mused/1073741824}")
        mlimit_gb=$(awk "BEGIN{printf \"%.0f\", $mlimit/1073741824}")
        mpct=$(awk "BEGIN{printf \"%.0f\", $mused*100/$mlimit}")
        printf "  $(bar $mpct) %sG/%sG (%s%%)\n" "$mused_gb" "$mlimit_gb" "$mpct"
    else
        free -h | awk 'NR==2{printf "  %s / %s (可用 %s)\n", $3, $2, $7}'
    fi
    echo ""

    # ===== 磁盘 =====
    echo "${BOLD}存储${RST}"
    df_data=$(df /root/autodl-tmp 2>/dev/null | awk 'NR==2{printf "%d %s", $5+0, $5}')
    df_used=$(df -h /root/autodl-tmp 2>/dev/null | awk 'NR==2{printf "%s/%s", $3, $2}')
    if [[ -n "$df_data" ]]; then
        pct=$(echo $df_data | cut -d' ' -f1)
        printf "  数据盘   $(bar $pct) %s\n" "$df_used"
    fi

    shm_pct=$(df /dev/shm 2>/dev/null | awk 'NR==2{print $5+0}')
    shm_used=$(df -h /dev/shm 2>/dev/null | awk 'NR==2{printf "%s/%s", $3, $2}')
    [[ -n "$shm_pct" ]] && printf "  /dev/shm $(bar $shm_pct) %s\n" "$shm_used"
    echo ""

    # ===== 当前任务（简洁）=====
    echo "${BOLD}当前任务${RST}"
    n_done=0; n_run=0; n_fail=0; running_name=""; running_elapsed=0
    if [[ -d "$STATE_DIR" ]]; then
        for f in "$STATE_DIR"/*.state; do
            [[ -f "$f" ]] || continue
            IFS='|' read -r ss t0 _ < "$f"
            case "$ss" in
                done) n_done=$((n_done+1)) ;;
                failed) n_fail=$((n_fail+1)) ;;
                running)
                    n_run=$((n_run+1))
                    running_name=$(basename "$f" .state)
                    running_elapsed=$(( $(date +%s) - t0 ))
                    ;;
            esac
        done
    fi
    printf "  ${G}完成${RST} %d   ${Y}运行中${RST} %d   ${R}失败${RST} %d\n" "$n_done" "$n_run" "$n_fail"
    if [[ -n "$running_name" ]]; then
        short=${running_name#train_}; short=${short#predict_}
        printf "  ${Y}▶${RST} %s  ${DIM}%s${RST}\n" "$short" "$(fmt_time $running_elapsed)"
    fi

    sleep "$REFRESH"
done
