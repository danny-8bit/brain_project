#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STATE_DIR="${STATE_DIR:-./run_state}"
LOG_DIR="${LOG_DIR:-./logs}"

C_RED=$'\033[31m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'
C_BLUE=$'\033[34m'; C_BOLD=$'\033[1m'; C_RST=$'\033[0m'

format_time() {
    local sec=$1
    if   [[ $sec -lt 60 ]];   then echo "${sec}s"
    elif [[ $sec -lt 3600 ]]; then printf "%dm%02ds" $((sec/60)) $((sec%60))
    else printf "%dh%02dm" $((sec/3600)) $((sec%3600/60))
    fi
}

echo "${C_BOLD}================ BraTS Pipeline 状态 ================${C_RST}"
echo ""

if [[ ! -d "$STATE_DIR" ]] || [[ -z "$(ls -A "$STATE_DIR" 2>/dev/null)" ]]; then
    echo "(尚未启动任何任务)"
    exit 0
fi

printf "${C_BOLD}%-40s %-10s %s${C_RST}\n" "任务" "状态" "耗时"
echo "─────────────────────────────────────────────────────"

n_done=0; n_run=0; n_fail=0; total=0

for f in "$STATE_DIR"/*.state; do
    [[ -f "$f" ]] || continue
    name=$(basename "$f" .state)
    IFS='|' read -r status t1 t2 < "$f"

    case "$status" in
        done)
            color="$C_GREEN"; sym="✓"; n_done=$((n_done+1))
            total=$((total + t1))
            time_str=$(format_time $t1) ;;
        running)
            color="$C_YELLOW"; sym="▶"; n_run=$((n_run+1))
            elapsed=$(($(date +%s) - t1))
            time_str="$(format_time $elapsed) (运行中)" ;;
        failed)
            color="$C_RED"; sym="✗"; n_fail=$((n_fail+1))
            time_str="$(format_time $t1) [exit=$t2]" ;;
        *) color="$C_RST"; sym="?"; time_str="$status" ;;
    esac

    printf "${color}%s %-38s %-10s${C_RST} %s\n" "$sym" "$name" "$status" "$time_str"
done

echo "─────────────────────────────────────────────────────"
printf "${C_GREEN}完成: %d${C_RST}  ${C_YELLOW}运行中: %d${C_RST}  ${C_RED}失败: %d${C_RST}  累计训练: %s\n" \
    "$n_done" "$n_run" "$n_fail" "$(format_time $total)"

if [[ $n_fail -gt 0 ]]; then
    echo ""
    echo "${C_RED}失败任务详情:${C_RST}"
    for f in "$STATE_DIR"/*.state; do
        IFS='|' read -r status t1 t2 < "$f"
        if [[ "$status" == "failed" ]]; then
            name=$(basename "$f" .state)
            echo "  - $name (日志: $LOG_DIR/$name.log)"
            echo "    最后 3 行错误:"
            tail -n 3 "$LOG_DIR/$name.log" 2>/dev/null | sed 's/^/      /'
        fi
    done
fi