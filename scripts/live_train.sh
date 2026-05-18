#!/usr/bin/env bash
# 实时显示训练关键事件（过滤 tqdm 刷屏）
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STATE_DIR="${STATE_DIR:-./run_state}"
LOG_DIR="${LOG_DIR:-./logs}"

R=$'\033[31m'; G=$'\033[32m'; Y=$'\033[33m'
C=$'\033[36m'; M=$'\033[35m'
BOLD=$'\033[1m'; DIM=$'\033[2m'; RST=$'\033[0m'

trap 'echo; exit 0' INT TERM

find_current() {
    for f in "$STATE_DIR"/*.state; do
        [[ -f "$f" ]] || continue
        ss=$(cut -d'|' -f1 "$f")
        name=$(basename "$f" .state)
        if [[ "$ss" == "running" ]] && [[ "$name" == train_* ]]; then
            echo "$name"
            return
        fi
    done
}

while true; do
    current=$(find_current)

    if [[ -z "$current" ]]; then
        clear
        echo "${DIM}等待训练任务启动...${RST}"
        sleep 5
        continue
    fi

    log_file="$LOG_DIR/${current}.log"
    [[ ! -f "$log_file" ]] && sleep 5 && continue

    clear
    short=${current#train_}
    echo "${BOLD}${C}━━ 实时训练: ${M}$short${RST}${BOLD}${C} ━━${RST}"
    echo "${DIM}(只显示关键事件，自动切换 fold)${RST}"
    echo ""

    # 显示历史 epoch 总结
    grep -E "^\[[0-9]+\-[0-9]+\-[0-9]+.*\]" "$log_file" 2>/dev/null | tail -n 15 | while IFS= read -r line; do
        if echo "$line" | grep -qE "Best|best"; then
            echo "${BOLD}${G}★ $line${RST}"
        elif echo "$line" | grep -qE "Val|mean=|dice"; then
            echo "${G}$line${RST}"
        elif echo "$line" | grep -qE "Epoch [0-9]+/"; then
            echo "${C}$line${RST}"
        else
            echo "${DIM}$line${RST}"
        fi
    done

    echo ""
    echo "${DIM}━━ 实时跟随 ━━${RST}"

    # 实时跟随：用 tail -F + 严格过滤
    (
        tail -F -n 0 "$log_file" 2>/dev/null | \
        stdbuf -oL grep -E "^\[[0-9]+\-[0-9]+\-[0-9]+|Best dice|saved|Error|FATAL|Warning|Loading dataset:" | \
        stdbuf -oL grep -vE "^Epoch [0-9]+: *[0-9]+%" | \
        while IFS= read -r line; do
            ts=$(date '+%H:%M:%S')
            # 过滤 Loading dataset 只保留 10% 倍数
            if echo "$line" | grep -qE "Loading dataset"; then
                pct=$(echo "$line" | grep -oE "[0-9]+%" | head -1 | tr -d '%')
                [[ -z "$pct" ]] && continue
                [[ $((pct % 10)) -ne 0 ]] && continue
                echo "${DIM}[$ts] 加载数据 ${pct}%${RST}"
                continue
            fi

            # 着色规则
            if echo "$line" | grep -qiE "error|fail|fatal|nan|inf"; then
                echo "${R}[$ts] $line${RST}"
            elif echo "$line" | grep -qiE "best dice|new best|saved.*best"; then
                echo "${BOLD}${G}[$ts] ★ $line${RST}"
            elif echo "$line" | grep -qiE "val|mean=|dice"; then
                echo "${G}[$ts] $line${RST}"
            elif echo "$line" | grep -qiE "epoch [0-9]+/"; then
                echo "${C}[$ts] $line${RST}"
            elif echo "$line" | grep -qiE "saved|checkpoint"; then
                echo "${Y}[$ts] $line${RST}"
            else
                echo "${DIM}[$ts]${RST} $line"
            fi
        done
    ) &
    TAIL_PID=$!

    # 监控任务变化
    while kill -0 $TAIL_PID 2>/dev/null; do
        sleep 10
        new_state=$(cut -d'|' -f1 "$STATE_DIR/${current}.state" 2>/dev/null)
        if [[ "$new_state" != "running" ]]; then
            kill $TAIL_PID 2>/dev/null
            wait $TAIL_PID 2>/dev/null
            echo ""
            echo "${Y}━━ $short 已结束 ($new_state)，切换下一个... ━━${RST}"
            sleep 3
            break
        fi
    done
done
