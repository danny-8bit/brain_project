#!/usr/bin/env bash
# ============================================================
# 一键启动：自动开 tmux + 训练 + 监控
#
# 用法:
#   bash scripts/launch.sh          # 启动完整训练
#   bash scripts/launch.sh attach   # 连上去看
#   bash scripts/launch.sh stop     # 停掉整个 pipeline
#   bash scripts/launch.sh status   # 查看状态
# ============================================================

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SESSION="brats"

cmd="${1:-start}"

case "$cmd" in
    start)
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "[ERR] tmux 会话 '$SESSION' 已存在，请先运行: bash scripts/launch.sh stop"
            echo "      或连上去看:  bash scripts/launch.sh attach"
            exit 1
        fi

        # 创建三窗格：训练 / 监控 / 命令行
        tmux new-session  -d -s "$SESSION" -n main
        tmux send-keys    -t "$SESSION:main" "source brain/bin/activate && bash scripts/run_all.sh ${@:2}" C-m

        tmux split-window -h -t "$SESSION:main"
        tmux send-keys    -t "$SESSION:main.1" "source brain/bin/activate && sleep 3 && bash scripts/monitor.sh" C-m

        tmux split-window -v -t "$SESSION:main.1"
        tmux send-keys    -t "$SESSION:main.2" "source brain/bin/activate && bash scripts/live_train.sh" C-m

        tmux select-pane  -t "$SESSION:main.0"

        echo "✓ 已启动 tmux 会话 '$SESSION'"
        echo ""
        echo "  连上去看:        bash scripts/launch.sh attach"
        echo "  查看状态:        bash scripts/launch.sh status"
        echo "  停止整个流程:    bash scripts/launch.sh stop"
        echo ""
        echo "  (3 秒后自动 attach...)"
        sleep 3
        tmux attach -t "$SESSION"
        ;;

    attach|a)
        tmux attach -t "$SESSION" || echo "[ERR] 没有运行中的会话"
        ;;

    stop|kill)
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            tmux kill-session -t "$SESSION"
            echo "✓ 已停止会话 '$SESSION'"
            echo "  注意：已经在运行的 python 进程可能未被杀掉，用 nvidia-smi 检查"
        else
            echo "[INFO] 没有运行中的会话"
        fi
        ;;

    status|s)
        bash scripts/status.sh
        echo ""
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "tmux 会话: ${SESSION} (运行中)"
        else
            echo "tmux 会话: 未运行"
        fi
        ;;

    *)
        echo "用法: $0 {start|attach|stop|status}"
        exit 1 ;;
esac