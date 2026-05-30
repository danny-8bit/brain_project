#!/usr/bin/env bash
set -euo pipefail

SESSION=${SESSION:-baselines}
LOG=${LOG:-logs/baselines_tmux.log}
CPU_LIMIT=${CPU_LIMIT:-14}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[FATAL] tmux not found. Please install tmux first."
  exit 1
fi

mkdir -p "$PROJECT_ROOT/logs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[INFO] tmux session already exists: $SESSION"
  echo "       Attach with: tmux attach -t $SESSION"
  exit 0
fi

CMD="cd '$PROJECT_ROOT' && source brain/bin/activate && CPU_LIMIT=$CPU_LIMIT bash scripts/run_baselines.sh 2>&1 | tee -a '$LOG'"

tmux new-session -d -s "$SESSION" "bash -lc \"$CMD\""

echo "[OK] started tmux session: $SESSION"
echo "     Log file: $LOG"
echo "     Attach: tmux attach -t $SESSION"
