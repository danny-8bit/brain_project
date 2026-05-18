#!/usr/bin/env bash
# 后台自动清理 + 磁盘告警
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

WORK_DIR="${WORK_DIR:-./work_dir}"
LOG_DIR="${LOG_DIR:-./logs}"
PROBS_DIR="${PROBS_DIR:-/dev/shm/brats_probs}"
PID_FILE="/tmp/brats_cleanup.pid"
CLEANUP_LOG="$LOG_DIR/auto_cleanup.log"

DISK_WARN=82
DISK_CRIT=90
SHM_WARN=80
SHM_CRIT=90
INTERVAL=600

log_msg() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$CLEANUP_LOG"; }

get_pct() { df "$1" 2>/dev/null | awk 'NR==2 {gsub("%",""); print $5}'; }

cleanup_loop() {
    log_msg "▶ 自动清理启动 (PID=$$)"
    local tick=0
    while true; do
        local disk=$(get_pct /root/autodl-tmp)
        local shm=$(get_pct /dev/shm)
        disk=${disk:-0}; shm=${shm:-0}

        # 常规：兜底删 last.pt
        find "$WORK_DIR" -name "last.pt" -mmin +30 -delete 2>/dev/null || true

        # 磁盘紧急
        if [[ $disk -ge $DISK_CRIT ]]; then
            log_msg "🚨 磁盘 ${disk}% 紧急清理"
            find "$WORK_DIR" -name "last.pt" -delete 2>/dev/null
            for f in "$LOG_DIR"/*.log; do
                [[ -f "$f" ]] || continue
                [[ $(stat -c%s "$f") -gt 10485760 ]] && tail -n 200 "$f" > "$f.tmp" && mv "$f.tmp" "$f"
            done
        elif [[ $disk -ge $DISK_WARN ]]; then
            [[ $((tick % 6)) -eq 0 ]] && log_msg "⚠ 磁盘 ${disk}%"
        fi

        # /dev/shm 紧急
        if [[ $shm -ge $SHM_CRIT ]]; then
            log_msg "🚨 /dev/shm ${shm}% 紧急"
        elif [[ $shm -ge $SHM_WARN ]]; then
            [[ $((tick % 6)) -eq 0 ]] && log_msg "⚠ /dev/shm ${shm}%"
        fi

        [[ $((tick % 6)) -eq 0 ]] && log_msg "📊 磁盘 ${disk}% | /dev/shm ${shm}%"
        tick=$((tick+1))
        sleep $INTERVAL
    done
}

case "${1:-status}" in
    start)
        if [[ -f "$PID_FILE" ]] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
            echo "已运行 (PID=$(cat $PID_FILE))"; exit 0
        fi
        mkdir -p "$LOG_DIR"
        nohup bash "$0" _daemon > /dev/null 2>&1 &
        echo $! > "$PID_FILE"
        sleep 1
        echo "✓ 启动 (PID=$(cat $PID_FILE))"
        ;;
    _daemon)
        cleanup_loop
        ;;
    stop)
        if [[ -f "$PID_FILE" ]] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
            kill "$(cat $PID_FILE)"; rm -f "$PID_FILE"
            echo "✓ 已停止"
        else
            rm -f "$PID_FILE"
            echo "未运行"
        fi
        ;;
    status)
        if [[ -f "$PID_FILE" ]] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
            echo "✓ 运行中 (PID=$(cat $PID_FILE))"
            echo "磁盘: $(df -h /root/autodl-tmp | awk 'NR==2{print $3"/"$2" ("$5")"}')"
            echo "/dev/shm: $(df -h /dev/shm | awk 'NR==2{print $3"/"$2" ("$5")"}')"
            echo ""
            tail -n 8 "$CLEANUP_LOG" 2>/dev/null
        else
            echo "✗ 未运行"
        fi
        ;;
    *) echo "用法: $0 {start|stop|status}"; exit 1 ;;
esac
