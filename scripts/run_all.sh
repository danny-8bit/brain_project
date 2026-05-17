#!/usr/bin/env bash
# ============================================================
# BraTS2021 SOTA 一键训练流水线
# 单卡 A800 80GB，预计 6~10 天完成
#
# 使用：
#   bash scripts/run_all.sh                # 全流程
#   bash scripts/run_all.sh --smoke        # 仅冒烟测试（2h）
#   bash scripts/run_all.sh --no-nnunet    # 跳过 nnU-Net
#   bash scripts/run_all.sh --infer-only   # 仅推理+融合
# ============================================================

set -uo pipefail

# ---------------- 配置 ----------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_ROOT="${DATA_ROOT:-./data/BraTS2021_TrainingData}"
FOLDS_JSON="${FOLDS_JSON:-./data/folds.json}"
WORK_DIR="${WORK_DIR:-./work_dir}"
LOG_DIR="${LOG_DIR:-./logs}"
STATE_DIR="${STATE_DIR:-./run_state}"

export nnUNet_raw="${nnUNet_raw:-/root/autodl-tmp/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/root/autodl-tmp/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-/root/autodl-tmp/nnUNet_results}"
NNUNET_DATASET_ID=1
NNUNET_DATASET_NAME="BraTSGlioma"

MONAI_MODELS=("segresnet" "swinunetr")
FOLDS=(0 1 2 3 4)
GPU=0

# 模式开关
SMOKE_TEST=0
RUN_NNUNET=1
RUN_TRAIN=1
RUN_INFER=1

# 解析参数
for arg in "$@"; do
    case "$arg" in
        --smoke)       SMOKE_TEST=1 ;;
        --no-nnunet)   RUN_NNUNET=0 ;;
        --infer-only)  RUN_TRAIN=0 ;;
        --no-infer)    RUN_INFER=0 ;;
        -h|--help)
            sed -n '1,15p' "${BASH_SOURCE[0]}"
            exit 0 ;;
        *) echo "未知参数: $arg" ; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR" "$STATE_DIR" "$WORK_DIR"

# ---------------- 颜色 ----------------
if [[ -t 1 ]]; then
    C_RED=$'\033[31m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'
    C_BLUE=$'\033[34m'; C_CYAN=$'\033[36m'; C_BOLD=$'\033[1m'; C_RST=$'\033[0m'
else
    C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""; C_CYAN=""; C_BOLD=""; C_RST=""
fi

# ---------------- 工具函数 ----------------
MAIN_LOG="$LOG_DIR/run_all.log"

log() {
    local ts="[$(date '+%Y-%m-%d %H:%M:%S')]"
    echo "${ts} $*" | tee -a "$MAIN_LOG"
}

section() {
    local title="$1"
    local sep="=================================================================="
    log ""
    log "${C_BOLD}${C_CYAN}${sep}${C_RST}"
    log "${C_BOLD}${C_CYAN}  ${title}${C_RST}"
    log "${C_BOLD}${C_CYAN}${sep}${C_RST}"
}

format_time() {
    local sec=$1
    if   [[ $sec -lt 60 ]];   then echo "${sec}s"
    elif [[ $sec -lt 3600 ]]; then printf "%dm%02ds" $((sec/60)) $((sec%60))
    else printf "%dh%02dm" $((sec/3600)) $((sec%3600/60))
    fi
}

# 运行一个步骤：自动跳过已完成、记录状态、不中断流程
run_step() {
    local step="$1"; shift
    local log_file="$LOG_DIR/${step}.log"
    local state_file="$STATE_DIR/${step}.state"

    if [[ -f "$state_file" && "$(cat $state_file)" == "done" ]]; then
        log "${C_BLUE}[SKIP]${C_RST}  $step  (已完成)"
        return 0
    fi

    log "${C_YELLOW}[START]${C_RST} $step"
    echo "running|$(date +%s)" > "$state_file"
    local t0=$(date +%s)

    if "$@" >"$log_file" 2>&1; then
        local elapsed=$(($(date +%s) - t0))
        echo "done|$elapsed" > "$state_file"
        log "${C_GREEN}[DONE]${C_RST}  $step  (耗时 $(format_time $elapsed))"
        return 0
    else
        local rc=$?
        local elapsed=$(($(date +%s) - t0))
        echo "failed|$elapsed|$rc" > "$state_file"
        log "${C_RED}[FAIL]${C_RST}  $step  (exit=$rc 耗时 $(format_time $elapsed))"
        log "       日志: $log_file"
        return $rc
    fi
}

trap 'log "${C_RED}[INTERRUPT]${C_RST} 用户中断 (Ctrl+C)"; exit 130' INT

# ---------------- 步骤 1: 环境检查 ----------------
check_env() {
    section "环境检查"

    if ! command -v nvidia-smi &>/dev/null; then
        log "${C_RED}[FATAL]${C_RST} nvidia-smi 不可用"
        exit 1
    fi

    log "GPU 信息:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 | \
        while read line; do log "  $line"; done

    log "PyTorch 检查:"
    python - <<'PY' 2>&1 | while read l; do log "  $l"; done
import torch, monai
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.cuda.is_available()} ({torch.version.cuda})")
print(f"MONAI    : {monai.__version__}")
print(f"GPU 数量 : {torch.cuda.device_count()}")
PY

    if [[ ! -d "$DATA_ROOT" ]]; then
        log "${C_RED}[FATAL]${C_RST} 数据目录不存在: $DATA_ROOT"
        exit 1
    fi

    local n_cases=$(find "$DATA_ROOT" -maxdepth 1 -mindepth 1 -type d | wc -l)
    log "数据集 case 数: ${C_BOLD}${n_cases}${C_RST}"

    log "内存:"
    free -h | awk 'NR==2{print "  total="$2"  used="$3"  available="$7}' | \
        while read l; do log "$l"; done

    log "磁盘:"
    df -h "$PROJECT_ROOT" /root/autodl-tmp 2>/dev/null | tail -n +2 | \
        awk '{printf "  %s  used=%s/%s (%s)\n", $6, $3, $2, $5}' | \
        while read l; do log "$l"; done
}

# ---------------- 步骤 2: 数据准备 ----------------
prepare_data() {
    section "数据准备"

    if [[ ! -f "$FOLDS_JSON" ]]; then
        run_step "make_folds" \
            python scripts/make_folds.py \
                --data_root "$DATA_ROOT" \
                --out "$FOLDS_JSON" \
                --num_folds 5
    else
        log "${C_BLUE}[SKIP]${C_RST}  make_folds  (已存在: $FOLDS_JSON)"
    fi

    if [[ "$RUN_NNUNET" == "1" ]]; then
        local ds_id_str=$(printf '%03d' $NNUNET_DATASET_ID)
        local ds_folder="${nnUNet_raw}/Dataset${ds_id_str}_${NNUNET_DATASET_NAME}"

        if [[ ! -f "$ds_folder/dataset.json" ]]; then
            run_step "nnunet_convert" \
                python prepare_nnunet_brats.py \
                    --src "$DATA_ROOT" \
                    --dst_raw "$nnUNet_raw" \
                    --dataset_id $NNUNET_DATASET_ID \
                    --dataset_name $NNUNET_DATASET_NAME
        else
            log "${C_BLUE}[SKIP]${C_RST}  nnunet_convert (已转换)"
        fi

        local pp_folder="${nnUNet_preprocessed}/Dataset${ds_id_str}_${NNUNET_DATASET_NAME}"
        if [[ ! -f "$pp_folder/nnUNetPlans.json" ]]; then
            run_step "nnunet_preprocess" \
                nnUNetv2_plan_and_preprocess -d $NNUNET_DATASET_ID --verify_dataset_integrity
        else
            log "${C_BLUE}[SKIP]${C_RST}  nnunet_preprocess (已完成)"
        fi
    fi
}

# ---------------- 步骤 3: 冒烟测试 ----------------
smoke_test() {
    section "冒烟测试 (50 epoch SegResNet fold 0)"

    # 临时配置：跑 50 epoch 看看是否能通
    cat > /tmp/smoke_base.yaml <<EOF
seed: 2024
gpu: 0
amp: true
deep_supervision: true
ema: true
ema_decay: 0.999
grad_clip: 1.0
data_root: $DATA_ROOT
folds_json: $FOLDS_JSON
num_folds: 5
num_workers: 8
cache_rate: 0.3
roi_size: [128, 128, 128]
batch_size: 2
sw_batch_size: 4
max_epochs: 50
warmup_epochs: 5
val_interval: 25
lr: 3.0e-4
weight_decay: 1.0e-5
use_swa: false
work_dir: ./work_dir_smoke
EOF

    run_step "smoke_test" \
        python scripts/train.py \
            --base_config /tmp/smoke_base.yaml \
            --model_config configs/segresnet.yaml \
            --fold 0 --gpu $GPU

    log "${C_GREEN}冒烟测试完成，可以查看 logs/smoke_test.log 确认 dice 是否合理${C_RST}"
}

# ---------------- 步骤 4: 训练 nnU-Net ----------------
train_nnunet() {
    section "训练 nnU-Net v2 (5-fold)"
    for FOLD in "${FOLDS[@]}"; do
        run_step "train_nnunet_fold${FOLD}" \
            nnUNetv2_train $NNUNET_DATASET_ID 3d_fullres $FOLD || true
    done
}

# ---------------- 步骤 5: 训练 MONAI 模型 ----------------
train_monai() {
    local model="$1"
    section "训练 ${model} (5-fold)"
    for FOLD in "${FOLDS[@]}"; do
        run_step "train_${model}_fold${FOLD}" \
            python scripts/train.py \
                --base_config configs/base.yaml \
                --model_config "configs/${model}.yaml" \
                --fold $FOLD --gpu $GPU || true
    done
}

# ---------------- 步骤 6: 推理 ----------------
predict_monai() {
    local model="$1"
    section "推理 ${model}"
    for FOLD in "${FOLDS[@]}"; do
        local ckpt="$WORK_DIR/${model}/fold_${FOLD}/best.pt"
        if [[ ! -f "$ckpt" ]]; then
            log "${C_YELLOW}[SKIP]${C_RST} predict_${model}_fold${FOLD} (ckpt 不存在)"
            continue
        fi
        run_step "predict_${model}_fold${FOLD}" \
            python scripts/predict.py \
                --base_config configs/base.yaml \
                --model_config "configs/${model}.yaml" \
                --ckpt "$ckpt" \
                --fold $FOLD \
                --out_dir "$WORK_DIR/probs/${model}/fold_${FOLD}" \
                --use_ema --tta || true
    done

    # 合并 OOF
    mkdir -p "$WORK_DIR/probs/${model}_oof"
    cp $WORK_DIR/probs/${model}/fold_*/*.npz "$WORK_DIR/probs/${model}_oof/" 2>/dev/null || true
    local n=$(ls "$WORK_DIR/probs/${model}_oof/"*.npz 2>/dev/null | wc -l)
    log "${C_GREEN}[INFO]${C_RST}  ${model} OOF 概率图: ${n} 个 case"
}

# ---------------- 步骤 7: 融合 + 网格搜索 ----------------
ensemble_and_search() {
    section "融合 + 网格搜索"

    local prob_dirs=()
    for d in nnunet "${MONAI_MODELS[@]/%/_oof}"; do
        [[ "$d" == "nnunet" ]] && d="nnunet_oof"
        if [[ -d "$WORK_DIR/probs/$d" ]] && \
           [[ $(ls "$WORK_DIR/probs/$d"/*.npz 2>/dev/null | wc -l) -gt 0 ]]; then
            prob_dirs+=("$WORK_DIR/probs/$d")
        fi
    done

    if [[ ${#prob_dirs[@]} -lt 2 ]]; then
        log "${C_RED}[FAIL]${C_RST} 可用模型 OOF 概率图少于 2 个，无法融合"
        return 1
    fi

    log "参与融合的模型:"
    for d in "${prob_dirs[@]}"; do log "  - $d"; done

    run_step "grid_search" \
        python scripts/grid_search.py \
            --prob_dirs "${prob_dirs[@]}" \
            --ref_dir "$DATA_ROOT" \
            --folds_json "$FOLDS_JSON"
}

# ---------------- 步骤 8: 汇总报告 ----------------
summary() {
    section "任务状态汇总"

    local n_total=0 n_done=0 n_failed=0 n_running=0 total_time=0

    printf "%-40s %-10s %s\n" "任务" "状态" "耗时"
    printf "%-40s %-10s %s\n" "----------------------------------------" "--------" "--------"

    for f in "$STATE_DIR"/*.state; do
        [[ -f "$f" ]] || continue
        local name=$(basename "$f" .state)
        local content=$(cat "$f")
        local status=$(echo "$content" | cut -d'|' -f1)
        local elapsed=$(echo "$content" | cut -d'|' -f2)

        n_total=$((n_total + 1))
        local color=""
        case "$status" in
            done)
                color="$C_GREEN"; n_done=$((n_done + 1))
                total_time=$((total_time + elapsed)) ;;
            failed)  color="$C_RED";    n_failed=$((n_failed + 1)) ;;
            running) color="$C_YELLOW"; n_running=$((n_running + 1)) ;;
        esac

        local time_str=""
        if [[ "$status" == "done" ]]; then
            time_str=$(format_time $elapsed)
        fi

        printf "%-40s ${color}%-10s${C_RST} %s\n" "$name" "$status" "$time_str" \
            | tee -a "$MAIN_LOG"
    done

    log ""
    log "总任务: $n_total | ${C_GREEN}完成: $n_done${C_RST} | ${C_RED}失败: $n_failed${C_RST} | ${C_YELLOW}运行中: $n_running${C_RST}"
    log "累计训练时间: $(format_time $total_time)"
}

# ---------------- 主流程 ----------------
main() {
    local pipeline_start=$(date +%s)

    section "BraTS2021 SOTA Pipeline 启动"
    log "PROJECT_ROOT : $PROJECT_ROOT"
    log "WORK_DIR     : $WORK_DIR"
    log "LOG_DIR      : $LOG_DIR"
    log "MODE         : smoke=$SMOKE_TEST  nnunet=$RUN_NNUNET  train=$RUN_TRAIN  infer=$RUN_INFER"

    check_env
    prepare_data

    if [[ "$SMOKE_TEST" == "1" ]]; then
        smoke_test
        log "${C_GREEN}冒烟测试模式结束。如确认无误，去掉 --smoke 跑完整流程。${C_RST}"
        exit 0
    fi

    if [[ "$RUN_TRAIN" == "1" ]]; then
        [[ "$RUN_NNUNET" == "1" ]] && train_nnunet
        for MODEL in "${MONAI_MODELS[@]}"; do
            train_monai "$MODEL"
        done
    fi

    if [[ "$RUN_INFER" == "1" ]]; then
        for MODEL in "${MONAI_MODELS[@]}"; do
            predict_monai "$MODEL"
        done
        ensemble_and_search
    fi

    summary

    local total=$(($(date +%s) - pipeline_start))
    section "Pipeline 完成 (总耗时 $(format_time $total))"
}

main "$@"