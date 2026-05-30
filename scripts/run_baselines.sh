#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CPU_THREADS=$(nproc)
CPU_LIMIT=${CPU_LIMIT:-14}
if [[ $CPU_THREADS -gt $CPU_LIMIT ]]; then
  CPU_THREADS=$CPU_LIMIT
fi
DEFAULT_THREADS=$CPU_THREADS
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$DEFAULT_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$DEFAULT_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$DEFAULT_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$DEFAULT_THREADS}"

for var in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS; do
  val="${!var}"
  if [[ -z "$val" || ! "$val" =~ ^[1-9][0-9]*$ ]]; then
    export "$var=$DEFAULT_THREADS"
  fi
done

BASE_CONFIG="${BASE_CONFIG:-configs/base_short.yaml}"
MODELS=${MODELS:-"unet3d vnet attention_unet resunet unetpp deepmedic unetr transbts"}
FOLDS=${FOLDS:-"0 1 2 3 4"}
GPU=${GPU:-0}
WORK_DIR="${WORK_DIR:-./work_dir}"
PROBS_DIR="${PROBS_DIR:-/dev/shm/brats_probs}"
SKIP_TRAIN_IF_BEST=${SKIP_TRAIN_IF_BEST:-1}
SKIP_PRED_IF_PROBS=${SKIP_PRED_IF_PROBS:-1}

mkdir -p "$WORK_DIR" "$PROBS_DIR"

echo "[INFO] base_config=$BASE_CONFIG"
echo "[INFO] models=$MODELS"
echo "[INFO] folds=$FOLDS"
echo "[INFO] threads: OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS NUMEXPR=$NUMEXPR_NUM_THREADS"
echo "[INFO] skip_train_if_best=$SKIP_TRAIN_IF_BEST skip_pred_if_probs=$SKIP_PRED_IF_PROBS"

for model in $MODELS; do
  for fold in $FOLDS; do
    last_ckpt="$WORK_DIR/${model}/fold_${fold}/last.pt"
    best_ckpt="$WORK_DIR/${model}/fold_${fold}/best.pt"
    resume_args=()
    if [[ -f "$last_ckpt" ]]; then
      echo "[INFO] resume from: $last_ckpt"
      resume_args=(--resume "$last_ckpt")
    fi
    if [[ "$SKIP_TRAIN_IF_BEST" == "1" && -f "$best_ckpt" ]]; then
      echo "[SKIP] train $model fold=$fold (found best.pt)"
    else
      echo "[TRAIN] $model fold=$fold"
      python scripts/train.py \
        --base_config "$BASE_CONFIG" \
        --model_config "configs/${model}.yaml" \
        --fold "$fold" --gpu "$GPU" \
        "${resume_args[@]}"
    fi

    ckpt="$WORK_DIR/${model}/fold_${fold}/best.pt"
    if [[ ! -f "$ckpt" ]]; then
      echo "[WARN] missing ckpt: $ckpt"
      continue
    fi

    pred_out="$PROBS_DIR/${model}/fold_${fold}"
    if [[ "$SKIP_PRED_IF_PROBS" == "1" ]] && ls "$pred_out"/*.npz >/dev/null 2>&1; then
      echo "[SKIP] pred $model fold=$fold (found npz in $pred_out)"
    else
      echo "[PRED] $model fold=$fold"
      python scripts/predict.py \
        --base_config "$BASE_CONFIG" \
        --model_config "configs/${model}.yaml" \
        --ckpt "$ckpt" \
        --fold "$fold" \
        --out_dir "$pred_out" \
        --use_ema --tta \
        --save_csv \
        --model_name "$model"
    fi
  done

  echo "[MERGE] $model OOF"
  mkdir -p "$PROBS_DIR/${model}_oof"
  cp -n "$PROBS_DIR/${model}/fold_"*"/"*.npz "$PROBS_DIR/${model}_oof/" 2>/dev/null || true
  echo "[INFO] $(ls "$PROBS_DIR/${model}_oof"/*.npz 2>/dev/null | wc -l) cases"
  echo
done