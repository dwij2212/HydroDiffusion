#!/usr/bin/env bash
set -euo pipefail

cd /projects/standard/kumarv/shared/dwij/HydroDiffusion
mkdir -p logs/tmux_evals

PYTHON=/users/6/mehta423/anaconda3/envs/wstatt/bin/python
MODEL_NAME="${1:?Usage: $0 MODEL_NAME PHYSICAL_GPU_ID [EVAL_DATASET]}"
PHYSICAL_GPU_ID="${2:?Usage: $0 MODEL_NAME PHYSICAL_GPU_ID [EVAL_DATASET]}"
EVAL_DATASET="${3:-${EVAL_DATASET:-test}}"
SEEDS=(3407 3408 3409 3410 3411)

BASIN_ARGS=()
if [[ -n "${BASIN_ASSIGNMENT_CSV:-}" ]]; then
  BASIN_ARGS=(--basin_split_csv "$BASIN_ASSIGNMENT_CSV")
fi

MODEL_ARGS=()
case "$MODEL_NAME" in
  seq2seq_lstm)
    EPOCHS=30
    ;;
  encdec_lstm)
    EPOCHS=30
    ;;
  seq2seq_ssm)
    EPOCHS=50
    MODEL_ARGS=(
      --d_model 128 --d_state 128 --lr 4e-4 --lr_min 4e-5
      --weight_decay 3e-2 --wd 2e-2 --lr_dt 0.001
      --min_dt 0.01 --max_dt 0.1 --warmup 0 --n_layers 6
      --ssm_dropout 0.12 --cfi 10 --cfr 10
    )
    ;;
  decoder_only_ssm)
    EPOCHS=60
    MODEL_ARGS=(
      --d_model 256 --d_state 256 --lr 3e-5 --lr_min 3e-6
      --weight_decay 0.00 --wd 4e-5 --lr_dt 0.001
      --min_dt 0.01 --max_dt 0.1 --warmup 1 --n_layers 6
      --batch_size 128 --ssm_dropout 0.2 --cfi 10 --cfr 10
      --pool_type power --predict_mode velocity
    )
    ;;
  decoder_only_lstm)
    EPOCHS=60
    ;;
  diffusion_lstm)
    EPOCHS=60
    MODEL_ARGS=(--predict_mode velocity)
    ;;
  *)
    echo "ERROR: unknown model '$MODEL_NAME'" >&2
    exit 2
    ;;
esac

find_latest_run_dir() {
  local model=$1
  local seed=$2
  local latest_dir

  latest_dir=$(find runs -maxdepth 1 -type d -name "*${model}*seed${seed}*" 2>/dev/null | sort -V | tail -1)

  if [[ -z "$latest_dir" ]]; then
    latest_dir=$(find runs -maxdepth 1 -type d -name "*seed${seed}*" 2>/dev/null | sort -V | tail -1)
  fi

  echo "$latest_dir"
}

echo "Evaluating $MODEL_NAME on physical GPU $PHYSICAL_GPU_ID"
echo "Eval dataset: $EVAL_DATASET"
echo "Python: $PYTHON"
echo "Started: $(date)"

for SEED in "${SEEDS[@]}"; do
  RUN_DIR=$(find_latest_run_dir "$MODEL_NAME" "$SEED")

  if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
    echo "ERROR: could not find trained run directory for $MODEL_NAME seed $SEED" >&2
    exit 1
  fi

  if [[ ! -f "$RUN_DIR/best_model.pt" ]]; then
    echo "ERROR: $RUN_DIR does not contain best_model.pt" >&2
    exit 1
  fi

  LOG="logs/tmux_evals/${MODEL_NAME}_seed${SEED}_${EVAL_DATASET}_gpu${PHYSICAL_GPU_ID}_$(date +%Y%m%d_%H%M%S).log"
  echo "=========================================="
  echo "Evaluating: Model=$MODEL_NAME, Seed=$SEED, Physical GPU=$PHYSICAL_GPU_ID"
  echo "Run dir: $RUN_DIR"
  echo "Log: $LOG"
  echo "=========================================="

  CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU_ID" "$PYTHON" main.py evaluate_npy \
    --model_name "$MODEL_NAME" \
    --seed "$SEED" \
    --gpu 0 \
    --no_static false \
    --concat_static true \
    --run_dir "$RUN_DIR" \
    --epochs "$EPOCHS" \
    --eval_dataset "$EVAL_DATASET" \
    --forcing_source daymet \
    "${MODEL_ARGS[@]}" \
    "${BASIN_ARGS[@]}" \
    2>&1 | tee "$LOG"

  echo "Completed: Model=$MODEL_NAME, Seed=$SEED, Run dir=$RUN_DIR at $(date)"
done

echo "All five seeds evaluated for $MODEL_NAME at $(date)"
