#!/usr/bin/env bash
set -euo pipefail

cd /projects/standard/kumarv/shared/dwij/HydroDiffusion
mkdir -p logs/tmux_runs

PYTHON=/users/6/mehta423/anaconda3/envs/nsdiff/bin/python
MODEL_NAME="${1:?Usage: $0 MODEL_NAME PHYSICAL_GPU_ID}"
PHYSICAL_GPU_ID="${2:?Usage: $0 MODEL_NAME PHYSICAL_GPU_ID}"
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
      --ssm_dropout 0.2 --cfi 10 --cfr 10 --pool_type power
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

echo "Running $MODEL_NAME on physical GPU $PHYSICAL_GPU_ID"
echo "Python: $PYTHON"
echo "Started: $(date)"

for SEED in "${SEEDS[@]}"; do
  LOG="logs/tmux_runs/${MODEL_NAME}_seed${SEED}_gpu${PHYSICAL_GPU_ID}_$(date +%Y%m%d_%H%M%S).log"
  echo "=========================================="
  echo "Training: Model=$MODEL_NAME, Seed=$SEED, Physical GPU=$PHYSICAL_GPU_ID"
  echo "Log: $LOG"
  echo "=========================================="

  CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU_ID" "$PYTHON" main.py train_npy \
    --model_name "$MODEL_NAME" \
    --seed "$SEED" \
    --gpu 0 \
    --no_static false \
    --concat_static true \
    --epochs "$EPOCHS" \
    --forcing_source daymet \
    "${MODEL_ARGS[@]}" \
    "${BASIN_ARGS[@]}" \
    2>&1 | tee "$LOG"

  echo "Completed: Model=$MODEL_NAME, Seed=$SEED at $(date)"
done

echo "All five seeds completed for $MODEL_NAME at $(date)"
