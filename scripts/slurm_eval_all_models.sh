#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=hydrodiff_eval
#SBATCH --output=logs/hydrodiff_eval_%A_%a.out
#SBATCH --error=logs/hydrodiff_eval_%A_%a.err
#SBATCH --time=23:59:59
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=mehta423@umn.edu
#SBATCH --array=0-4

set -e
mkdir -p logs

cd /projects/standard/kumarv/shared/dwij/HydroDiffusion

# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nsdiff

# ===== CONFIGURE THIS FOR EACH MODEL =====
MODEL_NAME="${MODEL_NAME:-seq2seq_lstm}"
# seq2seq_lstm | encdec_lstm | seq2seq_ssm | decoder_only_ssm | decoder_only_lstm | diffusion_lstm

EVAL_DATASET="${EVAL_DATASET:-test}"  # Options: "val" or "test"

# Define seeds
SEEDS=(3407 3408 3409 3410 3411)

# Get the seed for this array task (default to 0 if not running through sbatch)
ARRAY_ID=${SLURM_ARRAY_TASK_ID:-0}
SEED=${SEEDS[$ARRAY_ID]}

echo "=========================================="
echo "Evaluating: Model=$MODEL_NAME, Seed=$SEED"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-0 (local)}"
echo "EVAL_DATASET: $EVAL_DATASET"
echo "=========================================="

# Build basin assignment argument if env var is set
BASIN_ARG=""
if [[ -n "$BASIN_ASSIGNMENT_CSV" ]]; then
  BASIN_ARG="--basin_split_csv=$BASIN_ASSIGNMENT_CSV"
  echo "Using basin assignment: $BASIN_ASSIGNMENT_CSV"
fi

# Function to find the latest run directory for a model and seed
find_latest_run_dir() {
  local model=$1
  local seed=$2
  
  # Look for directories matching pattern with model name
  local latest_dir=$(find runs -maxdepth 1 -type d -name "*${model}*seed${seed}*" 2>/dev/null | \
                     sort -V | tail -1)
  
  if [[ -z "$latest_dir" ]]; then
    # Fallback: try without model name, just seed
    latest_dir=$(find runs -maxdepth 1 -type d -name "*seed${seed}*" 2>/dev/null | \
                 sort -V | tail -1)
  fi
  
  echo "$latest_dir"
}

# Find the trained model directory
RUN_DIR=$(find_latest_run_dir "$MODEL_NAME" "$SEED")

if [[ -z "$RUN_DIR" ]] || [[ ! -d "$RUN_DIR" ]]; then
  echo "ERROR: Could not find trained model directory for $MODEL_NAME with seed $SEED"
  echo "Searched in runs/ directory. Please ensure training completed successfully."
  exit 1
fi

echo "Using run directory: $RUN_DIR"

# Model-specific hyperparameters
if [[ "$MODEL_NAME" == "seq2seq_lstm" ]]; then
  EPOCHS=30
  MODEL_ARGS=""

elif [[ "$MODEL_NAME" == "encdec_lstm" ]]; then
  EPOCHS=30
  MODEL_ARGS=""

elif [[ "$MODEL_NAME" == "seq2seq_ssm" ]]; then
  EPOCHS=50
  MODEL_ARGS="--d_model=128 --d_state=128 --lr=4e-4 --lr_min=4e-5 --weight_decay=3e-2 --wd=2e-2 --lr_dt=0.001 --min_dt=0.01 --max_dt=0.1 --warmup=0 --n_layer=6 --ssm_dropout=0.12 --cfi=10 --cfr=10"

elif [[ "$MODEL_NAME" == "decoder_only_ssm" ]]; then
  EPOCHS=60
  MODEL_ARGS="--d_model=256 --d_state=256 --lr=3e-5 --lr_min=3e-6 --weight_decay=0.00 --wd=4e-5 --lr_dt=0.001 --min_dt=0.01 --max_dt=0.1 --warmup=1 --n_layer=6 --batch_size=128 --ssm_dropout=0.2 --cfi=10 --cfr=10 --pool_type='power' --predict_mode='velocity'"

elif [[ "$MODEL_NAME" == "decoder_only_lstm" ]]; then
  EPOCHS=60
  MODEL_ARGS=""

elif [[ "$MODEL_NAME" == "diffusion_lstm" ]]; then
  EPOCHS=60
  MODEL_ARGS="--predict_mode='velocity'"

else
  echo "ERROR: Unknown model $MODEL_NAME"
  exit 1
fi

echo "Running evaluation for $MODEL_NAME..."

python3 main.py evaluate_npy \
  --model_name="$MODEL_NAME" \
  --seed="$SEED" \
  --gpu=0 \
  --no_static=false \
  --concat_static=true \
  --run_dir="$RUN_DIR" \
  --epochs="$EPOCHS" \
  --eval_dataset="$EVAL_DATASET" \
  --forcing_source='daymet' \
  $MODEL_ARGS \
  $BASIN_ARG

if [ $? -eq 0 ]; then
  echo "=========================================="
  echo "Evaluation completed successfully for $MODEL_NAME with seed $SEED"
  echo "=========================================="
else
  echo "ERROR: Evaluation failed for $MODEL_NAME with seed $SEED at $RUN_DIR"
  exit 1
fi
