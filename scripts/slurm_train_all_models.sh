#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=hydrodiff_train
#SBATCH --output=logs/hydrodiff_train_%A_%a.out
#SBATCH --error=logs/hydrodiff_train_%A_%a.err
#SBATCH --time=23:59:59
#SBATCH --partition=kgml01,msigpu
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

# Define seeds
SEEDS=(3407 3408 3409 3410 3411)

# Get the seed for this array task (default to 0 if not running through sbatch)
ARRAY_ID=${SLURM_ARRAY_TASK_ID:-0}
SEED=${SEEDS[$ARRAY_ID]}

echo "=========================================="
echo "Training: Model=$MODEL_NAME, Seed=$SEED"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-0 (local)}"
echo "=========================================="

# Build basin assignment argument if env var is set
BASIN_ARG=""
if [[ -n "$BASIN_ASSIGNMENT_CSV" ]]; then
  BASIN_ARG="--basin_split_csv=$BASIN_ASSIGNMENT_CSV"
  echo "Using basin assignment: $BASIN_ASSIGNMENT_CSV"
fi

# Model-specific hyperparameters
if [[ "$MODEL_NAME" == "seq2seq_lstm" ]]; then
  EPOCHS=30
  MODEL_ARGS=""

elif [[ "$MODEL_NAME" == "encdec_lstm" ]]; then
  EPOCHS=30
  MODEL_ARGS=""

elif [[ "$MODEL_NAME" == "seq2seq_ssm" ]]; then
  EPOCHS=50
  MODEL_ARGS="--d_model=128 --d_state=128 --lr=4e-4 --lr_min=4e-5 --weight_decay=3e-2 --wd=2e-2 --lr_dt=0.001 --min_dt=0.01 --max_dt=0.1 --warmup=0 --n_layers=6 --ssm_dropout=0.12 --cfi=10 --cfr=10"

elif [[ "$MODEL_NAME" == "decoder_only_ssm" ]]; then
  EPOCHS=60
  MODEL_ARGS="--d_model=256 --d_state=256 --lr=3e-5 --lr_min=3e-6 --weight_decay=0.00 --wd=4e-5 --lr_dt=0.001 --min_dt=0.01 --max_dt=0.1 --warmup=1 --n_layers=6 --ssm_dropout=0.2 --cfi=10 --cfr=10 --pool_type='power'"

elif [[ "$MODEL_NAME" == "decoder_only_lstm" ]]; then
  EPOCHS=60
  MODEL_ARGS=""

elif [[ "$MODEL_NAME" == "diffusion_lstm" ]]; then
  EPOCHS=60
  MODEL_ARGS="--predict_mode=velocity"

else
  echo "ERROR: Unknown model $MODEL_NAME"
  exit 1
fi

echo "Running training for $MODEL_NAME..."

python3 main.py train_npy \
  --model_name="$MODEL_NAME" \
  --seed="$SEED" \
  --gpu=0 \
  --no_static=false \
  --concat_static=true \
  --epochs="$EPOCHS" \
  --forcing_source='daymet' \
  $MODEL_ARGS \
  $BASIN_ARG

if [ $? -eq 0 ]; then
  echo "=========================================="
  echo "Training completed successfully for $MODEL_NAME with seed $SEED"
  echo "=========================================="
else
  echo "ERROR: Training failed for $MODEL_NAME with seed $SEED"
  exit 1
fi
