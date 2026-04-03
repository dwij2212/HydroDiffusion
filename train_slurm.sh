#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=hydrodiff_decoder_ssm
#SBATCH --output=logs/hydrodiff_decoder_ssm_%A_%a.out
#SBATCH --error=logs/hydrodiff_decoder_ssm_%A_%a.err
#SBATCH --time=23:59:59
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehta423@umn.edu
#SBATCH --array=0-4

# Usage: sbatch train_slurm.sh [static|no_static]
#   static_flag defaults to "static"

set -e
mkdir -p logs reports

cd /users/6/mehta423/projects/HydroDiffusion

# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nsdiff

static_flag=${1:-static}

# -------------------------------- static flags -------------------------------
if [[ "$static_flag" == "static" ]]; then
  no_static=false
  concat_static=true
elif [[ "$static_flag" == "no_static" ]]; then
  no_static=true
  concat_static=false
else
  echo "ERROR: static_flag must be 'static' or 'no_static'"
  exit 1
fi

# Seeds matching original train.sh (firstseed=3407, nseeds=5)
SEEDS=(3407 3408 3409 3410 3411)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID | seed=$SEED | static_flag=$static_flag | no_static=$no_static | concat_static=$concat_static ==="

python3 main.py train_npy \
  --model_name="decoder_only_ssm" \
  --seed="$SEED" \
  --gpu=0 \
  --no_static="$no_static" \
  --concat_static="$concat_static" \
  --epochs=60 \
  --d_model=256 \
  --d_state=256 \
  --lr=3e-5 \
  --lr_min=3e-6 \
  --weight_decay=0.00 \
  --wd=4e-5 \
  --lr_dt=0.001 \
  --min_dt=0.01 \
  --max_dt=0.1 \
  --warmup=1 \
  --n_layers=6 \
  --ssm_dropout=0.2 \
  --cfi=10 \
  --cfr=10 \
  --pool_type='power' \
  --forcing_source='daymet'

echo "Done: seed=$SEED static_flag=$static_flag"
