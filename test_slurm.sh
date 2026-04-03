#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=hydrodiff_eval_ens
#SBATCH --output=logs/hydrodiff_eval_ens_%A_%a.out
#SBATCH --error=logs/hydrodiff_eval_ens_%A_%a.err
#SBATCH --time=23:59:59
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehta423@umn.edu
#SBATCH --array=0-4

# Usage:
#   sbatch test_slurm.sh <model> <static|no_static> [test_stride] [runs_root] [run_tag] [note]
#
# Examples:
#   sbatch test_slurm.sh decoder_only_ssm static
#   sbatch test_slurm.sh decoder_only_ssm static 7
#   sbatch test_slurm.sh decoder_only_ssm static 7 runs 2803_1204 eval7
#
# Notes:
#   - One array task evaluates one seed from [3407..3411].
#   - If run_tag is not provided, the latest run_*_seed<seed> directory in runs_root is used.
#   - Local shell runs are supported by defaulting array_id=0 unless SLURM_ARRAY_TASK_ID is set.

set -eo pipefail

mkdir -p logs reports
cd /users/6/mehta423/projects/HydroDiffusion

source ~/anaconda3/etc/profile.d/conda.sh
conda activate nsdiff

model=${1:-decoder_only_ssm}
static_flag=${2:-static}
test_stride=${3:-7}
runs_root=${4:-runs}
run_tag=${5:-}
note=${6:-}

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

if ! [[ "$test_stride" =~ ^[0-9]+$ ]] || [[ "$test_stride" -lt 1 ]]; then
  echo "ERROR: test_stride must be a positive integer"
  exit 1
fi

SEEDS=(3407 3408 3409 3410 3411)
array_id="${SLURM_ARRAY_TASK_ID:-0}"
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "INFO: SLURM_ARRAY_TASK_ID is not set. Running in local mode with array_id=0 (seed=${SEEDS[0]})."
  echo "INFO: To test another seed locally, run: SLURM_ARRAY_TASK_ID=<0-4> bash test_slurm.sh ..."
fi
if (( array_id < 0 || array_id >= ${#SEEDS[@]} )); then
  echo "ERROR: array_id=$array_id is out of range 0-$(( ${#SEEDS[@]} - 1 ))"
  exit 1
fi
SEED=${SEEDS[$array_id]}

pattern="run_*_seed${SEED}"
if [[ -n "$run_tag" ]]; then
  pattern="run_${run_tag}*_seed${SEED}"
fi

run_dir=""
while IFS= read -r candidate; do
  if [[ -f "$candidate/best_model.pt" ]]; then
    run_dir="$candidate"
    break
  fi
done < <(find "$runs_root" -maxdepth 1 -mindepth 1 -type d -name "$pattern" -printf '%T@ %p\n' | sort -nr | awk '{print $2}')

if [[ -z "$run_dir" ]]; then
  echo "ERROR: Could not find run dir with best_model.pt for seed=$SEED using pattern '$pattern' under '$runs_root'"
  exit 1
fi

logfile="reports/test_slurm_${model}_${static_flag}_seed${SEED}_stride${test_stride}${note:+_${note}}.out"

echo "=== array_id=$array_id | seed=$SEED | model=$model | static_flag=$static_flag | test_stride=$test_stride ==="
echo "=== run_dir=$run_dir ==="
echo "=== Python path: $(which python3) ==="
echo "=== Conda env: $CONDA_DEFAULT_ENV ==="

{
  if [[ "$model" == "decoder_only_ssm" ]]; then
    python3 main.py evaluate_npy \
      --model_name="$model" \
      --seed="$SEED" \
      --gpu=0 \
      --no_static="$no_static" \
      --concat_static="$concat_static" \
      --run_dir="$run_dir" \
      --test_stride="$test_stride" \
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
      --batch_size=128 \
      --ssm_dropout=0.2 \
      --cfi=10 \
      --cfr=10 \
      --pool_type='power' \
      --predict_mode='velocity' \
      --forcing_source='daymet' 

  elif [[ "$model" == "seq2seq_ssm" ]]; then
    python3 main.py evaluate_npy \
      --model_name="$model" \
      --seed="$SEED" \
      --gpu=0 \
      --no_static="$no_static" \
      --concat_static="$concat_static" \
      --run_dir="$run_dir" \
      --test_stride="$test_stride" \
      --epochs=60 \
      --d_model=128 \
      --d_state=128 \
      --lr=4e-4 \
      --lr_min=4e-5 \
      --weight_decay=3e-2 \
      --wd=2e-2 \
      --lr_dt=0.001 \
      --min_dt=0.01 \
      --max_dt=0.1 \
      --warmup=0 \
      --n_layers=6 \
      --ssm_dropout=0.12 \
      --cfi=10 \
      --cfr=10 \
      --forcing_source='daymet'

  elif [[ "$model" == "seq2seq_lstm" || "$model" == "encdec_lstm" ]]; then
    python3 main.py evaluate_npy \
      --model_name="$model" \
      --seed="$SEED" \
      --gpu=0 \
      --no_static="$no_static" \
      --concat_static="$concat_static" \
      --run_dir="$run_dir" \
      --test_stride="$test_stride" \
      --epochs=30 \
      --forcing_source='daymet'

  else
    python3 main.py evaluate_npy \
      --model_name="$model" \
      --seed="$SEED" \
      --gpu=0 \
      --no_static="$no_static" \
      --concat_static="$concat_static" \
      --run_dir="$run_dir" \
      --test_stride="$test_stride" \
      --predict_mode='velocity' \
      --forcing_source='daymet'
  fi

  echo "Done: seed=$SEED model=$model test_stride=$test_stride run_dir=$run_dir"
} 2>&1 | tee "$logfile"

echo "Logs written to $logfile"
