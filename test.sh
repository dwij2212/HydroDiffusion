#!/usr/bin/env bash
#
# Usage: ./test.sh <model> <static|no_static> [gpu] [run_dir] [note]
#   model : seq2seq_lstm | encdec_lstm | seq2seq_ssm | diffusion_lstm | diffusion_unet | diffusion_ssm
#   flag  : static | no_static
#   gpu   : CUDA device ID (default 0)
#   run_dir: the running directory
#   note  : optional tag (no spaces) added to logfile name

model=$1
static_flag=$2
gpu_id=${3:-0}
run_dir=$4
note=${5:-}

export CUDA_VISIBLE_DEVICES="$gpu_id"

nseeds=1
firstseed=5534

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

mkdir -p reports

# ============================= main loop =====================================
for (( seed=firstseed; seed<firstseed+nseeds; seed++ )); do
  echo "=== seed=$seed | model=$model | no_static=$no_static | concat_static=$concat_static | GPU=$gpu_id | note='$note' ==="

  logfile="reports/test_${model}_${static_flag}_${seed}${note:+_${note}}.out"

  if [[ "$model" == "diffusion_ssm" ]]; then
    # ---------------------------- SSM branch ---------------------------------
    python3 main.py evaluate \
      --model_name="$model" \
      --seed="$seed" \
      --gpu=0 \
      --no_static="$no_static" \
      --concat_static="$concat_static" \
      --run_dir="$run_dir" \
      --epochs=200 \
      --d_model=256 \
      --lr=0.0002 \
      --lr_min=0.00004 \
      --weight_decay=0.00 \
      --wd=4e-5 \
      --lr_dt=0.001 \
      --min_dt=0.01 \
      --max_dt=0.1 \
      --warmup=1 \
      --n_layer=6 \
      --batch_size=128 \
      --ssm_dropout=0.2 \
      --d_state=256 \
      --cfi=10 \
      --cfr=10 \
      --pool_type='power'\
    > "$logfile" 2>&1 &
  else
    # ------------------------- other models branch ---------------------------
    python3 main.py evaluate \
      --model_name="$model" \
      --seed="$seed" \
      --gpu=0 \
      --no_static="$no_static" \
      --concat_static="$concat_static" \
      --run_dir="$run_dir" \
    > "$logfile" 2>&1 &
  fi

  echo "logs written to $logfile"
done
