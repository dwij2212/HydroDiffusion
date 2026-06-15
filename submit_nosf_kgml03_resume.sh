#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=nosf_msigpu_03
#SBATCH --output=/users/8/zhan8460/Desktop/HydroDiffusion/logs/nosf_msigpu_03_%j.txt
#SBATCH --error=/users/8/zhan8460/Desktop/HydroDiffusion/logs/nosf_msigpu_03_err_%j.txt
#SBATCH --time=2-00:00:00
#SBATCH --partition=kgml03
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --open-mode=append

source ~/.bashrc
conda activate hydrodiff
export LD_LIBRARY_PATH=/users/8/zhan8460/anaconda3/lib:$LD_LIBRARY_PATH
cd ~/Desktop/HydroDiffusion
git checkout no-sf-benchmark

# --- decoder_only_ssm seed=3407 ---
python3 main.py train_npy \
  --model_name=decoder_only_ssm \
  --seed=3407 \
  --gpu=0 \
  --no_static=false \
  --concat_static=true \
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
  --pool_type=power \
  --forcing_source=daymet \
  --run_dir=/users/8/zhan8460/Desktop/HydroDiffusion/runs/decoder_only_ssm_nosf_seed3407

# --- decoder_only_lstm seed=3407 ---
python3 main.py train_npy \
  --model_name=decoder_only_lstm \
  --seed=3407 \
  --gpu=0 \
  --no_static=false \
  --concat_static=true \
  --epochs=60 \
  --lr=3e-5 \
  --predict_mode=velocity \
  --forcing_source=daymet \
  --run_dir=/users/8/zhan8460/Desktop/HydroDiffusion/runs/decoder_only_lstm_nosf_seed3407

# --- diffusion_lstm seed=3407 ---
python3 main.py train_npy \
  --model_name=diffusion_lstm \
  --seed=3407 \
  --gpu=0 \
  --no_static=false \
  --concat_static=true \
  --epochs=60 \
  --lr=3e-5 \
  --predict_mode=velocity \
  --forcing_source=daymet \
  --run_dir=/users/8/zhan8460/Desktop/HydroDiffusion/runs/diffusion_lstm_nosf_seed3407
