#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=ssm_3410
#SBATCH --output=/users/8/zhan8460/Desktop/HydroDiffusion/logs/ssm_3410_%j.txt
#SBATCH --error=/users/8/zhan8460/Desktop/HydroDiffusion/logs/ssm_3410_err_%j.txt
#SBATCH --time=96:00:00
#SBATCH --partition=kgml02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

source ~/.bashrc
conda activate hydrodiff
cd ~/Desktop/HydroDiffusion
for seed in 3410 3411; do
    python main.py train_npy \
      --model_name=seq2seq_ssm \
      --seed=$seed \
      --gpu=0 \
      --no_static=false \
      --concat_static=true \
      --epochs=50 \
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
      --forcing_source=daymet
done
