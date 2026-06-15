#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=seq2seq_ssm_3409
#SBATCH --output=/users/8/zhan8460/Desktop/HydroDiffusion/logs/seq2seq_ssm_3409_%j.txt
#SBATCH --error=/users/8/zhan8460/Desktop/HydroDiffusion/logs/seq2seq_ssm_3409_err_%j.txt
#SBATCH --time=24:00:00
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

source ~/.bashrc
conda activate hydrodiff
cd ~/Desktop/HydroDiffusion
python main.py train_npy \
  --model_name=seq2seq_ssm \
  --seed=3409 \
  --gpu=0 \
  --no_static=false \
  --concat_static=true \
  --forcing_source=daymet
