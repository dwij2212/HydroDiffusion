#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=decoder_only_lstm_3407
#SBATCH --output=/users/8/zhan8460/Desktop/HydroDiffusion/logs/decoder_only_lstm_3407_%j.txt
#SBATCH --error=/users/8/zhan8460/Desktop/HydroDiffusion/logs/decoder_only_lstm_3407_err_%j.txt
#SBATCH --time=24:00:00
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

source ~/.bashrc
conda activate hydrodiff
cd ~/Desktop/HydroDiffusion
python main.py train_npy \
  --model_name=decoder_only_lstm \
  --seed=3407 \
  --gpu=0 \
  --no_static=false \
  --concat_static=true \
  --forcing_source=daymet
