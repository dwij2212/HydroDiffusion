#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=encdec_3409
#SBATCH --output=/users/8/zhan8460/Desktop/HydroDiffusion/logs/encdec_3409_%j.txt
#SBATCH --error=/users/8/zhan8460/Desktop/HydroDiffusion/logs/encdec_3409_err_%j.txt
#SBATCH --time=96:00:00
#SBATCH --partition=kgml02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

source ~/.bashrc
conda activate hydrodiff
cd ~/Desktop/HydroDiffusion
for seed in 3409 3410 3411; do
    python main.py train_npy \
      --model_name=encdec_lstm \
      --seed=$seed \
      --gpu=0 \
      --no_static=false \
      --concat_static=true \
      --epochs=30 \
      --forcing_source=daymet
done
