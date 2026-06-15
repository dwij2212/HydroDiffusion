#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=diffusion_lstm
#SBATCH --output=/users/8/zhan8460/Desktop/HydroDiffusion/logs/diffusion_lstm_%j.txt
#SBATCH --error=/users/8/zhan8460/Desktop/HydroDiffusion/logs/diffusion_lstm_err_%j.txt
#SBATCH --time=24:00:00
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

source ~/.bashrc
conda activate hydrodiff
cd ~/Desktop/HydroDiffusion
bash train.sh diffusion_lstm static cuda:0
