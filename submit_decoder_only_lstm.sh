#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=dec_only_lstm
#SBATCH --output=/users/8/zhan8460/Desktop/HydroDiffusion/logs/decoder_only_lstm_%j.txt
#SBATCH --error=/users/8/zhan8460/Desktop/HydroDiffusion/logs/decoder_only_lstm_err_%j.txt
#SBATCH --time=48:00:00
#SBATCH --partition=kgml02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

source ~/.bashrc
conda activate hydrodiff
cd ~/Desktop/HydroDiffusion
bash train.sh decoder_only_lstm static cuda:0
