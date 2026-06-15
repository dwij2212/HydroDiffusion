#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=new_dec
#SBATCH --output=/users/8/zhan8460/Desktop/HydroDiffusion/logs/new_dec_%j.txt
#SBATCH --error=/users/8/zhan8460/Desktop/HydroDiffusion/logs/new_dec_err_%j.txt
#SBATCH --time=96:00:00
#SBATCH --partition=kgml02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

source ~/.bashrc
conda activate hydrodiff
cd ~/Desktop/HydroDiffusion
bash train.sh decoder_only_lstm static cuda:0
