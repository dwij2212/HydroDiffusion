#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=new_diff
#SBATCH --output=/users/8/zhan8460/Desktop/HydroDiffusion/logs/new_diff_%j.txt
#SBATCH --error=/users/8/zhan8460/Desktop/HydroDiffusion/logs/new_diff_err_%j.txt
#SBATCH --time=96:00:00
#SBATCH --partition=kgml03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

source ~/.bashrc
conda activate hydrodiff
cd ~/Desktop/HydroDiffusion
bash train.sh diffusion_lstm static cuda:0
