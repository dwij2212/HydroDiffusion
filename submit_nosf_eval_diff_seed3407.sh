#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=eval_diff3407
#SBATCH --output=/users/8/zhan8460/Desktop/HydroDiffusion/logs/eval_diff_seed3407_%j.txt
#SBATCH --error=/users/8/zhan8460/Desktop/HydroDiffusion/logs/eval_diff_seed3407_err_%j.txt
#SBATCH --time=24:00:00
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G

source ~/.bashrc
conda activate hydrodiff
export LD_LIBRARY_PATH=/users/8/zhan8460/anaconda3/lib:$LD_LIBRARY_PATH
cd ~/Desktop/HydroDiffusion
git checkout no-sf-benchmark

bash test.sh diffusion_lstm static 0 /users/8/zhan8460/Desktop/HydroDiffusion/runs/diffusion_lstm_nosf_seed3407
