#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=eval_dec_3408
#SBATCH --output=/users/8/zhan8460/Desktop/HydroDiffusion/logs/eval_dec_3408_%j.txt
#SBATCH --error=/users/8/zhan8460/Desktop/HydroDiffusion/logs/eval_dec_3408_err_%j.txt
#SBATCH --time=24:00:00
#SBATCH --partition=kgml02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

source ~/.bashrc
conda activate hydrodiff
cd ~/Desktop/HydroDiffusion
bash test.sh decoder_only_lstm static 0 ./runs/run_1603_092702_4132_seed3408
