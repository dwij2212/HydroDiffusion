#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=eval_dssm_3411
#SBATCH --output=/users/8/zhan8460/Desktop/HydroDiffusion/logs/eval_dec_ssm_3411_msigpu_%j.txt
#SBATCH --error=/users/8/zhan8460/Desktop/HydroDiffusion/logs/eval_dec_ssm_3411_msigpu_err_%j.txt
#SBATCH --time=24:00:00
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
source ~/.bashrc
conda activate hydrodiff
export LD_LIBRARY_PATH=/users/8/zhan8460/anaconda3/lib:$LD_LIBRARY_PATH
cd ~/Desktop/HydroDiffusion
bash test.sh decoder_only_ssm static 0 ./runs/run_2103_0238_seed3411
