#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=eval_dssm_3410
#SBATCH --output=/users/8/zhan8460/Desktop/HydroDiffusion/logs/eval_dec_ssm_3410_kgml03_%j.txt
#SBATCH --error=/users/8/zhan8460/Desktop/HydroDiffusion/logs/eval_dec_ssm_3410_kgml03_err_%j.txt
#SBATCH --time=96:00:00
#SBATCH --partition=kgml03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
source ~/.bashrc
conda activate hydrodiff
export LD_LIBRARY_PATH=/users/8/zhan8460/anaconda3/lib:$LD_LIBRARY_PATH
cd ~/Desktop/HydroDiffusion
bash test.sh decoder_only_ssm static 0 ./runs/run_1903_1009_seed3410
