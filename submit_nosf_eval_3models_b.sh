#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=nosf_eval_b
#SBATCH --output=/users/8/zhan8460/Desktop/HydroDiffusion/logs/nosf_eval_b_%j.txt
#SBATCH --error=/users/8/zhan8460/Desktop/HydroDiffusion/logs/nosf_eval_b_err_%j.txt
#SBATCH --time=04:00:00
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G

source ~/.bashrc
conda activate hydrodiff
export LD_LIBRARY_PATH=/users/8/zhan8460/anaconda3/lib:$LD_LIBRARY_PATH
cd ~/Desktop/HydroDiffusion
git checkout no-sf-benchmark

bash test.sh decoder_only_ssm static 0 /users/8/zhan8460/Desktop/HydroDiffusion/runs/decoder_only_ssm_nosf_seed3407
bash test.sh decoder_only_lstm static 0 /users/8/zhan8460/Desktop/HydroDiffusion/runs/decoder_only_lstm_nosf_seed3407
bash test.sh diffusion_lstm static 0 /users/8/zhan8460/Desktop/HydroDiffusion/runs/diffusion_lstm_nosf_seed3407
