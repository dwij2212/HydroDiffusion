#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=nosf_eval
#SBATCH --output=/users/8/zhan8460/Desktop/HydroDiffusion/logs/nosf_eval_%j.txt
#SBATCH --error=/users/8/zhan8460/Desktop/HydroDiffusion/logs/nosf_eval_err_%j.txt
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

bash test.sh seq2seq_lstm static 0 /users/8/zhan8460/Desktop/HydroDiffusion/runs/seq2seq_lstm_nosf_seed3407
bash test.sh encdec_lstm static 0 /users/8/zhan8460/Desktop/HydroDiffusion/runs/encdec_lstm_nosf_seed3407
bash test.sh seq2seq_ssm static 0 /users/8/zhan8460/Desktop/HydroDiffusion/runs/seq2seq_ssm_nosf_seed3407
