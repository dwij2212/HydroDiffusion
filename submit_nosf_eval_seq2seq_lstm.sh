#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=eval_seq2seq_lstm
#SBATCH --output=/users/8/zhan8460/Desktop/HydroDiffusion/logs/eval_nosf_seq2seq_lstm_%j.txt
#SBATCH --error=/users/8/zhan8460/Desktop/HydroDiffusion/logs/eval_nosf_seq2seq_lstm_err_%j.txt
#SBATCH --time=08:00:00
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G

source ~/.bashrc
conda activate hydrodiff
export LD_LIBRARY_PATH=/users/8/zhan8460/anaconda3/lib:$LD_LIBRARY_PATH
cd ~/Desktop/HydroDiffusion
git checkout no-sf-benchmark

for seed in 3407 3408 3409 3410 3411; do
  bash test.sh seq2seq_lstm static 0 /users/8/zhan8460/Desktop/HydroDiffusion/runs/seq2seq_lstm_nosf_seed${seed}
done
