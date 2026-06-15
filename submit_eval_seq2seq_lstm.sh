#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=eval_seq2seq_lstm
#SBATCH --output=/users/8/zhan8460/Desktop/HydroDiffusion/logs/eval_seq2seq_lstm_%j.txt
#SBATCH --error=/users/8/zhan8460/Desktop/HydroDiffusion/logs/eval_seq2seq_lstm_err_%j.txt
#SBATCH --time=04:00:00
#SBATCH --partition=kgml02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
source ~/.bashrc
conda activate hydrodiff
cd ~/Desktop/HydroDiffusion
bash test.sh seq2seq_lstm static 0 ./runs/run_1803_1309_seed3407
bash test.sh seq2seq_lstm static 0 ./runs/run_1703_1651_seed3408
bash test.sh seq2seq_lstm static 0 ./runs/run_1703_2021_seed3409
bash test.sh seq2seq_lstm static 0 ./runs/run_1803_0033_seed3410
bash test.sh seq2seq_lstm static 0 ./runs/run_1803_0413_seed3411
