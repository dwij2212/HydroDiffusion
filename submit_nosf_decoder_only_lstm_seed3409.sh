#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=nosf_dec_lstm_3409
#SBATCH --output=/users/8/zhan8460/Desktop/HydroDiffusion/logs/nosf_dec_lstm_3409_%j.txt
#SBATCH --error=/users/8/zhan8460/Desktop/HydroDiffusion/logs/nosf_dec_lstm_3409_err_%j.txt
#SBATCH --time=12:00:00
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --open-mode=append

source ~/.bashrc
conda activate hydrodiff
export LD_LIBRARY_PATH=/users/8/zhan8460/anaconda3/lib:$LD_LIBRARY_PATH
cd ~/Desktop/HydroDiffusion
git checkout no-sf-benchmark

python3 main.py train_npy   --model_name=decoder_only_lstm   --seed=3409   --gpu=0   --no_static=false   --concat_static=true   --epochs=60   --lr=3e-5   --predict_mode=velocity   --forcing_source=daymet   --run_dir=/users/8/zhan8460/Desktop/HydroDiffusion/runs/decoder_only_lstm_nosf_seed3409
