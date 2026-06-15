#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=nosf_seq2seq_ssm_3409
#SBATCH --output=/users/8/zhan8460/Desktop/HydroDiffusion/logs/nosf_seq2seq_ssm_3409_%j.txt
#SBATCH --error=/users/8/zhan8460/Desktop/HydroDiffusion/logs/nosf_seq2seq_ssm_3409_err_%j.txt
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

python3 main.py train_npy   --model_name=seq2seq_ssm   --seed=3409   --gpu=0   --no_static=false   --concat_static=true   --epochs=50   --d_model=128   --d_state=128   --lr=4e-4   --lr_min=4e-5   --weight_decay=3e-2   --wd=2e-2   --lr_dt=0.001   --min_dt=0.01   --max_dt=0.1   --warmup=0   --n_layers=6   --ssm_dropout=0.12   --cfi=10   --cfr=10   --forcing_source=daymet   --run_dir=/users/8/zhan8460/Desktop/HydroDiffusion/runs/seq2seq_ssm_nosf_seed3409
