#!/bin/bash
#SBATCH --job-name=eval_sp_dlstm_3409
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_spatial_diff_lstm_3409_%j.txt

source ~/.bashrc
conda activate hydrodiff
cd ~/Desktop/HydroDiffusion
export PYTHONPATH=$(pwd):$PYTHONPATH
export LD_LIBRARY_PATH=/users/8/zhan8460/anaconda3/lib:$LD_LIBRARY_PATH
conda activate hydrodiff

python3 main.py evaluate_npy   --model_name=diffusion_lstm   --seed=3409   --gpu=0   --no_static=false   --concat_static=true   --run_dir=./runs/diff_lstm_spatial_seed3409   --forcing_source=daymet   --stride=1   --basin_split_csv=/projects/standard/kumarv/shared/dwij/inverse/basin_assignment_v1.csv
