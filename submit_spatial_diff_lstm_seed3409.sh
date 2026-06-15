#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=spatial_dlstm_3409
#SBATCH --output=/users/8/zhan8460/Desktop/HydroDiffusion/logs/spatial_diff_lstm_3409_%j.txt
#SBATCH --error=/users/8/zhan8460/Desktop/HydroDiffusion/logs/spatial_diff_lstm_3409_err_%j.txt
#SBATCH --time=24:00:00
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
source ~/.bashrc
conda activate hydrodiff
export LD_LIBRARY_PATH=/users/8/zhan8460/anaconda3/lib:$LD_LIBRARY_PATH
cd ~/Desktop/HydroDiffusion
git checkout spatial-split
python3 main.py train_npy   --model_name=diffusion_lstm   --seed=3409   --gpu=0   --no_static=false   --concat_static=true   --forcing_source=daymet   --basin_split_csv=/projects/standard/kumarv/shared/dwij/inverse/basin_assignment_v1.csv   --run_dir=/users/8/zhan8460/Desktop/HydroDiffusion/runs/diff_lstm_spatial_seed3409
