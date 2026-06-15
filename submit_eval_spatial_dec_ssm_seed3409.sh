#!/bin/bash
#SBATCH --job-name=eval_spatial_3409
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_spatial_3409_%j.txt

source ~/.bashrc
conda activate hydrodiff
cd ~/Desktop/HydroDiffusion
export PYTHONPATH=$(pwd):$PYTHONPATH
export LD_LIBRARY_PATH=/users/8/zhan8460/anaconda3/lib:$LD_LIBRARY_PATH
conda activate hydrodiff

python3 main.py evaluate_npy   --model_name=decoder_only_ssm   --seed=3409   --gpu=0   --no_static=false   --concat_static=true   --run_dir=./runs/dec_ssm_spatial_seed3409   --epochs=60   --d_model=256   --d_state=256   --lr=3e-5   --lr_min=3e-6   --weight_decay=0.00   --wd=4e-5   --lr_dt=0.001   --min_dt=0.01   --max_dt=0.1   --warmup=1   --n_layer=6   --batch_size=128   --ssm_dropout=0.2   --cfi=10   --cfr=10   --pool_type=power   --predict_mode=velocity   --forcing_source=daymet   --stride=1   --basin_split_csv=/projects/standard/kumarv/shared/dwij/inverse/basin_assignment_v1.csv
