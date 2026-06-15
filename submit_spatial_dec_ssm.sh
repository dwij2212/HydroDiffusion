#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=spatial_dec_ssm
#SBATCH --output=/users/8/zhan8460/Desktop/HydroDiffusion/logs/spatial_dec_ssm_%j.txt
#SBATCH --error=/users/8/zhan8460/Desktop/HydroDiffusion/logs/spatial_dec_ssm_err_%j.txt
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

for seed in 3407 3408 3409 3410 3411; do
python3 main.py train_npy \
  --model_name=decoder_only_ssm \
  --seed=${seed} \
  --gpu=0 \
  --no_static=false \
  --concat_static=true \
  --epochs=60 \
  --d_model=256 \
  --d_state=256 \
  --lr=3e-5 \
  --lr_min=3e-6 \
  --weight_decay=0.00 \
  --wd=4e-5 \
  --lr_dt=0.001 \
  --min_dt=0.01 \
  --max_dt=0.1 \
  --warmup=1 \
  --n_layers=6 \
  --ssm_dropout=0.2 \
  --cfi=10 \
  --cfr=10 \
  --pool_type=power \
  --forcing_source=daymet \
  --basin_split_csv=/projects/standard/kumarv/shared/dwij/inverse/basin_assignment_v1.csv \
  --run_dir=/users/8/zhan8460/Desktop/HydroDiffusion/runs/dec_ssm_spatial_seed${seed}
done
