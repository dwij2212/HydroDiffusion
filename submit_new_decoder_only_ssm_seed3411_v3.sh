#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=new_dssm_3411
#SBATCH --output=/users/8/zhan8460/Desktop/HydroDiffusion/logs/new_decoder_only_ssm_3411_v3_%j.txt
#SBATCH --error=/users/8/zhan8460/Desktop/HydroDiffusion/logs/new_decoder_only_ssm_3411_v3_err_%j.txt
#SBATCH --time=96:00:00
#SBATCH --partition=kgml03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
module load cuda/11.8.0-gcc-7.2.0-xqzqlf2
source ~/.bashrc
conda activate hydrodiff
export LD_LIBRARY_PATH=/users/8/zhan8460/anaconda3/lib:$LD_LIBRARY_PATH
cd ~/Desktop/HydroDiffusion
python3 main.py train_npy \
  --model_name=decoder_only_ssm \
  --seed=3411 \
  --gpu=0 \
  --no_static=False \
  --concat_static=True \
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
  --pool_type='power' \
  --forcing_source='daymet'
