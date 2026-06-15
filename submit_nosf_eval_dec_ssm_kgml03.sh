#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=eval_dec_ssm
#SBATCH --output=/users/8/zhan8460/Desktop/HydroDiffusion/logs/eval_dec_ssm_kgml03_%j.txt
#SBATCH --error=/users/8/zhan8460/Desktop/HydroDiffusion/logs/eval_dec_ssm_kgml03_err_%j.txt
#SBATCH --time=2-00:00:00
#SBATCH --partition=kgml03
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G

source ~/.bashrc
conda activate hydrodiff
export LD_LIBRARY_PATH=/users/8/zhan8460/anaconda3/lib:$LD_LIBRARY_PATH
cd ~/Desktop/HydroDiffusion
git checkout no-sf-benchmark

bash test.sh decoder_only_ssm static 0 /users/8/zhan8460/Desktop/HydroDiffusion/runs/decoder_only_ssm_nosf_seed3410
bash test.sh decoder_only_ssm static 0 /users/8/zhan8460/Desktop/HydroDiffusion/runs/decoder_only_ssm_nosf_seed3411
