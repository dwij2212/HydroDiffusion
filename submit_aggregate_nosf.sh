#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=aggregate_nosf
#SBATCH --output=/users/8/zhan8460/Desktop/HydroDiffusion/logs/aggregate_nosf_%j.txt
#SBATCH --error=/users/8/zhan8460/Desktop/HydroDiffusion/logs/aggregate_nosf_err_%j.txt
#SBATCH --time=04:00:00
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G

source ~/.bashrc
conda activate hydrodiff
export LD_LIBRARY_PATH=/users/8/zhan8460/anaconda3/lib:$LD_LIBRARY_PATH
cd ~/Desktop/HydroDiffusion
git checkout no-sf-benchmark

python3 aggregate_nosf.py
