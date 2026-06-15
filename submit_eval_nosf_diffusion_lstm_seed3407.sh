#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=eval_nosf_diffusion_lstm_3407
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=/users/8/zhan8460/Desktop/HydroDiffusion/logs/eval_nosf_diffusion_lstm_3407_%j.txt

source /users/8/zhan8460/anaconda3/bin/activate
conda activate hydrodiff
export LD_LIBRARY_PATH=/users/8/zhan8460/anaconda3/lib:$LD_LIBRARY_PATH
cd ~/Desktop/HydroDiffusion

# Make a private copy of test.sh for this job to avoid race conditions
JOB_TEST_SH=/tmp/test_eval_nosf_diffusion_lstm_3407_$$.sh
cp test.sh $JOB_TEST_SH
sed -i "s/^firstseed=.*/firstseed=3407/" $JOB_TEST_SH

bash $JOB_TEST_SH diffusion_lstm static 0 /users/8/zhan8460/Desktop/HydroDiffusion/runs/diffusion_lstm_nosf_seed3407

rm -f $JOB_TEST_SH
