#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=ablation_z
#SBATCH --output=logs/ablation_z_%j.txt
#SBATCH --error=logs/ablation_z_err_%j.txt
#SBATCH --time=24:00:00
#SBATCH --partition=a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
source ~/.bashrc
conda activate hydrodiff
cd ~/Desktop/HydroDiffusion

EMB_PATH="/projects/standard/kumarv/renga/Public/DATA/camels_us_531/PREPROCESSED/embeddings/CAE_cd32_nl1_rw1.0_sw1.0_cw1.0_t0.5_run1_best_train_embeddings_mean.npy"

for (( seed=3407; seed<=3411; seed++ )); do
  echo "=== variant=z seed=$seed ==="
  python3 yimeng_test/main.py train_npy \
    --model_name=decoder_only_ssm \
    --variant=z \
    --emb_dim=32 \
    --z_emb_path="$EMB_PATH" \
    --seed="$seed" \
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
    --pool_type='power' \
    --forcing_source='daymet'
done
