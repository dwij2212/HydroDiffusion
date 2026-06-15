#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=ablation_ssm
#SBATCH --output=logs/ablation_ssm_%j.txt
#SBATCH --error=logs/ablation_ssm_err_%j.txt
#SBATCH --time=48:00:00
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
source ~/.bashrc
conda activate hydrodiff
cd ~/Desktop/HydroDiffusion

EMB_PATH="/projects/standard/kumarv/renga/Public/DATA/camels_us_531/PREPROCESSED/embeddings/CAE_cd32_nl1_rw1.0_sw1.0_cw1.0_t0.5_run1_best_train_embeddings_mean.npy"

nseeds=5
firstseed=3407

for variant in z "z+static"; do
  for (( seed=firstseed; seed<firstseed+nseeds; seed++ )); do
    echo "=== variant=$variant seed=$seed ==="
    python3 yimeng_test/main.py train_npy \
      --model_name=decoder_only_ssm \
      --variant="$variant" \
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
done
