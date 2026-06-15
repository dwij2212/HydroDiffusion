#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=agg_nosf_rem
#SBATCH --output=/users/8/zhan8460/Desktop/HydroDiffusion/logs/aggregate_nosf_remaining_%j.txt
#SBATCH --error=/users/8/zhan8460/Desktop/HydroDiffusion/logs/aggregate_nosf_remaining_err_%j.txt
#SBATCH --time=08:00:00
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G

source ~/.bashrc
conda activate hydrodiff
export LD_LIBRARY_PATH=/users/8/zhan8460/anaconda3/lib:$LD_LIBRARY_PATH
cd ~/Desktop/HydroDiffusion
git checkout no-sf-benchmark

python3 - << 'PYEOF'
import numpy as np
import subprocess
import os

def aggregate_seeds(run_dirs, output_path):
    ens_list = []
    ref = None
    for r in run_dirs:
        d = np.load(f"{r}/predictions.npz", allow_pickle=True)
        if ref is None:
            ref = d
        ens_list.append(d["ens"].astype(np.float32))
    concat_ens = np.concatenate(ens_list, axis=1).astype(np.float32)
    np.savez(output_path,
        basins=ref["basins"],
        dates=ref["dates"],
        obs=ref["obs"],
        preds=concat_ens)
    print(f"[INFO] Saved ensemble aggregate to {output_path}, shape: {concat_ens.shape}")

seeds = [3407, 3408, 3409, 3410, 3411]
out_dir = "/projects/standard/kumarv/zhan8460/HydroDiffusion_aggregated_nosf"

for model in ["decoder_only_lstm", "diffusion_lstm"]:
    run_dirs = [f"runs/{model}_nosf_seed{seed}" for seed in seeds]
    output_path = f"{out_dir}/{model}_nosf_aggregated.npz"
    print(f"\n=== Aggregating {model} ===")
    aggregate_seeds(run_dirs, output_path)
    subprocess.run([
        "python3", "analysis/main_performance_full_evaluation.py",
        f"{model}_nosf_aggregated",
        output_path
    ])
PYEOF
