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
    print(f"[INFO] Saved to {output_path}, shape: {concat_ens.shape}")

os.makedirs("/projects/standard/kumarv/zhan8460/HydroDiffusion_aggregated_spatial", exist_ok=True)

seeds = [3407, 3408, 3409, 3410, 3411]
run_dirs = [f"runs/diff_lstm_spatial_seed{seed}" for seed in seeds]
output_path = "/projects/standard/kumarv/zhan8460/HydroDiffusion_aggregated_spatial/diff_lstm_spatial_aggregated.npz"

print("=== Aggregating diffusion_lstm spatial split ===")
aggregate_seeds(run_dirs, output_path)

subprocess.run([
    "python3", "analysis/main_performance_full_evaluation.py",
    "diff_lstm_spatial_aggregated",
    output_path
])
