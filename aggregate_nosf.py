import numpy as np
import subprocess
import os

def aggregate_seeds(run_dirs, output_path):
    preds_list = []
    ens_list = []
    ref = None
    is_ensemble = None

    for r in run_dirs:
        d = np.load(f"{r}/predictions.npz", allow_pickle=True)
        if ref is None:
            ref = d
            is_ensemble = "ens" in d

        if is_ensemble:
            ens_list.append(d["ens"].astype(np.float32))
        else:
            preds_list.append(d["preds"].astype(np.float32))

    if is_ensemble:
        # concat 5 seeds x 50 samples = 250 samples along axis=1
        concat_ens = np.concatenate(ens_list, axis=1).astype(np.float32)
        np.savez(output_path,
            basins=ref["basins"],
            dates=ref["dates"],
            obs=ref["obs"],
            preds=concat_ens)
        print(f"[INFO] Saved ensemble aggregate to {output_path}, shape: {concat_ens.shape}")
    else:
        avg_preds = np.mean(preds_list, axis=0).astype(np.float32)
        np.savez(output_path,
            basins=ref["basins"],
            dates=ref["dates"],
            obs=ref["obs"],
            preds=avg_preds)
        print(f"[INFO] Saved deterministic aggregate to {output_path}, shape: {avg_preds.shape}")

os.makedirs("/projects/standard/kumarv/zhan8460/HydroDiffusion_aggregated_nosf", exist_ok=True)

models = [
    "seq2seq_lstm",
    "encdec_lstm",
    "seq2seq_ssm",
    "decoder_only_ssm",
    "decoder_only_lstm",
    "diffusion_lstm",
]

seeds = [3407, 3408, 3409, 3410, 3411]

for model in models:
    run_dirs = [f"runs/{model}_nosf_seed{seed}" for seed in seeds]
    output_path = f"/projects/standard/kumarv/zhan8460/HydroDiffusion_aggregated_nosf/{model}_nosf_aggregated.npz"
    print(f"\n=== Aggregating {model} ===")
    aggregate_seeds(run_dirs, output_path)
    subprocess.run([
        "python3", "analysis/main_performance_full_evaluation.py",
        f"{model}_nosf_aggregated",
        output_path
    ])
