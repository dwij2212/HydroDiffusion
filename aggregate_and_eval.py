import numpy as np
import subprocess
import sys
import os

def aggregate_seeds(run_dirs, output_path):
    preds_list = []
    ref = None
    for r in run_dirs:
        d = np.load(f"{r}/predictions.npz", allow_pickle=True)
        if ref is None:
            ref = d
        preds_list.append(d["preds"].astype(np.float32))
    
    avg_preds = np.mean(preds_list, axis=0).astype(np.float32)
    
    np.savez(
        output_path,
        basins=ref["basins"],
        dates=ref["dates"],
        obs=ref["obs"],
        preds=avg_preds
    )
    print(f"[INFO] Aggregated predictions saved to {output_path}")
    print(f"[INFO] preds shape: {avg_preds.shape}, mean: {avg_preds.mean():.4f}")

if __name__ == "__main__":
    # seq2seq_lstm
    lstm_runs = [
        "runs/run_1803_1309_seed3407",
        "runs/run_1703_1651_seed3408",
        "runs/run_1703_2021_seed3409",
        "runs/run_1803_0033_seed3410",
        "runs/run_1803_0413_seed3411",
    ]
    os.makedirs("runs/aggregated", exist_ok=True)
    aggregate_seeds(lstm_runs, "runs/aggregated/seq2seq_lstm_aggregated.npz")
    
    # run evaluation
    subprocess.run([
        "python3", "analysis/main_performance_full_evaluation.py",
        "seq2seq_lstm_aggregated",
        "runs/aggregated/seq2seq_lstm_aggregated.npz"
    ])
