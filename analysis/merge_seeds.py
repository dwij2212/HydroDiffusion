import numpy as np
import sys
import os

model = sys.argv[1]
npz_paths = sys.argv[2:]
out_path = sys.argv[2].replace('seed3407', 'merged').replace(
    os.path.basename(sys.argv[2]), 'predictions_merged.npz')

print(f"[INFO] Merging {len(npz_paths)} seeds for {model}")

all_preds = []
obs = None
basins = None
dates = None

for path in npz_paths:
    data = np.load(path, allow_pickle=True)
    if obs is None:
        obs = data['obs']
        basins = data['basins']
        dates = data['dates']
    key = 'preds' if 'preds' in data else 'ens'
    all_preds.append(data[key])

merged = np.mean(all_preds, axis=0)
print(f"[INFO] Merged shape: {merged.shape}")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
print(f"[INFO] Saving to {out_path}")
np.savez(out_path, basins=basins, dates=dates, obs=obs, preds=merged)
print("[INFO] Done")
