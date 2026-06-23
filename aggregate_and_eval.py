import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


RUN_RE = re.compile(r"^run_\d+_\d+_(?P<model>.+)_seed(?P<seed>\d+)$")


def discover_completed_runs(runs_root):
    completed = defaultdict(dict)
    skipped = []

    for pred_path in sorted(runs_root.glob("run_*/predictions.npz")):
        run_dir = pred_path.parent
        match = RUN_RE.match(run_dir.name)
        if not match:
            skipped.append(run_dir)
            continue

        model = match.group("model")
        seed = int(match.group("seed"))
        if seed in completed[model]:
            raise ValueError(
                f"Found multiple evaluated runs for {model} seed {seed}: "
                f"{completed[model][seed]} and {run_dir}"
            )
        completed[model][seed] = run_dir

    return completed, skipped


def aggregate_seeds(run_dirs, output_path):
    pred_key = None
    sum_preds = None
    ens_memmap = None
    ens_tmp_path = output_path.with_suffix(".ens.tmp.npy")
    ens_offset = 0
    ref = None
    n_runs = 0
    pred_sum = 0.0
    pred_count = 0

    for run_dir in run_dirs:
        data = np.load(run_dir / "predictions.npz", allow_pickle=True)
        if ref is None:
            ref = data

        current_key = "preds" if "preds" in data.files else "ens" if "ens" in data.files else None
        if current_key is None:
            raise KeyError(f"{run_dir}/predictions.npz has neither 'preds' nor 'ens'")
        if pred_key is None:
            pred_key = current_key
        elif current_key != pred_key:
            raise ValueError(f"Mixed prediction keys found: {pred_key} and {current_key}")

        current_preds = data[pred_key].astype(np.float32)
        pred_sum += float(current_preds.sum(dtype=np.float64))
        pred_count += current_preds.size
        if pred_key == "ens":
            if current_preds.ndim != 3:
                raise ValueError(f"Expected ens to be 3D for {run_dir}, got {current_preds.shape}")

            if ens_memmap is None:
                total_members = current_preds.shape[1] * len(run_dirs)
                full_shape = (current_preds.shape[0], total_members, current_preds.shape[2])
                ens_memmap = np.lib.format.open_memmap(
                    ens_tmp_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=full_shape,
                )
            elif (
                current_preds.shape[0] != ens_memmap.shape[0]
                or current_preds.shape[2] != ens_memmap.shape[2]
            ):
                raise ValueError(
                    f"Shape mismatch for {run_dir}: {current_preds.shape} is incompatible "
                    f"with {ens_memmap.shape}"
                )

            next_offset = ens_offset + current_preds.shape[1]
            ens_memmap[:, ens_offset:next_offset, :] = current_preds
            ens_offset = next_offset
        else:
            if sum_preds is None:
                sum_preds = current_preds
            else:
                if current_preds.shape != sum_preds.shape:
                    raise ValueError(
                        f"Shape mismatch for {run_dir}: {current_preds.shape} != {sum_preds.shape}"
                    )
                sum_preds += current_preds
        n_runs += 1

    if pred_key == "ens":
        ens_memmap.flush()
        pred_array = ens_memmap
    else:
        pred_array = (sum_preds / n_runs).astype(np.float32)

    np.savez(
        output_path,
        basins=ref["basins"],
        dates=ref["dates"],
        obs=ref["obs"],
        **{pred_key: pred_array},
    )
    print(f"[INFO] Aggregated predictions saved to {output_path}", flush=True)
    print(f"[INFO] {pred_key} shape: {pred_array.shape}, mean: {pred_sum / pred_count:.4f}", flush=True)

    if ens_memmap is not None:
        del ens_memmap
        ens_tmp_path.unlink(missing_ok=True)


def run_evaluation(experiment, npz_path):
    cmd = [
        sys.executable,
        "analysis/main_performance_full_evaluation.py",
        experiment,
        str(npz_path),
    ]
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate completed seed predictions and run full evaluation."
    )
    parser.add_argument("--runs-root", default="runs", type=Path)
    parser.add_argument("--output-dir", default=Path("runs/aggregated"), type=Path)
    parser.add_argument("--required-seeds", default=5, type=int)
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="Aggregate/evaluate model groups even if fewer than required-seeds are complete.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional model group names to process, e.g. decoder_only_lstm_nosf.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    completed, skipped = discover_completed_runs(args.runs_root)
    if skipped:
        print("[WARN] Ignoring predictions in run dirs with unexpected names:", flush=True)
        for run_dir in skipped:
            print(f"  - {run_dir}", flush=True)

    selected = []
    for model, by_seed in sorted(completed.items()):
        if args.models and model not in args.models:
            continue

        seeds = sorted(by_seed)
        if not args.include_partial and len(seeds) < args.required_seeds:
            print(
                f"[SKIP] {model}: {len(seeds)}/{args.required_seeds} evaluated seeds "
                f"({', '.join(map(str, seeds))})",
                flush=True,
            )
            continue

        selected.append((model, seeds, [by_seed[seed] for seed in seeds]))

    if not selected:
        print("[INFO] No model groups selected for aggregation/evaluation.", flush=True)
        return

    for model, seeds, run_dirs in selected:
        experiment = f"{model}_aggregated"
        output_path = args.output_dir / f"{experiment}.npz"
        print(f"[INFO] Processing {model} with seeds: {', '.join(map(str, seeds))}", flush=True)
        aggregate_seeds(run_dirs, output_path)
        run_evaluation(experiment, output_path)


if __name__ == "__main__":
    main()
