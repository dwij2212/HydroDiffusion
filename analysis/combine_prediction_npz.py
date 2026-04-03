import argparse
from pathlib import Path

import numpy as np


def _load_ensemble_cube(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    if "ens" in npz.files:
        arr = npz["ens"]
        if arr.ndim != 3:
            raise ValueError(f"Expected ens to be 3D (N,S,H), got shape {arr.shape}")
        return arr
    if "preds" in npz.files:
        arr = npz["preds"]
        if arr.ndim != 2:
            raise ValueError(f"Expected preds to be 2D (N,H), got shape {arr.shape}")
        return arr[:, None, :]
    raise ValueError(f"Missing ensemble data key in npz. Found keys: {npz.files}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge prediction NPZ files into one ensemble NPZ.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input predictions.npz files")
    parser.add_argument("--output", required=True, help="Output merged npz file path")
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Skip files whose basins/dates/obs do not exactly match the reference",
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    ref_path = input_paths[0]
    ref = np.load(ref_path, allow_pickle=True)
    required = ["basins", "dates", "obs"]
    for k in required:
        if k not in ref.files:
            raise ValueError(f"Reference file missing key '{k}': {ref_path}")

    basins = ref["basins"]
    dates = ref["dates"]
    obs = ref["obs"]
    ref_ens = _load_ensemble_cube(ref)
    n_rows, _, horizon = ref_ens.shape

    kept = [ref_path]
    members = [ref_ens]
    skipped = []

    for p in input_paths[1:]:
        cur = np.load(p, allow_pickle=True)
        for k in required:
            if k not in cur.files:
                msg = f"{p} missing key '{k}'"
                if args.allow_mismatch:
                    skipped.append((p, msg))
                    cur = None
                    break
                raise ValueError(msg)
        if cur is None:
            continue

        cur_ens = _load_ensemble_cube(cur)

        mismatch_reasons = []
        if cur_ens.shape[0] != n_rows:
            mismatch_reasons.append(f"N mismatch: {cur_ens.shape[0]} vs {n_rows}")
        if cur_ens.shape[2] != horizon:
            mismatch_reasons.append(f"H mismatch: {cur_ens.shape[2]} vs {horizon}")
        if not np.array_equal(cur["basins"], basins):
            mismatch_reasons.append("basins mismatch")
        if not np.array_equal(cur["dates"], dates):
            mismatch_reasons.append("dates mismatch")
        if not np.array_equal(cur["obs"], obs):
            mismatch_reasons.append("obs mismatch")

        if mismatch_reasons:
            msg = "; ".join(mismatch_reasons)
            if args.allow_mismatch:
                skipped.append((p, msg))
                continue
            raise ValueError(f"Cannot merge {p}: {msg}")

        kept.append(p)
        members.append(cur_ens)

    merged = np.concatenate(members, axis=1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, basins=basins, dates=dates, obs=obs, ens=merged)

    print("Merged file written:", output_path)
    print("Merged shape (N,S,H):", merged.shape)
    print("Kept files:")
    for p in kept:
        print("  -", p)

    if skipped:
        print("Skipped files:")
        for p, reason in skipped:
            print(f"  - {p}: {reason}")


if __name__ == "__main__":
    main()
