"""
papercode/datasets_npy.py

Drop-in replacement for CamelsH5 that reads from a single .npy array
with shape [num_basins, num_days, 33]:
    - cols  0-26 : 27 static catchment attributes  (constant across time)
    - cols 27-31 : 5 dynamic forcing features       (PRCP, SRAD, Tmax, Tmin, Vp)
    - col  32    : streamflow (SF)

Normalization is computed at runtime from the **training** data passed in.
Streamflow is always normalized as log1p(streamflow) plus a global z-score.
The __getitem__ signature matches CamelsH5 exactly, so the existing
training / evaluation loops work without modification.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ------------------------------------------------------------------ #
#  Module-level helper: build everything once, share across splits    #
# ------------------------------------------------------------------ #

def load_npy_data(
    npy_path: str,
    dates_path: str,
    basin_list_path: str,
    sentinel: float = -999.0,
    remove_leap: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the raw .npy cube, dates array, and basin list.

    Returns
    -------
    data   : np.ndarray  (B, T, 33)  with sentinels replaced by NaN
    dates  : np.ndarray  (T,)        datetime.date objects, leap days removed
    basins : np.ndarray  (B,)        8-digit USGS gauge-ID strings
    """
    data = np.load(npy_path).astype(np.float32)
    dates = np.load(dates_path, allow_pickle=True)
    basins = np.load(basin_list_path, allow_pickle=True)

    # sentinel → NaN
    data[data == sentinel] = np.nan

    # optionally strip Feb-29 rows so every year has 365 days
    if remove_leap:
        date_df = pd.to_datetime(dates)
        leap_mask = (date_df.month == 2) & (date_df.day == 29)
        keep = ~leap_mask
        data = data[:, keep, :]
        dates = dates[keep]

    return data, dates, basins


def compute_normalization(
    data: np.ndarray,
    dates: np.ndarray,
    train_start: str = "1980-10-01",
    train_end: str = "1990-09-30",
) -> Dict[str, np.ndarray]:
    """
    Compute global mean / std of forcings and static attributes over the training period
    (all basins × training days).  Called once, result shared by all splits.

    Returns a dict with the same keys as the repo's SCALAR convention:
        input_means  (5,)
        input_stds   (5,)
        static_means (27,)
        static_stds  (27,)
    """
    date_series = pd.to_datetime(dates)
    mask = (date_series >= train_start) & (date_series <= train_end)
    train_idx = np.where(mask)[0]

    forcing = data[:, train_idx, 27:32]          # (B, T_train, 5)
    static  = data[:, 0, :27]                    # (B, 27) — constant catchment attrs

    f_flat = forcing.reshape(-1, 5)

    scalar = {
        "input_means":  np.nanmean(f_flat, axis=0),   # (5,)
        "input_stds":   np.nanstd(f_flat, axis=0),    # (5,)
        "static_means": np.nanmean(static, axis=0),   # (27,)
        "static_stds":  np.nanstd(static, axis=0),    # (27,)
    }
    # guard against zero std
    scalar["input_stds"][scalar["input_stds"] == 0] = 1.0
    scalar["static_stds"][scalar["static_stds"] == 0] = 1.0

    return scalar


def compute_global_log_stats(
    data: np.ndarray,
    dates: np.ndarray,
    train_start: str = "1980-10-01",
    train_end: str = "1990-09-30",
) -> Tuple[float, float]:
    """
    Compute global mean/std of log1p(streamflow) over train basins and dates.
    This is the ungauged-basin streamflow normalization.
    """
    date_series = pd.to_datetime(dates)
    mask = (date_series >= train_start) & (date_series <= train_end)
    train_idx = np.where(mask)[0]

    sf = data[:, train_idx, 32]
    log_sf = np.log1p(np.clip(sf, 0, None))

    global_log_mean = float(np.nanmean(log_sf))
    global_log_std = float(np.nanstd(log_sf))
    if global_log_std == 0:
        global_log_std = 1.0

    return global_log_mean, global_log_std


def load_basin_assignment_masks(
    basin_assignment_csv: str,
    basins: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Load train/test basin labels from CSV and return boolean masks."""
    csv_path = Path(basin_assignment_csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Basin assignment CSV not found: {csv_path}")

    assignment_df = pd.read_csv(
        csv_path,
        dtype={"Basin_ID": str, "Label": str},
        usecols=["Basin_ID", "Label"],
    )
    assignment_df["Basin_ID"] = assignment_df["Basin_ID"].fillna("").astype(str).str.strip()
    assignment_df["Label"] = (
        assignment_df["Label"].fillna("").astype(str).str.strip().str.lower()
    )

    if (assignment_df["Basin_ID"] == "").any():
        raise ValueError(f"Empty Basin_ID found in basin assignment CSV: {csv_path}")
    assignment_df["Basin_ID"] = assignment_df["Basin_ID"].str.zfill(8)

    invalid_labels = assignment_df.loc[
        ~assignment_df["Label"].isin(["train", "test"]), "Label"
    ].unique()
    if len(invalid_labels) > 0:
        raise ValueError(
            "Basin assignment CSV only supports 'train' and 'test' labels; "
            f"found {sorted(invalid_labels.tolist())}"
        )

    duplicate_ids = assignment_df.loc[
        assignment_df["Basin_ID"].duplicated(keep=False), "Basin_ID"
    ].unique()
    if len(duplicate_ids) > 0:
        preview = ", ".join(duplicate_ids[:10])
        raise ValueError(f"Duplicate Basin_ID entries found in basin assignment CSV: {preview}")

    basin_labels = assignment_df.set_index("Basin_ID")["Label"]
    basin_ids = pd.Series([str(b).strip().zfill(8) for b in basins])
    matched_labels = basin_ids.map(basin_labels)

    train_mask = matched_labels.eq("train").to_numpy()
    test_mask = matched_labels.eq("test").to_numpy()
    if not train_mask.any() or not test_mask.any():
        raise ValueError(
            "Basin assignment CSV did not match at least one train basin and one test basin"
        )

    print(f"Using basin assignment CSV: {csv_path}")
    print(f"  Basin subsets: train={int(train_mask.sum())}, test={int(test_mask.sum())}")

    return {"train": train_mask, "test": test_mask}


# ------------------------------------------------------------------ #
#  The Dataset                                                        #
# ------------------------------------------------------------------ #

class CamelsNPY(Dataset):
    """
    Drop-in replacement for ``CamelsH5``.

    Parameters
    ----------
    data : np.ndarray
        Full cube (B, T_total, 33) with NaN for missing values.
    dates : np.ndarray
        Date array (T_total,) matching ``data`` axis-1.
    basins : np.ndarray
        Basin-ID strings (B,) matching ``data`` axis-0.
    scalar : dict
        Normalization statistics (from ``compute_normalization``).
    global_log_mean, global_log_std : float
        Global log1p(streamflow) normalization statistics.
    split_start, split_end : str
        Date strings for the current split (e.g. '1980-10-01', '1990-09-30').
    seq_length : int
        Number of past days the model sees (default 365).
    forecast_horizon : int
        Number of future days to predict (default 8).
    stride : int
        Step between consecutive window starts.  stride=1 matches the
        original repo (maximum overlap).  Use stride=90 or stride=30
        for faster debugging runs.
    concat_static : bool
        If True the training loop will concatenate static attrs to x itself.
        The dataset always returns attrs separately; concat happens in the loop.
    no_static : bool
        If True, static attributes are *not* returned.
    include_dates : bool
        Whether to include the date string per sample (needed for eval).
    is_train : bool
        If True, windows with NaN / negative SF targets are dropped.
    """

    def __init__(
        self,
        data: np.ndarray,
        dates: np.ndarray,
        basins: np.ndarray,
        scalar: Dict[str, np.ndarray],
        global_log_mean: float,
        global_log_std: float,
        split_start: str,
        split_end: str,
        seq_length: int = 365,
        forecast_horizon: int = 8,
        stride: int = 1,
        concat_static: bool = False,
        no_static: bool = False,
        include_dates: bool = False,
        is_train: bool = True,
        model_name: str = "lstm",
        **kwargs,            # absorb unused keys so callers don't need to filter
    ):
        super().__init__()
        self.scalar = scalar
        self.no_static = no_static
        self.concat_static = concat_static
        self.include_dates = include_dates
        self.seq_length = int(seq_length)
        self.forecast_horizon = int(forecast_horizon)
        self.total_len = self.seq_length + self.forecast_horizon
        self.global_log_mean = float(global_log_mean)
        self.global_log_std = float(global_log_std)

        # ---- date bounds for this split ----
        date_series = pd.to_datetime(dates)
        mask = (date_series >= split_start) & (date_series <= split_end)
        split_idx = np.where(mask)[0]
        if split_idx.size == 0:
            raise ValueError(f"No dates found in split [{split_start}, {split_end}]")
        self.dates_all = dates
        split_start_idx = int(split_idx[0])
        split_end_idx = int(split_idx[-1])

        # ---- normalize static attrs (z-score across basins) ----
        static_raw = data[:, 0, :27].copy()                        # (B, 27)
        self.static_normed = (
            (static_raw - scalar["static_means"]) / scalar["static_stds"]
        ).astype(np.float32)                                        # (B, 27)

        # ---- normalize forcing globally over the full timeline ----
        self.forcing_all = data[:, :, 27:32].copy()                 # (B, T_all, 5)
        self.forcing_all = (
            (self.forcing_all - scalar["input_means"]) / scalar["input_stds"]
        ).astype(np.float32)

        self.sf_all = data[:, :, 32].copy()                         # (B, T_all)
        sf_log = np.log1p(np.clip(self.sf_all, 0, None)).astype(np.float32)
        self.sf_all = (
            (sf_log - self.global_log_mean) / self.global_log_std
        ).astype(np.float32)

        # Expand 5 forcing variables to 15 virtual channels lazily in __getitem__.
        # Final channel order matches the existing code expectations:
        # [prcp_nldas, prcp_maurer, prcp_daymet, srad_nldas, ...]
        self._forcing_15_idx = np.array(
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            dtype=np.int64,
        )

        self.basin_ids = np.asarray([str(b) for b in basins], dtype=object)

        # ---- build compact index arrays for valid windows ----
        # Stores only (basin_idx, start_idx_on_full_timeline), not full windows.
        basin_idx_chunks = []
        start_idx_chunks = []
        T_all = self.sf_all.shape[1]

        # Match the HydroFlow split semantics:
        #   past    = [start, start + seq_length)
        #   future  = [start + seq_length, start + seq_length + forecast_horizon)
        # The final history day is inside the split, and all future targets stay
        # inside the split. History before split_start is allowed.
        w_start_min = max(0, split_start_idx - self.seq_length + 1)
        w_start_max = min(
            split_end_idx - self.seq_length - self.forecast_horizon + 1,
            T_all - self.total_len,
        )

        if w_start_min > w_start_max:
            self.sample_basin_idx = np.empty((0,), dtype=np.int32)
            self.sample_start_idx = np.empty((0,), dtype=np.int32)
            self.num_samples = 0
            print(f"CamelsNPY [{split_start}→{split_end}]: "
                  f"0 samples from 0 basins")
            return

        start_candidates = np.arange(
            w_start_min, w_start_max + 1, int(stride), dtype=np.int32
        )

        for b_idx in range(self.sf_all.shape[0]):
            valid_sf = ~np.isnan(self.sf_all[b_idx])
            valid_forcing = ~np.isnan(self.forcing_all[b_idx]).any(axis=-1)

            window_valid = np.convolve(
                valid_sf.astype(np.int16),
                np.ones(self.total_len, dtype=np.int16),
                mode="valid",
            ) == self.total_len
            forcing_window_valid = np.convolve(
                valid_forcing.astype(np.int16),
                np.ones(self.total_len, dtype=np.int16),
                mode="valid",
            ) == self.total_len

            if window_valid.size == 0:
                continue

            target_valid = np.convolve(
                valid_sf.astype(np.int16),
                np.ones(self.forecast_horizon, dtype=np.int16),
                mode="valid",
            ) == self.forecast_horizon

            target_start_offset = self.seq_length
            keep = (
                window_valid[start_candidates]
                & forcing_window_valid[start_candidates]
                & target_valid[target_start_offset + start_candidates]
            )
            starts = start_candidates[keep]
            if starts.size == 0:
                continue

            basin_idx_chunks.append(np.full(starts.shape[0], b_idx, dtype=np.int32))
            start_idx_chunks.append(starts)

        if basin_idx_chunks:
            self.sample_basin_idx = np.concatenate(basin_idx_chunks, axis=0)
            self.sample_start_idx = np.concatenate(start_idx_chunks, axis=0)
        else:
            self.sample_basin_idx = np.empty((0,), dtype=np.int32)
            self.sample_start_idx = np.empty((0,), dtype=np.int32)

        self.num_samples = int(self.sample_start_idx.shape[0])
        basins_used = len(np.unique(self.sample_basin_idx)) if self.num_samples > 0 else 0
        print(f"CamelsNPY [{split_start}→{split_end}]: "
              f"{self.num_samples:,} samples from {basins_used} basins")

    # ------------------------------------------------------------------
    def __len__(self):
        return self.num_samples

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        b_idx = int(self.sample_basin_idx[idx])
        start = int(self.sample_start_idx[idx])

        x_5 = self.forcing_all[b_idx, start : start + self.total_len]    # (L+H, 5)
        x_15 = x_5[:, self._forcing_15_idx]                               # (L+H, 15)
        x_t = torch.from_numpy(np.ascontiguousarray(x_15))

        y_start = start + self.seq_length
        y_end = y_start + self.forecast_horizon
        y_t = torch.from_numpy(np.ascontiguousarray(self.sf_all[b_idx, y_start:y_end]))

        norm_m = torch.tensor([self.global_log_mean], dtype=torch.float32)
        norm_s = torch.tensor([self.global_log_std], dtype=torch.float32)
        basin = self.basin_ids[b_idx]
        date = str(self.dates_all[y_start]) if self.include_dates else ""

        if self.no_static:
            return x_t, y_t, norm_m, norm_s, basin, date

        attrs = torch.from_numpy(self.static_normed[b_idx])
        return x_t, attrs, y_t, norm_m, norm_s, basin, date
