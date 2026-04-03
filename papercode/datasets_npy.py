"""
papercode/datasets_npy.py

Drop-in replacement for CamelsH5 that reads from a single .npy array
with shape [num_basins, num_days, 33]:
    - cols  0-26 : 27 static catchment attributes  (constant across time)
    - cols 27-31 : 5 dynamic forcing features       (PRCP, SRAD, Tmax, Tmin, Vp)
    - col  32    : streamflow (SF)

Normalization is computed at runtime from the **training** data passed in.
The __getitem__ signature matches CamelsH5 exactly, so the existing
training / evaluation loops work without modification.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

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
    Compute global mean / std of forcings and SF over the training period
    (all basins × training days).  Called once, result shared by all splits.

    Returns a dict with the same keys as the repo's SCALAR convention:
        input_means  (5,)
        input_stds   (5,)
        output_mean  (1,)
        output_std   (1,)
        static_means (27,)
        static_stds  (27,)
    """
    date_series = pd.to_datetime(dates)
    mask = (date_series >= train_start) & (date_series <= train_end)
    train_idx = np.where(mask)[0]

    forcing = data[:, train_idx, 27:32]          # (B, T_train, 5)
    sf      = data[:, train_idx, 32:33]          # (B, T_train, 1)
    static  = data[:, 0, :27]                    # (B, 27) — constant per basin

    f_flat = forcing.reshape(-1, 5)
    s_flat = sf.reshape(-1, 1)

    scalar = {
        "input_means":  np.nanmean(f_flat, axis=0),   # (5,)
        "input_stds":   np.nanstd(f_flat, axis=0),    # (5,)
        "output_mean":  np.nanmean(s_flat, axis=0),   # (1,)
        "output_std":   np.nanstd(s_flat, axis=0),    # (1,)
        "static_means": np.nanmean(static, axis=0),   # (27,)
        "static_stds":  np.nanstd(static, axis=0),    # (27,)
    }
    # guard against zero std
    scalar["input_stds"][scalar["input_stds"] == 0] = 1.0
    scalar["output_stds"] = np.where(scalar["output_std"] == 0, 1.0, scalar["output_std"])
    scalar["static_stds"][scalar["static_stds"] == 0] = 1.0

    return scalar


# ------------------------------------------------------------------ #
#  Per-basin discharge statistics (needed for NSE loss)              #
# ------------------------------------------------------------------ #

def compute_per_basin_q_stats(
    data: np.ndarray,
    dates: np.ndarray,
    train_start: str = "1980-10-01",
    train_end: str = "1990-09-30",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-basin mean & std of raw streamflow over the training period.

    Returns
    -------
    q_means : (B, 1)
    q_stds  : (B, 1)
    """
    date_series = pd.to_datetime(dates)
    mask = (date_series >= train_start) & (date_series <= train_end)
    train_idx = np.where(mask)[0]

    sf = data[:, train_idx, 32]                  # (B, T_train)
    q_means = np.nanmean(sf, axis=1, keepdims=True).astype(np.float32)  # (B, 1)
    q_stds  = np.nanstd(sf, axis=1, keepdims=True).astype(np.float32)  # (B, 1)

    # guard: if a basin has std==0 (e.g. all NaN), set to 1 to avoid /0
    q_stds[q_stds == 0] = 1.0
    return q_means, q_stds


# ------------------------------------------------------------------ #
#  Sliding-window builder (mirrors create_h5_files logic)            #
# ------------------------------------------------------------------ #

def _build_windows(
    forcing: np.ndarray,
    sf: np.ndarray,
    dates_split: np.ndarray,
    seq_length: int,
    forecast_horizon: int,
    basin_idx: int,
    basin_id: str,
    q_mean: float,
    q_std: float,
    is_train: bool,
    include_dates: bool,
    stride: int = 1,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Create sliding windows for **one basin, one split**.

    Each window:
      x  : (seq_length + forecast_horizon, 5)  — normalized forcing
           (past 365 days + future 8 days of meteorological forcing,
            concatenated exactly as in create_h5_files / CamelsH5)
      y  : (forecast_horizon,)                  — raw (un-normalized) SF targets

    Windows whose target contains **any** NaN or negative value are dropped.

    Parameters
    ----------
    stride : int
        Step size between consecutive window start positions.
        stride=1  → original HydroDiffusion behaviour (~3 280 windows / basin / 10 yr)
        stride=90 → ~36 windows / basin / 10 yr (good for fast debugging)

    Returns None if no valid windows exist for this basin.
    """
    T = forcing.shape[0]               # time-steps in this split
    total_len = seq_length + forecast_horizon
    N = T - total_len + 1               # number of possible windows
    if N <= 0:
        return None

    # pre-allocate
    x_list = []
    y_list = []
    date_list = []

    for i in range(0, N, stride):
        # target: fh days of raw SF starting at (i + seq_length)
        # include nowcasting day: indices [i+seq_length-1 .. i+seq_length+fh-2]
        y_start = i + seq_length - 1        # nowcast day
        y_end   = y_start + forecast_horizon
        target = sf[y_start:y_end]          # (fh,)

        # skip if any target is NaN (streamflow is now z-score normalized,
        # so negative values are valid — only NaN means missing data)
        if np.isnan(target).any():
            continue

        # input: (seq_length + fh) consecutive days of forcing
        x_window = forcing[i : i + total_len]    # (seq_length+fh, 5)
        x_list.append(x_window)
        y_list.append(target)

        if include_dates:
            # date of the last past day (i.e. the "nowcast" day)
            date_list.append(str(dates_split[i + seq_length - 1]))

    if len(x_list) == 0:
        return None

    x_arr = np.stack(x_list).astype(np.float32)   # (n, total_len, 5)
    y_arr = np.stack(y_list).astype(np.float32)    # (n, fh)

    n = x_arr.shape[0]
    return {
        "x":       x_arr,
        "y":       y_arr,
        "q_mean":  np.full((n, 1), q_mean, dtype=np.float32),
        "q_std":   np.full((n, 1), q_std,  dtype=np.float32),
        "basin":   np.array([basin_id] * n),
        "dates":   np.array(date_list) if include_dates else None,
    }


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
    q_means, q_stds : np.ndarray
        Per-basin discharge statistics (B,1) from ``compute_per_basin_q_stats``.
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
        q_means: np.ndarray,
        q_stds: np.ndarray,
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

        # ---- date mask for this split ----
        date_series = pd.to_datetime(dates)
        mask = (date_series >= split_start) & (date_series <= split_end)
        split_idx = np.where(mask)[0]
        self.dates_split = dates[split_idx]

        # ---- normalize static attrs (z-score across basins) ----
        static_raw = data[:, 0, :27].copy()                        # (B, 27)
        self.static_normed = (
            (static_raw - scalar["static_means"]) / scalar["static_stds"]
        ).astype(np.float32)                                        # (B, 27)

        # ---- normalize forcing globally ----
        # work on the split slice only to save memory
        self.forcing_split = data[:, split_idx, 27:32].copy()       # (B, T_split, 5)
        self.forcing_split = (
            (self.forcing_split - scalar["input_means"]) / scalar["input_stds"]
        ).astype(np.float32)

        self.sf_split = data[:, split_idx, 32].copy()               # (B, T_split)
        # Normalize streamflow per-basin using training q_mean/q_std.
        # evaluate_npy._denorm inverts this as: pred * q_std + q_mean
        self.sf_split = (
            (self.sf_split - q_means[:, 0:1]) / q_stds[:, 0:1]
        ).astype(np.float32)

        # Expand 5 forcing variables to 15 virtual channels lazily in __getitem__.
        # Final channel order matches the existing code expectations:
        # [prcp_nldas, prcp_maurer, prcp_daymet, srad_nldas, ...]
        self._forcing_15_idx = np.array(
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            dtype=np.int64,
        )

        self.q_means = np.asarray(q_means, dtype=np.float32).reshape(-1)
        self.q_stds = np.asarray(q_stds, dtype=np.float32).reshape(-1)
        self.basin_ids = np.asarray([str(b) for b in basins], dtype=object)

        # ---- build compact index arrays for valid windows ----
        # Stores only (basin_idx, start_idx_in_split), not full windows.
        basin_idx_chunks = []
        start_idx_chunks = []
        T_split = self.sf_split.shape[1]

        for b_idx in range(self.sf_split.shape[0]):
            N = T_split - self.total_len + 1
            if N <= 0:
                continue

            valid_sf = ~np.isnan(self.sf_split[b_idx])
            # target starts at (start + seq_length - 1) and spans forecast_horizon steps
            target_valid = np.convolve(
                valid_sf.astype(np.int16),
                np.ones(self.forecast_horizon, dtype=np.int16),
                mode="valid",
            ) == self.forecast_horizon

            start_candidates = np.arange(0, N, int(stride), dtype=np.int32)
            target_start_offset = self.seq_length - 1
            keep = target_valid[target_start_offset + start_candidates]
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

        x_5 = self.forcing_split[b_idx, start : start + self.total_len]  # (L+H, 5)
        x_15 = x_5[:, self._forcing_15_idx]                               # (L+H, 15)
        x_t = torch.from_numpy(np.ascontiguousarray(x_15))

        y_start = start + self.seq_length - 1
        y_end = y_start + self.forecast_horizon
        y_t = torch.from_numpy(np.ascontiguousarray(self.sf_split[b_idx, y_start:y_end]))

        q_m = torch.tensor([self.q_means[b_idx]], dtype=torch.float32)
        q_s = torch.tensor([self.q_stds[b_idx]], dtype=torch.float32)
        basin = self.basin_ids[b_idx]
        date = str(self.dates_split[y_start]) if self.include_dates else ""

        if self.no_static:
            return x_t, y_t, q_m, q_s, basin, date

        attrs = torch.from_numpy(self.static_normed[b_idx])
        return x_t, attrs, y_t, q_m, q_s, basin, date
