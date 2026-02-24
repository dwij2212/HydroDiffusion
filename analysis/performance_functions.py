"""
performance_functions.py
------------------------
Hydrological performance metrics for streamflow evaluation.

All functions accept a pandas DataFrame with at minimum two columns:
    - 'qobs'  : observed streamflow (any non-negative units)
    - 'qsim'  : simulated / modelled streamflow (same units)

Rows containing NaN in either column are automatically dropped inside
each function via a shared helper.

References
----------
- Nash & Sutcliffe (1970) J. Hydrol.
- Gupta et al. (2009) J. Hydrol.  [alpha / beta NSE decomposition]
- Kling et al. (2012) J. Hydrol.  [KGE]
- Yilmaz et al. (2008) WRR        [FHV / FLV / FDC signatures]
- Kratzert et al. (2019) WRR      [FDC-based signatures used in LSTM hydrology]
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame):
    """Return obs and sim arrays with NaN rows removed."""
    d = df[["qobs", "qsim"]].dropna()
    obs = d["qobs"].values.astype(float)
    sim = d["qsim"].values.astype(float)
    return obs, sim


# ─────────────────────────────────────────────────────────────
# 1. Nash-Sutcliffe Efficiency  (NSE)
# ─────────────────────────────────────────────────────────────

def nse(df: pd.DataFrame) -> float:
    """Nash-Sutcliffe Efficiency.

    NSE = 1 - Σ(obs-sim)² / Σ(obs-mean(obs))²

    Returns
    -------
    float
        NSE in (−∞, 1].  Returns np.nan if all obs are equal.
    """
    obs, sim = _clean(df)
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom == 0:
        return np.nan
    return float(1.0 - np.sum((obs - sim) ** 2) / denom)


# ─────────────────────────────────────────────────────────────
# 2. Alpha and Beta decomposition of NSE  (Gupta et al. 2009)
# ─────────────────────────────────────────────────────────────

def alpha_nse(df: pd.DataFrame) -> float:
    """Alpha component of NSE decomposition (ratio of std deviations).

    α = σ_sim / σ_obs
    """
    obs, sim = _clean(df)
    if np.std(obs) == 0:
        return np.nan
    return float(np.std(sim) / np.std(obs))


def beta_nse(df: pd.DataFrame) -> float:
    """Beta component of NSE decomposition (normalised mean bias).

    β = (μ_sim − μ_obs) / σ_obs
    """
    obs, sim = _clean(df)
    if np.std(obs) == 0:
        return np.nan
    return float((np.mean(sim) - np.mean(obs)) / np.std(obs))


# ─────────────────────────────────────────────────────────────
# 3. Kling-Gupta Efficiency  (KGE, Kling et al. 2012)
# ─────────────────────────────────────────────────────────────

def kge(df: pd.DataFrame):
    """Kling-Gupta Efficiency and its three components.

    KGE = 1 − √[(r−1)² + (α−1)² + (β−1)²]

    where
        r = Pearson correlation
        α = σ_sim / σ_obs   (variability ratio)
        β = μ_sim / μ_obs   (bias ratio)

    Returns
    -------
    kge_val : float
    r       : float  (Pearson correlation)
    alpha   : float  (variability ratio)
    beta    : float  (bias ratio)
    """
    obs, sim = _clean(df)
    mu_obs, mu_sim = obs.mean(), sim.mean()
    sig_obs, sig_sim = obs.std(), sim.std()

    if sig_obs == 0 or mu_obs == 0:
        return np.nan, np.nan, np.nan, np.nan

    r = float(np.corrcoef(obs, sim)[0, 1])
    alpha = float(sig_sim / sig_obs)
    beta  = float(mu_sim  / mu_obs)
    kge_val = float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    return kge_val, r, alpha, beta


# ─────────────────────────────────────────────────────────────
# 4. Bias  (relative mean bias)
# ─────────────────────────────────────────────────────────────

def bias(df: pd.DataFrame) -> float:
    """Relative mean bias.

    bias = (μ_sim − μ_obs) / μ_obs
    """
    obs, sim = _clean(df)
    mu_obs = obs.mean()
    if mu_obs == 0:
        return np.nan
    return float((sim.mean() - mu_obs) / mu_obs)


# ─────────────────────────────────────────────────────────────
# 5. Standard-deviation ratio
# ─────────────────────────────────────────────────────────────

def stdev_rat(df: pd.DataFrame) -> float:
    """Ratio of standard deviations  σ_sim / σ_obs."""
    obs, sim = _clean(df)
    if np.std(obs) == 0:
        return np.nan
    return float(np.std(sim) / np.std(obs))


# ─────────────────────────────────────────────────────────────
# 6. Zero-flow frequency
# ─────────────────────────────────────────────────────────────

def zero_freq(df: pd.DataFrame, threshold: float = 0.001):
    """Fraction of timesteps with near-zero flow.

    Parameters
    ----------
    threshold : float
        Values below this are considered zero (default 0.001).

    Returns
    -------
    obs_zero_freq : float  (observed)
    sim_zero_freq : float  (simulated)
    """
    obs, sim = _clean(df)
    obs_zf = float((obs < threshold).mean())
    sim_zf = float((sim < threshold).mean())
    return obs_zf, sim_zf


# ─────────────────────────────────────────────────────────────
# 7. High-flow statistics  (mean of top-10 % flows)
# ─────────────────────────────────────────────────────────────

def high_flows(df: pd.DataFrame, h: float = 0.10):
    """Mean of the top-h fraction of flows.

    Parameters
    ----------
    h : float
        Fraction of highest flows to average (default 0.10 = top 10 %).

    Returns
    -------
    obs_high : float
    sim_high : float
    """
    obs, sim = _clean(df)
    n = max(1, int(np.round(h * len(obs))))
    obs_high = float(np.sort(obs)[-n:].mean())
    sim_high = float(np.sort(sim)[-n:].mean())
    return obs_high, sim_high


# ─────────────────────────────────────────────────────────────
# 8. Low-flow statistics  (mean of bottom-10 % flows)
# ─────────────────────────────────────────────────────────────

def low_flows(df: pd.DataFrame, l: float = 0.10):
    """Mean of the bottom-l fraction of flows.

    Parameters
    ----------
    l : float
        Fraction of lowest flows to average (default 0.10 = bottom 10 %).

    Returns
    -------
    obs_low : float
    sim_low : float
    """
    obs, sim = _clean(df)
    n = max(1, int(np.round(l * len(obs))))
    obs_low = float(np.sort(obs)[:n].mean())
    sim_low = float(np.sort(sim)[:n].mean())
    return obs_low, sim_low


# ─────────────────────────────────────────────────────────────
# 9. Generic quantile getter
# ─────────────────────────────────────────────────────────────

def get_quant(df: pd.DataFrame, q: float):
    """Return the q-th quantile of obs and sim.

    Parameters
    ----------
    q : float
        Quantile in [0, 1].

    Returns
    -------
    obs_q : float
    sim_q : float
    """
    obs, sim = _clean(df)
    return float(np.quantile(obs, q)), float(np.quantile(sim, q))


# ─────────────────────────────────────────────────────────────
# 10. Flow Duration Curve  (FDC)
# ─────────────────────────────────────────────────────────────

def flow_duration_curve(df: pd.DataFrame, n_points: int = 100):
    """Compute the flow duration curve for obs and sim.

    The FDC is the empirical exceedance-probability curve:
    flows sorted in descending order plotted against the
    fraction of time they are exceeded.

    Parameters
    ----------
    n_points : int
        Number of evenly-spaced exceedance probabilities to return
        (default 100).

    Returns
    -------
    obs_fdc : np.ndarray  shape (n_points,)
    sim_fdc : np.ndarray  shape (n_points,)
        Flow values at each exceedance probability level.
    """
    obs, sim = _clean(df)
    probs = np.linspace(0.0, 1.0, n_points)
    # quantile at prob p  ≡  value exceeded (1-p) fraction of the time
    obs_fdc = np.quantile(np.sort(obs), 1.0 - probs)
    sim_fdc = np.quantile(np.sort(sim), 1.0 - probs)
    return obs_fdc, sim_fdc


# ─────────────────────────────────────────────────────────────
# 11. FHV — Peak-flow bias (Yilmaz 2008)
# ─────────────────────────────────────────────────────────────

def FHV(df: pd.DataFrame, h: float = 0.02) -> float:
    """Peak-flow (high-volume) bias of the FDC.

    FHV = Σ(sim_top − obs_top) / Σ(obs_top) × 100  [%]

    Parameters
    ----------
    h : float
        Top fraction of flows considered (default 0.02 = top 2 %).

    Returns
    -------
    float
        FHV in percent.  Positive ⇒ over-prediction.
    """
    obs, sim = _clean(df)
    # sort descending
    obs_s = np.sort(obs)[::-1]
    sim_s = np.sort(sim)[::-1]
    n = max(1, int(np.round(h * len(obs_s))))
    obs_top = obs_s[:n]
    sim_top = sim_s[:n]
    denom = np.sum(obs_top) + 1e-6
    return float(np.sum(sim_top - obs_top) / denom * 100.0)


# ─────────────────────────────────────────────────────────────
# 12. FLV — Low-flow bias (Yilmaz 2008)
# ─────────────────────────────────────────────────────────────

def FLV(df: pd.DataFrame, l: float = 0.70) -> float:
    """Low-flow bias of the FDC (log-transformed).

    The bottom (1-l) fraction of the FDC is used (e.g. l=0.7 → bottom 30 %).

    FLV = −(Σlog(sim_low) − Σlog(obs_low)) / (Σlog(obs_low) + ε) × 100  [%]

    Parameters
    ----------
    l : float
        Threshold exceedance fraction; flows below this fraction are the
        low-flow segment (default 0.70).

    Returns
    -------
    float
        FLV in percent.  Negative ⇒ under-prediction of low flows.
    """
    obs, sim = _clean(df)
    # replace zeros to avoid log(0)
    obs = np.where(obs == 0, 1e-6, obs)
    sim = np.where(sim == 0, 1e-6, sim)
    # sort descending
    obs_s = np.sort(obs)[::-1]
    sim_s = np.sort(sim)[::-1]
    n_start = int(np.round(l * len(obs_s)))
    obs_low = np.log(obs_s[n_start:] + 1e-6)
    sim_low = np.log(sim_s[n_start:] + 1e-6)
    qol = np.sum(obs_low - obs_low.min())
    qsl = np.sum(sim_low - sim_low.min())
    return float(-1.0 * (qsl - qol) / (qol + 1e-6) * 100.0)


# ─────────────────────────────────────────────────────────────
# 13. Mass balance
# ─────────────────────────────────────────────────────────────

def mass_balance(df: pd.DataFrame):
    """Volumetric mass-balance errors split into total, positive, and negative.

    total : (Σsim − Σobs) / Σobs            (relative total volume error)
    pos   : Σmax(sim−obs, 0) / Σobs         (relative over-prediction volume)
    neg   : Σmax(obs−sim, 0) / Σobs         (relative under-prediction volume)

    Returns
    -------
    total : float
    pos   : float
    neg   : float
    """
    obs, sim = _clean(df)
    total_obs = np.sum(obs) + 1e-6
    diff = sim - obs
    total = float(np.sum(diff) / total_obs)
    pos   = float(np.sum(np.maximum(diff,  0.0)) / total_obs)
    neg   = float(np.sum(np.maximum(-diff, 0.0)) / total_obs)
    return total, pos, neg


# ─────────────────────────────────────────────────────────────
# 14. Baseflow index
# ─────────────────────────────────────────────────────────────

def baseflow_index(df: pd.DataFrame, alpha: float = 0.925, n_passes: int = 3):
    """Estimate the Baseflow Index (BFI) using the recursive digital filter
    of Eckhardt (2005), applied to both obs and sim.

    BFI = Σbaseflow / Σtotalflow

    The filter is:
        b[t] = (α · b[t-1] + (1-α)/2 · (Q[t] + Q[t-1])) clipped to Q[t]

    Parameters
    ----------
    alpha : float
        Filter parameter (default 0.925, commonly used in hydrology).
    n_passes : int
        Number of forward passes of the filter (default 3).

    Returns
    -------
    obs_bfi : float
    sim_bfi : float
    """
    def _bfi(q: np.ndarray) -> float:
        q = np.maximum(q, 0.0)
        b = q.copy()
        for _ in range(n_passes):
            b_new = np.empty_like(b)
            b_new[0] = b[0]
            for t in range(1, len(q)):
                b_new[t] = min(
                    alpha * b_new[t-1] + (1.0 - alpha) / 2.0 * (q[t] + q[t-1]),
                    q[t]
                )
            b = b_new
        total = np.sum(q)
        return float(np.sum(b) / total) if total > 0 else np.nan

    obs, sim = _clean(df)
    return _bfi(obs), _bfi(sim)
