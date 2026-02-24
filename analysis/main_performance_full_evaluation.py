import os
import sys
import numpy as np
import pandas as pd

from performance_functions import (
    baseflow_index, bias, flow_duration_curve, get_quant,
    high_flows, low_flows, nse, alpha_nse, beta_nse,
    kge, stdev_rat, zero_freq, FHV, FLV, mass_balance
)

# ============================================================
# Probabilistic helper functions
# ============================================================

def crps_ensemble_batch(X, y, batch=4000):
    """Unbiased ensemble CRPS."""
    N, S = X.shape
    out = np.empty(N, dtype=np.float64)
    for i0 in range(0, N, batch):
        i1 = min(N, i0 + batch)
        Xi = X[i0:i1]
        yi = y[i0:i1, None]
        term1 = np.mean(np.abs(Xi - yi), axis=1)
        diffs = np.abs(Xi[:, :, None] - Xi[:, None, :])
        term2 = 0.5 * np.mean(diffs, axis=(1, 2))
        out[i0:i1] = term1 - term2
    return out

def quantile_reliability(ens_vec, obs_vec, qs=None):
    if qs is None:
        qs = np.linspace(0.05, 0.95, 19)
    E = np.asarray(ens_vec, float)
    y = np.asarray(obs_vec, float).ravel()
    qs = np.asarray(qs, float).ravel()
    valid = np.isfinite(y) & np.isfinite(E).all(axis=1)
    E = E[valid]; y = y[valid]
    pred_q = np.quantile(E, qs, axis=1)
    emp_freq = (y[None, :] <= pred_q).mean(axis=1)
    return pd.DataFrame({"q": qs, "emp_freq": emp_freq, "n": np.full(qs.shape, y.size, int)})

def pit_values(ens_vec, obs_vec):
    E = np.asarray(ens_vec, float)
    y = np.asarray(obs_vec, float).ravel()
    return (E <= y[:, None]).sum(axis=1) / E.shape[1]

def roc_auc_from_prob(p, y):
    order = np.argsort(-p)
    y_sorted = y[order].astype(np.float64)
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    P, N = tp[-1], fp[-1]
    if P == 0 or N == 0:
        return np.nan
    return np.trapz(y=tp/P, x=fp/N)

def event_reliability(p, y, bins=np.linspace(0,1,11)):
    inds = np.digitize(p, bins) - 1
    out = []
    for b in range(len(bins)-1):
        mask = inds == b
        if mask.any():
            out.append({
                "bin_left": bins[b], "bin_right": bins[b+1],
                "count": int(mask.sum()),
                "mean_p": float(p[mask].mean()),
                "obs_freq": float(y[mask].mean())
            })
    return pd.DataFrame(out)

# ============================================================
# Main
# ============================================================

def main():
    if len(sys.argv) < 3:
        print("Usage: python main_performance_full_evaluation.py <experiment_name> <npz_path>")
        # A det example: python main_performance_full_evaluation.py seq2seq_ssm /home/yihan/diffusion_ssm/runs/run_2910_1858_seed3407/deterministic_epoch49.npz

        sys.exit(1)

    experiment = sys.argv[1]
    npz_path   = sys.argv[2]
    out_root = "analysis/ensemble_stats"
    os.makedirs(out_root, exist_ok=True)

    # ---- load npz ----
    print(f"[INFO] Loading {npz_path}")
    f = np.load(npz_path, allow_pickle=True)
    keys = list(f.keys())
    print(f"[INFO] Found keys: {keys}")
    basins = f["basins"]
    dates  = f["dates"]
    obs    = f["obs"]
    if "preds" in keys:
      ens = f["preds"]
    elif "ens" in keys:
      ens    = f["ens"]

    # Detect deterministic vs probabilistic
    # Deterministic = ens.ndim == 2 (N,H)
    deterministic = (ens.ndim == 2)
    print(f"[INFO] Detected {'deterministic' if deterministic else 'ensemble'} npz")

    if deterministic:
        # reshape to (N,1,H) for uniform handling
        ens = ens[:, None, :]

    N, S, H = ens.shape
    unique_basins = np.unique(basins)
    basin_idx = {b: np.where(basins == b)[0] for b in unique_basins}

    # =====================================================
    # Deterministic metrics
    # =====================================================
    leadtime_stats = {lead: [] for lead in range(1, H+1)}

    for b in unique_basins:
        idx = basin_idx[b]
        for lead in range(1, H+1):
            y = obs[idx, lead-1].astype(float)
            x_mean = ens[idx, :, lead-1].mean(axis=1)
            df_single = pd.DataFrame({"qsim": x_mean.clip(min=0), "qobs": y}).dropna()
            if df_single.empty:
                continue

            obs5, sim5 = get_quant(df_single, 0.05)
            obs95, sim95 = get_quant(df_single, 0.95)
            obs0, sim0 = zero_freq(df_single)
            obsH, simH = high_flows(df_single)
            obsL, simL = low_flows(df_single)
            e_fhv = FHV(df_single, 0.1)
            e_flv = FLV(df_single, 0.3)
            e_nse = nse(df_single)
            e_nse_alpha = alpha_nse(df_single)
            e_nse_beta = beta_nse(df_single)
            e_kge, r, alpha, beta = kge(df_single)
            m_total, m_pos, m_neg = mass_balance(df_single)
            e_bias = bias(df_single)
            e_stdev = stdev_rat(df_single)
            obsFDC, simFDC = flow_duration_curve(df_single)

            leadtime_stats[lead].append({
                "basin": b, "lead_time": lead,
                "nse": e_nse, "alpha_nse": e_nse_alpha, "beta_nse": e_nse_beta,
                "kge": e_kge, "kge_r": r, "kge_alpha": alpha, "kge_beta": beta,
                "fhv_01": e_fhv, "flv": e_flv,
                "massbias_total": m_total, "massbias_pos": m_pos, "massbias_neg": m_neg,
                "bias": e_bias, "stdev": e_stdev,
                "obs5": obs5, "sim5": sim5, "obs95": obs95, "sim95": sim95,
                "obs0": obs0, "sim0": sim0, "obsL": obsL, "simL": simL,
                "obsH": obsH, "simH": simH, "obsFDC": obsFDC, "simFDC": simFDC
            })

    # Scalar metric columns — exclude array-valued FDC columns from aggregation
    SCALAR_METRICS = [
        "nse", "alpha_nse", "beta_nse",
        "kge", "kge_r", "kge_alpha", "kge_beta",
        "fhv_01", "flv",
        "massbias_total", "massbias_pos", "massbias_neg",
        "bias", "stdev",
        "obs5", "sim5", "obs95", "sim95",
        "obs0", "sim0", "obsL", "simL", "obsH", "simH",
    ]

    # Collect per-lead median rows for the final summary table
    summary_rows = []

    for lead, rows in leadtime_stats.items():
        df_stats = pd.DataFrame(rows)
        if df_stats.empty:
            continue

        scalar_df = df_stats[["basin", "lead_time"] + SCALAR_METRICS].copy()

        mean_stats   = scalar_df[SCALAR_METRICS].mean()
        median_stats = scalar_df[SCALAR_METRICS].median()
        q25_stats    = scalar_df[SCALAR_METRICS].quantile(0.25)
        q75_stats    = scalar_df[SCALAR_METRICS].quantile(0.75)

        mean_stats["basin"]   = "mean";   mean_stats["lead_time"]   = lead
        median_stats["basin"] = "median"; median_stats["lead_time"] = lead
        q25_stats["basin"]    = "q25";    q25_stats["lead_time"]    = lead
        q75_stats["basin"]    = "q75";    q75_stats["lead_time"]    = lead

        df_stats = pd.concat(
            [df_stats,
             mean_stats.to_frame().T,
             median_stats.to_frame().T,
             q25_stats.to_frame().T,
             q75_stats.to_frame().T],
            ignore_index=True
        )
        out_csv = os.path.join(out_root, f"{experiment}_det_lead{lead}.csv")

        summary_rows.append({
            "lead_time": lead,
            **{m: float(median_stats[m]) for m in SCALAR_METRICS},
            **{f"{m}_q25": float(q25_stats[m]) for m in SCALAR_METRICS},
            **{f"{m}_q75": float(q75_stats[m]) for m in SCALAR_METRICS},
        })

    # ── Write & print the single aggregate summary ──────────────────────────
    df_summary = pd.DataFrame(summary_rows).sort_values("lead_time")
    summary_csv = os.path.join(out_root, f"{experiment}_summary.csv")


    # Overall aggregate across all lead times (median of per-lead medians)
    overall_median = df_summary[SCALAR_METRICS].median()
    overall_q25    = df_summary[[f"{m}_q25" for m in SCALAR_METRICS]].median()
    overall_q75    = df_summary[[f"{m}_q75" for m in SCALAR_METRICS]].median()

    REPORT_METRICS = [
        ("NSE",   "nse"),
        ("KGE",   "kge"),
        ("COR",   "kge_r"),
        ("FHV%",  "fhv_01"),
        ("FLV%",  "flv"),
        ("Bias",  "bias"),
    ]

    col_w = 26
    header  = f"{'Metric':<12}" + "".join(f"{'Lead '+str(r['lead_time']):<{col_w}}" for _, r in df_summary.iterrows())
    divider = "─" * len(header)

    print(f"\n{'═'*len(header)}")
    print(f"  SUMMARY TABLE  —  {experiment}")
    print(f"{'═'*len(header)}")
    print(f"  Median across basins  |  IQR = [Q25, Q75]")
    print(divider)
    print(header)
    print(divider)

    for label, col in REPORT_METRICS:
        row_str = f"{label:<12}"
        for _, r in df_summary.iterrows():
            med = r[col]
            lo  = r[f"{col}_q25"]
            hi  = r[f"{col}_q75"]
            cell = f"{med:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            row_str += f"{cell:<{col_w}}"
        print(row_str)

    print(divider)

    # Overall (all leads pooled)
    print(f"\n  Overall (median across leads):")
    print(f"  {'Metric':<10} {'Median':>8}   IQR [Q25, Q75]")
    print(f"  {'─'*50}")
    for label, col in REPORT_METRICS:
        med = float(overall_median[col])
        lo  = float(overall_q25[f"{col}_q25"])
        hi  = float(overall_q75[f"{col}_q75"])
        print(f"  {label:<10} {med:>+8.3f}   [{lo:+.3f}, {hi:+.3f}]")
    print()

    # =====================================================
    # Skip probabilistic metrics if deterministic
    # =====================================================
    if deterministic:
        print("[INFO] Deterministic run detected — skipping ensemble metrics.")
        print("All deterministic evaluation results saved.")
        return

    # =====================================================
    # Probabilistic metrics (for ensemble npz)
    # =====================================================
    basin_thresh = {b: np.nanpercentile(obs[basin_idx[b], :H].reshape(-1), 95.0) for b in unique_basins}
    prob_rows, rel_quant_rows, pit_rows, event_rel_rows = (
        {lead: [] for lead in range(1, H+1)} for _ in range(4)
    )

    for b in unique_basins:
        idx = basin_idx[b]; thr = basin_thresh[b]
        for lead in range(1, H+1):
            Y = obs[idx, lead-1].astype(float)
            X = ens[idx, :, lead-1].astype(float)

            crps_vec = crps_ensemble_batch(X, Y)
            crps_mean = float(np.nanmean(crps_vec))
            spread_std = X.std(axis=1)
            q05, q95 = np.quantile(X, [0.05, 0.95], axis=1)
            width_90 = q95 - q05

            qrel_df = quantile_reliability(X, Y); qrel_df.insert(0, "basin", b); qrel_df.insert(1, "lead", lead)
            pit_df = pd.DataFrame({"basin": b, "lead": lead, "pit": pit_values(X, Y)})

            y_evt = (Y > thr).astype(int)
            p_evt = (X > thr).mean(axis=1)
            auc = roc_auc_from_prob(p_evt, y_evt)
            rel_evt_df = event_reliability(p_evt, y_evt); rel_evt_df.insert(0, "basin", b); rel_evt_df.insert(1, "lead", lead)

            rel_quant_rows[lead].append(qrel_df)
            pit_rows[lead].append(pit_df)
            event_rel_rows[lead].append(rel_evt_df)

            prob_rows[lead].append({
                "basin": b, "lead_time": lead,
                "crps": crps_mean,
                "sharp_std": float(np.nanmean(spread_std)),
                "sharp_w90": float(np.nanmean(width_90)),
                "event_auc": auc, "event_threshold": thr, "n_samples": len(Y)
            })

    PROB_METRICS = ["crps", "sharp_std", "sharp_w90", "event_auc"]
    prob_summary_rows = []

    for lead in range(1, H+1):
        df_prob = pd.DataFrame(prob_rows[lead])
        if df_prob.empty:
            continue
        med_row  = df_prob[PROB_METRICS].median()
        mean_row = df_prob[PROB_METRICS].mean()
        q25_row  = df_prob[PROB_METRICS].quantile(0.25)
        q75_row  = df_prob[PROB_METRICS].quantile(0.75)

        for tag, r in [("mean", mean_row), ("median", med_row), ("q25", q25_row), ("q75", q75_row)]:
            r = r.copy(); r["basin"] = tag; r["lead_time"] = lead
            df_prob = pd.concat([df_prob, r.to_frame().T], ignore_index=True)

        prob_summary_rows.append({
            "lead_time": lead,
            **{m: float(med_row[m]) for m in PROB_METRICS},
            **{f"{m}_q25": float(q25_row[m]) for m in PROB_METRICS},
            **{f"{m}_q75": float(q75_row[m]) for m in PROB_METRICS},
        })

    if prob_summary_rows:
        df_prob_summary = pd.DataFrame(prob_summary_rows).sort_values("lead_time")
        prob_summary_csv = os.path.join(out_root, f"{experiment}_prob_summary.csv")

        col_w = 26
        header_p = f"{'Metric':<12}" + "".join(
            f"{'Lead '+str(int(r['lead_time'])):<{col_w}}" for _, r in df_prob_summary.iterrows()
        )
        print(f"\n{'═'*len(header_p)}")
        print(f"  PROBABILISTIC SUMMARY  —  {experiment}")
        print(f"{'═'*len(header_p)}")
        print(f"  Median across basins  |  IQR = [Q25, Q75]")
        print("─" * len(header_p))
        print(header_p)
        print("─" * len(header_p))
        for label, col in [("CRPS", "crps"), ("Sharp-STD", "sharp_std"),
                            ("Sharp-W90", "sharp_w90"), ("AUC", "event_auc")]:
            row_str = f"{label:<12}"
            for _, r in df_prob_summary.iterrows():
                med = r[col]; lo = r[f"{col}_q25"]; hi = r[f"{col}_q75"]
                cell = f"{med:.3f} [{lo:.3f}, {hi:.3f}]"
                row_str += f"{cell:<{col_w}}"
            print(row_str)
        print()

    print("All deterministic + probabilistic evaluation results saved.")

    # stats of preds and obs for debugging
    print(f"Overall stats - preds: mean {ens.mean():.4f}, std {ens.std():.4f} | obs: mean {obs.mean():.4f}, std {obs.std():.4f}")
    print(f"Overall stats - preds: min {ens.min():.4f}, max {ens.max():.4f} | obs: min {obs.min():.4f}, max {obs.max():.4f}")
    

if __name__ == "__main__":
    main()
