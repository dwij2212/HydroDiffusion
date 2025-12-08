import os
import pickle
from pathlib import Path
import pdb

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from papercode.datasets import CamelsH5
from papercode.utils import get_basin_list
from papercode.datautils import load_scaler
from papercode.diffusion_lstm import EncoderDecoderDiffusionLSTM
from papercode.diffusion_utils import ddim_sample

from papercode.lstm import Seq2SeqLSTM, EncoderDecoderDetLSTM
from papercode.SSM_test import HOPE, setup_optimizer, Seq2SeqSSM

# --------------------------------------------------------------------
SCALER = load_scaler("/home/eecs/erichson/yihan/diffusion_ssm/global_scaler.json")
PER_BASIN_NORM = False

# --------------------------------------------------------------------
def load_train_q_stats(train_h5_path: Path) -> dict:
    """Basin-specific mean/std if you choose PER_BASIN_NORM = True."""
    stats = {}
    with h5py.File(train_h5_path, "r") as f:
        y      = f["target_data"][:]                 # (N, H)
        basins = [b.decode("ascii") for b in f["sample_2_basin"][:]]

    basin_to_y = {}
    for b, yy in zip(basins, y):
        basin_to_y.setdefault(b, []).append(yy)
    for b, ys in basin_to_y.items():
        ys = np.vstack(ys)
        stats[b] = {"mean": float(ys.mean()), "std": float(ys.std())}
    return stats


# --------------------------------------------------------------------
def _build_det_model(cfg, device):
    """Constructs a deterministic model identical to training code."""
    dyn_in       = 15
    in_size_dyn  = dyn_in if (cfg["no_static"] or not cfg["concat_static"]) else 42
    static_size  = 0 if cfg["no_static"] else (in_size_dyn - dyn_in)
    H            = cfg["forecast_horizon"]

    name = cfg["model_name"]
    if name == 'seq2seq_lstm':   
        return Seq2SeqLSTM(
            in_size   = in_size_dyn,
            hidden    = cfg['hidden_size'],
            horizon   = H,
            n_layers  = cfg.get('n_layers', 1),
            dropout   = cfg['dropout'],
        ).to(cfg['DEVICE'])
    elif name == "encdec_lstm":
        model = EncoderDecoderDetLSTM(
            past_features   = dyn_in,
            future_features = 1,
            static_size     = static_size,
            hidden          = cfg["hidden_size"],
            horizon         = H,
            dropout         = cfg["dropout"]
        )
    elif name == "seq2seq_ssm":
        hope = HOPE(
            d_input   = in_size_dyn,
            d_output  = 1,
            d_model   = cfg['d_model'],
            n_layers  = cfg['n_layers'],
            dropout   = cfg['ssm_dropout'],
            cfg       = cfg,
            prenorm   = cfg['prenorm'],
        )
        return Seq2SeqSSM(
            ssm_backbone   = hope,
            seq_len = 365,
            forecast_horizon = cfg['forecast_horizon']
        ).to(cfg['DEVICE'])
    else:
        raise ValueError(f"Unsupported deterministic model {name}")

    return model.to(device)


# --------------------------------------------------------------------
def evaluate(cfg: dict):
    """
    * diffusion_lstm -> DDIM ensemble (existing behaviour)
    * all other model names -> single deterministic forward pass
    """
    device      = cfg["DEVICE"]
    run_dir     = Path(cfg["run_dir"])
    model_pt    = run_dir / "best_model.pt"
    ckpt_npz    = run_dir / "predictions_checkpoint.npz"

    # ------------------------- shared/individual HDF5 -------------------
    if cfg.get("h5_dir"):
        shared   = Path(cfg["h5_dir"])
        train_h5 = shared / "data/train/train_data.h5"
        test_h5  = shared / "data/test/test_data.h5"
        db_path  = shared / "attributes.db"
    else:
        train_h5 = run_dir / "data/train/train_data.h5"
        test_h5  = run_dir / "data/test/test_data.h5"
        db_path  = run_dir / "attributes.db"

    train_stats = load_train_q_stats(train_h5)

    # fast-return cache
    if ckpt_npz.exists():
        return np.load(ckpt_npz, allow_pickle=True)

    # ------------------------- build correct model ----------------------
    if cfg["model_name"] == "diffusion_lstm":
        model = EncoderDecoderDiffusionLSTM(
            past_features    = 15,
            static_size      = 0 if cfg["no_static"] else 27,
            hidden_size      = cfg["hidden_size"],
            time_emb_dim     = cfg.get("time_emb_dim", 16),
            forecast_horizon = cfg["forecast_horizon"]
        ).to(device)
        is_diffusion = True
    else:
        model = _build_det_model(cfg, device)
        is_diffusion = False

    model.load_state_dict(torch.load(model_pt, map_location=device))
    model.eval()

    # ------------------------- dataset / loader -------------------------
    basins_all = get_basin_list(cfg["camels_root"])
    basins     = basins_all
    print(f"[INFO] Evaluating {len(basins)} basins")

    test_ds = CamelsH5(
        test_h5,
        basins,
        str(db_path),
        concat_static    = cfg["concat_static"],
        no_static        = cfg["no_static"],
        model_name       = cfg["model_name"],
        forecast_horizon = cfg["forecast_horizon"],
        include_dates    = True,
    )
    test_loader = DataLoader(test_ds,
                             batch_size   = cfg.get("batch_size", 256),
                             shuffle      = False,
                             num_workers  = cfg["num_workers"])

    # ------------------------- storage ----------------------------------
    all_preds, all_tgts, all_basin_ids, all_dates, all_ens = [], [], [], [], []

    # ----- helpers for de-normalisation
    def _denorm(arr, b_ids):
        if PER_BASIN_NORM:
            m = torch.tensor([train_stats[b]["mean"] for b in b_ids],
                             device=device).unsqueeze(1)
            s = torch.tensor([train_stats[b]["std"]  for b in b_ids],
                             device=device).unsqueeze(1)
        else:
            m = torch.tensor(SCALER["QObs(mm/d)_mean"], device=device)
            s = torch.tensor(SCALER["QObs(mm/d)_std"],  device=device)
        return arr * s + m

    # ------------------------- MAIN LOOP --------------------------------
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="batches"):
            if cfg["no_static"]:
                x_d, y_t, q_m, q_s, basin_batch, date_batch = batch
                static_attrs = None
            else:
                x_d, static_attrs, y_t, q_m, q_s, basin_batch, date_batch = batch
                static_attrs = static_attrs.to(device)

            x_d   = x_d.to(device)
            y_t   = y_t.to(device)
            fh    = cfg["forecast_horizon"]
            B     = x_d.size(0)

            # -------- choose scalers per batch --------------------------
            def denorm(a):  # a is (B,H) or (S,B,H)
                return _denorm(a, basin_batch)

            if is_diffusion:
                # ========================================================
                # DIFFUSION BRANCH
                # ========================================================
                x_past = x_d[:, :-fh, :]
                #precip = x_d[:, -fh:, 0:1]

                # build velocity-prediction wrapper
                def v_fn(noisy, t, s_attr, pcp):
                    return model(x_past,
                                 noisy.unsqueeze(-1),
                                 t,
                                 s_attr,
                                 pcp).squeeze(-1)

                num_ens   = cfg.get("num_samples", 50)
                ddim_steps= cfg.get("ddim_steps", 50)
                ens = []
                for _ in range(num_ens):
                    smp = ddim_sample(v_fn,
                                      shape         = (B, fh),
                                      device        = device,
                                      static_attrs  = static_attrs,
                                      future_precip = precip,
                                      eta           = 0.0,
                                      steps         = ddim_steps)
                    ens.append(smp)                           # (B,H)

                ens  = torch.stack(ens, dim=0)               # (S,B,H)
                ens_d = denorm(ens)                          # de-norm
                preds = ens_d.mean(dim=0)                    # (B,H)
                all_ens.append(ens_d.cpu().numpy().transpose(1,0,2))
            else:
                # ========================================================
                #  DETERMINISTIC BRANCH
                # ========================================================
                model_name = cfg["model_name"]
                if model_name == "seq2seq_lstm":
                    x_full = x_d
                    if (not cfg["no_static"]) and cfg["concat_static"]:
                        stat = static_attrs.unsqueeze(1).repeat(1, x_full.size(1), 1)
                        x_full = torch.cat([x_full, stat], dim=-1)
                    preds = model(x_full).squeeze(-1)        # (B,H)

                elif model_name == "encdec_lstm":
                    x_past      = x_d[:, :-fh, :]
                    future_prec = x_d[:, -fh:, 0:1]
                    if (not cfg["no_static"]) and cfg["concat_static"]:
                        stat_p = static_attrs.unsqueeze(1).repeat(1, x_past.size(1),     1)
                        stat_f = static_attrs.unsqueeze(1).repeat(1, future_prec.size(1), 1)
                        x_past      = torch.cat([x_past,      stat_p], dim=-1)
                        future_prec = torch.cat([future_prec, stat_f], dim=-1)
                    preds = model(x_past,
                                  future_prec,
                                  None if cfg["concat_static"] else static_attrs
                                 ).squeeze(-1)               # (B,H)

                elif model_name == "seq2seq_ssm":
                    x_full = x_d
                    if (not cfg['no_static']) and cfg['concat_static']:
                        static_exp = static_attrs.unsqueeze(1)              # (B,1,27)
                        static_exp = static_exp.repeat(1, x_full.size(1), 1)
                        x_full = torch.cat([x_full, static_exp], dim=-1)    # (B,L_c,42)
                    preds_all = model(x_full)                   # (B, Lc, 1)
                    preds     = preds_all[:, -fh:, :]           # (B, fh, 1)

                preds = denorm(preds)                        # (B,H')
                all_ens.append(None)                         # placeholder

            # -------- save for whole dataset -----------------------------
            all_preds.append(preds.cpu().numpy())
            all_tgts .append(y_t.cpu().numpy())
            all_basin_ids.extend(basin_batch)
            all_dates.extend(pd.to_datetime(date_batch))

    # ------------------------- flatten & save ----------------------------
    preds_arr = np.vstack(all_preds)
    tgts_arr  = np.vstack(all_tgts)
    bas       = np.array(all_basin_ids, dtype=object)
    dts       = np.array(all_dates,    dtype=object)
    ens_arr   = None if all_ens[0] is None else np.vstack(all_ens)
    
    
    if preds_arr.ndim == 3 and preds_arr.shape[-1] == 1:
        preds_arr = preds_arr.squeeze(-1)
    if tgts_arr.ndim == 3 and tgts_arr.shape[-1] == 1:
        tgts_arr = tgts_arr.squeeze(-1)

    np.savez_compressed(
        ckpt_npz,
        preds       = preds_arr,
        basin_ids   = bas,
        timestamps  = dts,
        all_members = ens_arr,
        targets     = tgts_arr,
    )
    print(f"[INFO] Saved raw arrays to {ckpt_npz}")

    # ------------------------- per-basin DataFrames ----------------------
    per_basin = {}
    for idx, b in enumerate(bas):
        per_basin.setdefault(b, {"dates": [], "preds": [], "obs": []})
        per_basin[b]["dates"].append(dts[idx])
        per_basin[b]["preds"].append(preds_arr[idx])
        per_basin[b]["obs"].append(tgts_arr[idx])

    dfs = {}
    for b, d in per_basin.items():
        obs_mat   = np.vstack(d["obs"])
        pred_mat  = np.vstack(d["preds"])
        dates     = pd.to_datetime(d["dates"])
        pdb.set_trace()
        Hcol      = pred_mat.shape[1]
        cols      = ["q_obs_t+1"] + [f"qsim_t+{i+1}" for i in range(Hcol)]
        df        = pd.DataFrame(
            np.hstack([obs_mat[:, :1], pred_mat]), index=dates, columns=cols
        )
        dfs[b] = df

    pkl_out = run_dir / "test_results.pkl"
    with open(pkl_out, "wb") as f:
        pickle.dump(dfs, f)
    print(f"[INFO] Saved per-basin DataFrames to {pkl_out}")
