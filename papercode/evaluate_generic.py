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
from papercode.diffusion_utils import ddim_sample

from papercode.lstm import Seq2SeqLSTM, EncoderDecoderDetLSTM
from papercode.SSM_test import HOPE, Seq2SeqSSM

from papercode.backbones.lstm import GenericLSTM
from papercode.backbones.unet import unet
from papercode.backbones.unet_film import unet_film

from papercode.backbones.unet_film_v2 import unet_film_v2
from papercode.backbones.unet_film_v3 import unet_film_v3
from papercode.backbones.unet_attention import unet_attention
from papercode.backbones.unet_attention_film_v2 import unet_attention_film_v2
from papercode.backbones.unet_attention_film_v3 import unet_attention_film_v3

from papercode.backbones.ssm_v1 import ssm_v1
from papercode.backbones.ssm_v2 import ssm_v2
from papercode.backbones.ssm_encoder import ssm_encoder

from papercode.diffusion_wrapper import EncoderDecoderDiffusionWrapper

# --------------------------------------------------------------------
SCALER = load_scaler("/home/yihan/diffusion_ssm/global_scaler.json")
PER_BASIN_NORM = False

# --------------------------------------------------------------------
def load_train_q_stats(train_h5_path: Path) -> dict:
    """Basin-specific mean/std if you choose PER_BASIN_NORM = True."""
    stats = {}
    with h5py.File(train_h5_path, "r") as f:
        y = f["target_data"][:]                 # (N, H)
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
            n_layers  = cfg.get('lstm_nlayers', 1),
            dropout   = cfg['dropout'],
        ).to(device)

    elif name == "encdec_lstm":
        return EncoderDecoderDetLSTM(
            past_features   = dyn_in,
            future_features = 1,
            static_size     = static_size,
            hidden          = cfg["hidden_size"],
            horizon         = H,
            dropout         = cfg["dropout"]
        ).to(device)

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
            ssm_backbone     = hope,
            seq_len          = cfg['seq_length'],
            forecast_horizon = cfg['forecast_horizon']
        ).to(device)

    else:
        raise ValueError(f"Unsupported deterministic model {name}")

# --------------------------------------------------------------------
def evaluate(cfg: dict):
    """
    * diffusion_lstm -> DDIM ensemble (new GenericLSTM + wrapper)
    * all other model names -> single deterministic forward pass
    """
    device   = cfg["DEVICE"]
    run_dir  = Path(cfg["run_dir"])
    model_pt = run_dir / "best_model.pt" # todo
    #model_pt = run_dir / "model_epoch58.pt" # todo
    cache    = run_dir / "predictions_checkpoint.npz"

    # --- data paths ---
    if cfg.get("h5_dir"):
        base     = Path(cfg["h5_dir"]) / "data"
        train_h5 = base / "train/train_data.h5"
        test_h5  = base / "test/test_data.h5"
        db_path  = Path(cfg["h5_dir"]) / "attributes.db"
    else:
        train_h5 = run_dir / "data/train/train_data.h5"
        test_h5  = run_dir / "data/test/test_data.h5"
        db_path  = run_dir / "attributes.db"

    train_stats = load_train_q_stats(train_h5)

    # fast-return
    if cache.exists():
        return np.load(cache, allow_pickle=True)

    # --- build model ---
    is_diffusion = (cfg["model_name"] in ["diffusion_lstm", "diffusion_unet", "diffusion_ssm", "diffusion_ssm_unet", "diffusion_ssm_lstm"])
    if is_diffusion:
        # instead of old EncoderDecoderDiffusionLSTM, we do:
        dyn_in      = 15
        in_size_dyn = dyn_in if (cfg["no_static"] or not cfg["concat_static"]) else 42
        static_size = 0 if cfg["no_static"] else (in_size_dyn - dyn_in)
        
        if cfg["model_name"] == "diffusion_lstm":
            # 1) encoder: GenericLSTM over the past (with static already concatenated in data)
            encoder = GenericLSTM(
                input_size = in_size_dyn,
                hidden_size= cfg["hidden_size"],
                dropout    = cfg["dropout"],
                init_forget_bias=cfg['initial_forget_gate_bias'],
                batch_first= True
            )
            # 2) decoder: GenericLSTM consuming one-dim noise
            decoder = GenericLSTM(
                input_size = 16, #todo, 2 for concatenation, 1 for no concatenation, 258 for concat also t_emb, 4 for nowcasting
                hidden_size= cfg["hidden_size"],
                dropout    = cfg["dropout"],
                init_forget_bias=cfg['initial_forget_gate_bias'],
                batch_first= True
            )
            # 3) wrap them
            model = EncoderDecoderDiffusionWrapper(
                encoder      = encoder,
                decoder      = decoder,
                hidden_size  = cfg["hidden_size"],
                time_emb_dim = cfg.get("time_emb_dim", 256),
                decoder_name = 'lstm'
            ).to(device)
            
        elif cfg["model_name"] == "diffusion_unet":
            encoder = GenericLSTM(
                input_size = in_size_dyn,
                hidden_size = cfg['hidden_size'],
                num_layers = cfg.get('lstm_nlayers',1),
                dropout = cfg['dropout'],
                init_forget_bias=cfg['initial_forget_gate_bias'],
                batch_first = True
            )
            
            '''
            decoder = unet_film_v3(
                input_dim   = 2,   #  todo  , 2 for concatenation             
                hidden_dim  = cfg['unet_nfeat'],     
                static_dim  = cfg.get('static_dim', 27),
                h_lstm_dim  = cfg['hidden_size']         
            ) # unet, unet_film, unet_film_v2, unet_film_v3todo
            '''
            
            decoder = unet_attention_film_v3(
                input_dim  = 1,                     # 1 streamflow (or runoff) channel
                hidden_dim = cfg["unet_nfeat"],     # e.g., 64
                static_dim = cfg.get("static_dim", 27),
                h_lstm_dim = cfg["hidden_size"],    # size of your LSTM/temb vector
                dropout    = cfg.get("dropout", 0.1),
                attn_heads = cfg.get("attn_heads", 4),
            ) # unet_attention, unet_attention_film_v2, unet_attention_film_v3 todo
            
            model = EncoderDecoderDiffusionWrapper(
                encoder = encoder,
                decoder = decoder,
                hidden_size = cfg['hidden_size'],
                time_emb_dim = cfg['time_emb_dim'],
                decoder_name = 'unet'
            ).to(cfg['DEVICE'])
            
        elif cfg['model_name'] == 'diffusion_ssm':
            # build a single GenericLSTM and wrap it as encoder + decoder
            '''
            encoder = GenericLSTM(
                input_size = in_size_dyn,
                hidden_size = cfg['hidden_size'],
                num_layers = cfg.get('lstm_nlayers',1),
                dropout = cfg['dropout'],
                init_forget_bias=cfg['initial_forget_gate_bias'],
                batch_first = True
            )
            
            '''
            # todo, ssm encoder instead of lstm
            encoder = ssm_encoder(
                d_input    = in_size_dyn, 
                d_model    = cfg['d_model'],
                n_layers   = cfg['n_layers'],
                cfg        = cfg,
                static_dim = cfg.get('static_dim', 27),
                dropout    = cfg['ssm_dropout'],
                pool_type  = cfg['pool_type']
            )
            
            decoder = ssm_v2(
                d_input = 1, 
                d_model = cfg['d_model'],
                n_layers = cfg['n_layers'],
                cfg = cfg,                     
                time_emb_dim = cfg['time_emb_dim'],
                enc_hidden_dim = cfg['d_model'],
                static_dim = cfg.get('static_dim', 27),
                dropout = cfg['ssm_dropout'],
                time_full    = False, # False to set embedding at step 0 only. True to set condition for each step.
                enc_full     = False,
                static_full  = False
            )    # todo, ssm_v1, ssm_v2
            model = EncoderDecoderDiffusionWrapper(
                encoder = encoder,
                decoder = decoder,
                hidden_size = cfg['hidden_size'],
                time_emb_dim = cfg['time_emb_dim'],
                decoder_name = 'ssm'
            ).to(cfg['DEVICE'])     
            
            
        elif cfg['model_name'] == 'diffusion_ssm_unet':
            encoder = ssm_encoder(
                d_input    = in_size_dyn, 
                d_model    = cfg['d_model'],
                n_layers   = cfg['n_layers'],
                cfg        = cfg,
                static_dim = cfg.get('static_dim', 27),
                dropout    = cfg['ssm_dropout'],
            )
            decoder = unet_attention_film_v2(
                input_dim  = 1,                     # 1 streamflow (or runoff) channel
                hidden_dim = cfg["unet_nfeat"],     # e.g., 64
                static_dim = cfg.get("static_dim", 27),
                h_lstm_dim = cfg["hidden_size"],    # size of your LSTM/temb vector
                dropout    = cfg.get("dropout", 0.1),
                attn_heads = cfg.get("attn_heads", 4),
            ) # unet_attention, unet_attention_film_v2, unet_attention_film_v3 #todo
            model = EncoderDecoderDiffusionWrapper(
                encoder = encoder,
                decoder = decoder,
                hidden_size = cfg['hidden_size'],
                time_emb_dim = cfg['time_emb_dim'],
                decoder_name = 'unet'
            ).to(cfg['DEVICE'])      
            
            
        elif cfg['model_name'] == 'diffusion_ssm_lstm':
            encoder = ssm_encoder(
                d_input    = in_size_dyn, 
                d_model    = cfg['d_model'],
                n_layers   = cfg['n_layers'],
                cfg        = cfg,
                static_dim = cfg.get('static_dim', 27),
                dropout    = cfg['ssm_dropout'],
            )
            decoder = GenericLSTM(
                input_size = 2, # x_t is scalar per time-step # todo
                hidden_size = cfg['hidden_size'],
                num_layers = cfg.get('lstm_nlayers',1),
                dropout = cfg['dropout'],
                init_forget_bias=cfg['initial_forget_gate_bias'],
                batch_first =True
            )
            model = EncoderDecoderDiffusionWrapper(
                encoder = encoder,
                decoder = decoder,
                hidden_size = cfg['hidden_size'],
                time_emb_dim = cfg['time_emb_dim'],
                decoder_name = 'lstm'
            ).to(cfg['DEVICE'])   
                 
    else:
        model = _build_det_model(cfg, device)

    #model.load_state_dict(torch.load(model_pt, map_location=device)) # todo, original eval
    
    # ------------------------------------------------------------------ #
    # restore checkpoint  (raw + EMA) todo, eval with ema
    # ------------------------------------------------------------------ #
    ckpt = torch.load(model_pt, map_location=device)
    
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:        
        model.load_state_dict(ckpt)
        
    # -- EMA buffer (optional) -----------------------------------------
    ema = None
    if isinstance(ckpt, dict) and "ema" in ckpt:
        from torch_ema import ExponentialMovingAverage
        ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
        ema.load_state_dict(ckpt["ema"])

    model.eval()
    
    

    # --- data loader ---
    basins_all = get_basin_list(cfg["camels_root"])
    test_ds    = CamelsH5(
        test_h5,
        basins_all,
        str(db_path),
        concat_static    = cfg["concat_static"],
        no_static        = cfg["no_static"],
        model_name       = cfg["model_name"],
        forecast_horizon = cfg["forecast_horizon"],
        include_dates    = True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = cfg.get("batch_size", 256),
        shuffle     = False,
        num_workers = cfg["num_workers"]
    )

    # --- evaluation storage ---
    all_preds, all_tgts, all_basin_ids, all_dates, all_ens = [], [], [], [], []

    def _denorm(arr, b_ids):
        if PER_BASIN_NORM:
            m = torch.tensor([train_stats[b]["mean"] for b in b_ids], device=device).unsqueeze(1)
            s = torch.tensor([train_stats[b]["std"]  for b in b_ids], device=device).unsqueeze(1)
        else:
            m = torch.tensor(SCALER["QObs(mm/d)_mean"], device=device)
            s = torch.tensor(SCALER["QObs(mm/d)_std"],  device=device)
        return arr * s + m
        
        
    context_mgr = ema.average_parameters() if ema is not None else torch.no_grad()
    with context_mgr:
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="batches"):
                if cfg["no_static"]:
                    x_d, y_t, q_m, q_s, basin_batch, date_batch = batch
                    static_attrs = None
                else:
                    x_d, static_attrs, y_t, q_m, q_s, basin_batch, date_batch = batch
                    static_attrs = static_attrs.to(device)
    
                x_d = x_d.to(device)
                y_t = y_t.to(device)
                fh  = cfg["forecast_horizon"]
                B   = x_d.size(0)
    
                def denorm(a): return _denorm(a, basin_batch)
                
                # for diffusion models
                if is_diffusion:
                    x_past = x_d[:, :-fh, :]
                    future_prec = x_d[:, -fh:, :] # forcing choice, todo
                    if (not cfg['no_static']) and cfg['concat_static']:
                        stat_p = static_attrs.unsqueeze(1).repeat(1, x_past.size(1), 1)  # [batch, seq_len, 27]
                        stat_f = static_attrs.unsqueeze(1).repeat(1, future_prec.size(1), 1)
                        x_past = torch.cat([x_past, stat_p], dim=-1)
                    
                    # wrapper’s built-in sampler
                    ens = []
                    for _ in range(cfg.get("num_samples",50)):
                        samp = model.sample_ddim(
                            x_past      = x_past,
                            static_attributes = stat_f, 
                            future_pcp  = future_prec,
                            num_steps   = cfg.get("ddim_steps",10),
                            eta         = 0.0
                        )  # returns (B, H)
                        ens.append(samp)
                    ens  = torch.stack(ens, dim=0)    # (S,B,H)
                    ens_d= denorm(ens)                # de-norm
                    preds= ens_d.mean(dim=0)          # (B,H)
                    all_ens.append(ens_d.cpu().numpy().transpose(1,0,2))
    
                else:
                    if cfg["model_name"] == "seq2seq_lstm":
                        x_full = x_d
                        if not cfg["no_static"] and cfg["concat_static"]:
                            stat = static_attrs.unsqueeze(1).repeat(1,x_full.size(1),1)
                            x_full = torch.cat([x_full, stat], dim=-1)
                        preds = model(x_full).squeeze(-1)
    
                    elif cfg["model_name"] == "encdec_lstm":
                        x_past      = x_d[:, :-fh, :]
                        future_prec = x_d[:, -fh:, 2:3] # forcing choice, todo
                        if not cfg["no_static"] and cfg["concat_static"]:
                            stat_p = static_attrs.unsqueeze(1).repeat(1, x_past.size(1),     1)
                            stat_f = static_attrs.unsqueeze(1).repeat(1, future_prec.size(1), 1)
                            x_past      = torch.cat([x_past,      stat_p], dim=-1)
                            future_prec = torch.cat([future_prec, stat_f], dim=-1)
                        preds = model(x_past, future_prec, None).squeeze(-1)
    
                    else:  # seq2seq_ssm
                        x_full = x_d
                        if not cfg["no_static"] and cfg["concat_static"]:
                            static_exp = static_attrs.unsqueeze(1).repeat(1,x_full.size(1),1)
                            x_full = torch.cat([x_full, static_exp], dim=-1)
                        preds_all = model(x_full)
                        preds     = preds_all[:, -fh:, :].squeeze(-1)
    
                    preds = denorm(preds)
                    all_ens.append(None)
    
                all_preds.append(preds.cpu().numpy())
                all_tgts .append(y_t.cpu().numpy())
                all_basin_ids.extend(basin_batch)
                all_dates.extend(pd.to_datetime(date_batch))

    # --- save ---
    preds_arr = np.vstack(all_preds)
    tgts_arr  = np.vstack(all_tgts)
    bas       = np.array(all_basin_ids, dtype=object)
    dts       = np.array(all_dates,    dtype=object)
    ens_arr   = None if all_ens[0] is None else np.vstack(all_ens)

    # squeeze last dim if needed
    if preds_arr.ndim==3 and preds_arr.shape[-1]==1:
        preds_arr = preds_arr.squeeze(-1)
    if tgts_arr.ndim==3  and tgts_arr.shape[-1] ==1:
        tgts_arr  = tgts_arr.squeeze(-1)
    '''
    np.savez_compressed(
        cache,
        preds       = preds_arr,
        basin_ids   = bas,
        timestamps  = dts,
        all_members = ens_arr,
        targets     = tgts_arr,
    )
    print(f"[INFO] Saved raw arrays to {cache}")
    '''
    
    # per-basin DataFrames (same as before)
    per_basin = {}
    for i,b in enumerate(bas):
        per_basin.setdefault(b, {"dates": [], "preds": [], "obs": []})
        per_basin[b]["dates"].append(dts[i])
        per_basin[b]["preds"].append(preds_arr[i])
        per_basin[b]["obs"].append(tgts_arr[i])

    dfs = {}
    for b,d in per_basin.items():
        obs_mat  = np.vstack(d["obs"])
        pred_mat = np.vstack(d["preds"])
        dates    = pd.to_datetime(d["dates"])
        Hcol     = pred_mat.shape[1]
        cols     = ["q_obs_t+1"] + [f"qsim_t+{i+1}" for i in range(Hcol)]
        dfs[b]   = pd.DataFrame(
            np.hstack([obs_mat[:,:1], pred_mat]),
            index=dates, columns=cols
        )

    pkl_out = run_dir / "test_results.pkl" # todo
    with open(pkl_out, "wb") as f:
        pickle.dump(dfs, f)
    print(f"[INFO] Saved per-basin DataFrames to {pkl_out}")
