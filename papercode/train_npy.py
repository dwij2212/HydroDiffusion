import torch
import numpy as np
import json
import pickle
from pathlib import Path, PosixPath
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, LambdaLR
import torch.optim as optim
from typing import Dict, List
import random
import torch.nn.functional as F
import torch.nn as nn
import pdb
import sys
from multiprocessing import get_context

from papercode.datasets_npy import CamelsNPY
from papercode.nseloss import NSELoss
from diffusers.optimization import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from torch_ema import ExponentialMovingAverage

#lstm
from papercode.lstm import Seq2SeqLSTM, EncoderDecoderDetLSTM # deterministic
from papercode.SSM_test import HOPE, setup_optimizer
from papercode.backbones.lstm import GenericLSTM
from papercode.decoder_only_lstm import decoder_only_lstm # diffusion

# unet
from papercode.backbones.unet_attention_film_v2 import unet_attention_film_v2
from papercode.backbones.unet_attention_film_v3 import unet_attention_film_v3

# ssm
from papercode.backbones.ssm_v1 import ssm_v1
from papercode.backbones.ssm_v2 import ssm_v2
from papercode.backbones.ssm_encoder import ssm_encoder
from papercode.decoder_only_ssm import decoder_only_ssm # diffusion
from papercode.seq2seq_ssm import seq2seq_ssm # deterministic


from papercode.diffusion_wrapper import EncoderDecoderDiffusionWrapper
from papercode.diffusion_utils import diffusion_params
import os, gc

from papercode.datasets_npy import (
    load_npy_data, compute_normalization, compute_per_basin_q_stats, CamelsNPY
)

RAW_DIR = '/projects/standard/kumarv/renga/Public/DATA/camels_us_531/RAW'

def _make_loader(dataset, batch_size, shuffle, num_workers):
    nw = int(num_workers)
    kw = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
        drop_last=False,
        multiprocessing_context=get_context("fork"), 
    )
    if nw > 0:
        kw["prefetch_factor"] = 1
    return DataLoader(**kw)
    
def _close_loader(loader):
    try:
        it = getattr(loader, "_iterator", None)
        if it is not None:
            it._shutdown_workers() 
            loader._iterator = None
    except Exception:
        pass

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)


def _setup_run(cfg: Dict) -> Dict:
    if cfg.get("run_dir") is not None:
        # Resume mode: reuse existing directory
        base = Path(cfg["run_dir"])
        print(f"Resuming run at: {base}")
    else:
        # Fresh run: create new timestamped directory
        now = datetime.now().strftime("%d%m_%H%M")
        run_name = f"run_{now}_{cfg['model_name']}_nosf_seed{cfg['seed']}"
        base = Path(__file__).resolve().parent.parent / "runs" / run_name

    (base / "data" / "train").mkdir(parents=True, exist_ok=True)
    (base / "data" / "val").mkdir(parents=True, exist_ok=True)
    (base / "data" / "test").mkdir(parents=True, exist_ok=True)

    cfg["run_dir"] = base
    cfg["train_dir"] = base / "data" / "train"
    cfg["val_dir"] = base / "data" / "val"
    cfg["test_dir"] = base / "data" / "test"

    with open(base / "cfg.json", "w") as f:
        json.dump({k: str(v) for k,v in cfg.items()}, f, indent=4)

    return cfg

def _prepare_data(cfg: Dict) -> Dict:
    print('=== Step 1: Load raw data ===')
    data, dates, basins = load_npy_data(
        npy_path=os.path.join(RAW_DIR, 'data.npy'),
        dates_path=os.path.join(RAW_DIR, 'dates.npy'),
        basin_list_path=os.path.join(RAW_DIR, 'Basin_List.npy'),
    )

    print()
    print('=== Step 2: Compute normalization from training period ===')
    scalar = compute_normalization(data, dates)
    for k, v in scalar.items():
        print(f'  {k}: {v}')

    print()
    print('=== Step 3: Compute per-basin Q stats ===')
    q_means, q_stds = compute_per_basin_q_stats(data, dates)
    print(f'q_means: {q_means.shape}, q_stds: {q_stds.shape}')
    print(f'q_means range: [{q_means.min():.4f}, {q_means.max():.4f}]')
    print(f'q_stds  range: [{q_stds.min():.4f}, {q_stds.max():.4f}]')

    cfg['data'] = data
    cfg['dates'] = dates
    cfg['basins'] = basins
    cfg['scalar'] = scalar
    cfg['q_means'] = q_means
    cfg['q_stds'] = q_stds

    return cfg


def train(cfg):
    import json
    from pathlib import Path

    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])

    cfg = _setup_run(cfg)
    cfg = _prepare_data(cfg)

    data = cfg['data']
    dates = cfg['dates']
    basins = cfg['basins']
    scalar = cfg['scalar']
    q_means = cfg['q_means']
    q_stds = cfg['q_stds']
                        
    print()
    print('=== Step 4: Build TRAIN dataset ===')

    train_start = cfg['train_start']
    train_end   = cfg['train_end']
    val_start   = cfg['val_start']
    val_end     = cfg['val_end']
    train_ds = CamelsNPY(
        data=data, dates=dates, basins=basins,
        scalar=scalar, q_means=q_means, q_stds=q_stds,
        split_start=train_start, split_end=train_end,
        seq_length=365, forecast_horizon=8,
        stride=1,
        concat_static=True, no_static=False,
        include_dates=False, is_train=True,
    )
    print(f'Train samples: {len(train_ds)}')

    print()
    print('=== Step 5: Build VAL dataset ===')
    val_ds = CamelsNPY(
        data=data, dates=dates, basins=basins,
        scalar=scalar, q_means=q_means, q_stds=q_stds,
        split_start=val_start, split_end=val_end,
        stride=1,
        seq_length=365, forecast_horizon=8,
        concat_static=True, no_static=False,
        include_dates=False, is_train=True,
    )
    print(f'Val samples: {len(val_ds)}')

    print("BATCH size:", cfg['batch_size'])
    
    train_loader = _make_loader(train_ds, cfg['batch_size'], True,  cfg['num_workers'])
    val_loader   = _make_loader(val_ds,   cfg['batch_size'], False, cfg['num_workers'])

    try:
        print("loaded data.")
        
        device = torch.device(cfg['DEVICE'])
        torch.cuda.manual_seed_all(cfg['seed'])
    
        model = _build_model(cfg).to(device)
        model.apply(init_weights)
        ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
        
        # param count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{total_params/1e6:.1f}M params ({trainable_params/1e6:.1f}M trainable)")
    
        # — compute total & warmup steps (in **batches**)
        total_steps  = len(train_loader) * cfg["epochs"]
        warmup_steps = len(train_loader) * cfg["warmup"]

        # split params of ssm into groups
        optimizer = setup_optimizer(      
            model,
            lr = cfg["lr"], 
            weight_decay  = cfg["weight_decay"],
            epochs        = cfg["epochs"],   
            warmup_epochs = cfg["warmup"],        
        )
        
    
        # cosine schedule with linear warm-up
        scheduler = get_linear_schedule_with_warmup( # linear schedule
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        
        def lr_schedule(epoch):
            if epoch <= 10:
                return 1.0                    # 1e-3
            elif epoch <= 25:
                return 0.5                    # 5e-4
            else:
                return 0.1                    # 1e-4
        
        optimizer_lstm = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=cfg["weight_decay"])
        scheduler_lstm = LambdaLR(optimizer_lstm, lr_lambda=lambda e: lr_schedule(e))
        
        print("warmup_steps, total_steps:", warmup_steps, total_steps)
    
        print("LR @   0:", optimizer.param_groups[0]['lr'])
        print("LR @ warmup_steps:", optimizer.param_groups[0]['lr'] * scheduler.lr_lambdas[0](warmup_steps))
        print("LR @ total_steps//2:", optimizer.param_groups[0]['lr'] * scheduler.lr_lambdas[0](total_steps//2))
        print("LR @ total_steps:", optimizer.param_groups[0]['lr'] * scheduler.lr_lambdas[0](total_steps))
            
        loss_fn = torch.nn.MSELoss() if cfg['use_mse'] else NSELoss()
    
        best_val = float('inf')
        patience_ctr = 0
        train_losses, val_losses = [], []
        start_epoch = 1

        loss_log_path = cfg['run_dir'] / "loss_history.json"
        loss_log = {"train_loss": [], "val_loss": []}

        # --- checkpoint resume ---
        ckpt_path = cfg['run_dir'] / "checkpoint.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            ema.load_state_dict(ckpt["ema"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if "optimizer_lstm" in ckpt:
                optimizer_lstm.load_state_dict(ckpt["optimizer_lstm"])
                scheduler_lstm.load_state_dict(ckpt["scheduler_lstm"])
            best_val = ckpt["best_val"]
            patience_ctr = ckpt["patience_ctr"]
            start_epoch = ckpt["epoch"] + 1
            loss_log = ckpt.get("loss_log", loss_log)
            print(f"Resumed from checkpoint at epoch {ckpt['epoch']}, best_val={best_val:.6f}")

        for epoch in range(start_epoch, cfg['epochs'] + 1):
            if cfg['model_name'] in ['diffusion_lstm', 'diffusion_unet', 'diffusion_ssm', 'decoder_only_ssm', 'decoder_only_lstm', 'diffusion_ssm_unet', 'diffusion_ssm_lstm']:
                train_loss = train_diffusion_epoch(cfg, model, optimizer, scheduler, train_loader, epoch, ema)
            elif cfg['model_name'] in ['seq2seq_ssm', 'seq2seq_lstm', 'encdec_lstm']:
                if cfg['model_name'] == 'seq2seq_ssm':
                    train_loss = train_epoch(cfg, model, optimizer, scheduler, loss_fn, train_loader, epoch, ema)
                else:
                    train_loss = train_epoch(cfg, model, optimizer_lstm, scheduler_lstm, loss_fn, train_loader, epoch, ema)
                    scheduler_lstm.step()

    
            train_losses.append(train_loss)
            loss_log["train_loss"].append(train_loss)
            tqdm.write(f"Epoch {epoch} TRAINING loss: {train_loss:.6f}")
            
            state = {
                  "model": model.state_dict(),
                  "ema":   ema.state_dict(),
                  "epoch": epoch
              }

            if cfg['model_name'] in ['diffusion_lstm', 'diffusion_unet', 'diffusion_ssm', 'decoder_only_ssm', 'decoder_only_lstm', 'diffusion_ssm_unet', 'diffusion_ssm_lstm']:
                val_loss = validate_diffusion_epoch(cfg, model, val_loader, epoch, ema)
            else:
                val_loss = validate_epoch(cfg, model, val_loader, loss_fn, epoch, ema)
            val_losses.append(val_loss)
            loss_log["val_loss"].append(val_loss)
            
            active_opt = optimizer_lstm if cfg['model_name'] in ['seq2seq_lstm', 'encdec_lstm'] else optimizer
            tqdm.write(f"Current learning rate: {active_opt.param_groups[0]['lr']:.6g}")

            if val_loss < best_val:
                best_val = val_loss
                patience_ctr = 0
                torch.save(state, cfg['run_dir'] / "best_model.pt")
                tqdm.write("New best model saved.")
            else:
                patience_ctr += 1
                tqdm.write(f"No improvement. Patience: {patience_ctr}/10")
                if patience_ctr >= 15:
                    tqdm.write("Early stopping triggered.")
                    break

            # Save checkpoint every epoch
            ckpt = {
                "epoch":         epoch,
                "model":         model.state_dict(),
                "ema":           ema.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "scheduler":     scheduler.state_dict(),
                "optimizer_lstm": optimizer_lstm.state_dict(),
                "scheduler_lstm": scheduler_lstm.state_dict(),
                "best_val":      best_val,
                "patience_ctr":  patience_ctr,
                "loss_log":      loss_log,
            }
            torch.save(ckpt, cfg['run_dir'] / "checkpoint.pt")

            # Dynamically write log after each epoch
            with open(loss_log_path, "w") as f:
                json.dump(loss_log, f, indent=2)
    
        print(f"Final loss log saved to: {loss_log_path}")
        
    finally:
        _close_loader(train_loader)
        _close_loader(val_loader)
        del train_loader, val_loader
        gc.collect()

def train_diffusion_epoch(cfg, model, optimizer, scheduler, loader, epoch, ema):
    model.train()
    total_loss, total_n = 0.0, 0
    fh = cfg['forecast_horizon']
    pbar = tqdm(loader,
            desc=f"Epoch {epoch} [Diffusion]",
            dynamic_ncols=True)

    for batch in pbar:
        if cfg['no_static']:
            x_d, y_norm, *_ = batch
            static_attrs = None
        else:
            x_d, static_attrs, y_norm, *_ = batch
            static_attrs = static_attrs.unsqueeze(1).to(cfg['DEVICE'])

        x_d, y_norm = x_d.to(cfg['DEVICE']), y_norm.to(cfg['DEVICE'])
        
        nldas_idx  = [0,  3,  6,  9, 12]
        maurer_idx = [1,  4,  7, 10, 13]
        daymet_idx = [2,  5,  8, 11, 14]
        idx_map = {
            'nldas':  nldas_idx,
            'maurer': maurer_idx,
            'daymet': daymet_idx
        }
        
        which = cfg['forcing_source']
        
        if cfg['forcing_source'] != 'all':
            x_d = x_d[:, :, idx_map[which]] 

        x_past = x_d[:, :-fh, :]
        
        if cfg['model_name'] in ['decoder_only_ssm','decoder_only_lstm']:
            future_precip = x_d[:, -fh+1:, :]
        else:
            future_precip = x_d[:, -fh:, :]
            
        # ======================================= 
        # todo, slice out the nowcast y_norm
        #y_norm = y_norm[:,1:]
        # ======================================= 

        if (not cfg['no_static']) and cfg['concat_static'] and cfg['model_name'] not in ['decoder_only_ssm','decoder_only_lstm']:
            stat_p = static_attrs.expand(-1, x_past.size(1), static_attrs.size(-1))  # [batch, seq_len, 27]
            #stat_f = static_attrs.repeat(1, future_precip.size(1), 1)
            x_past = torch.cat([x_past, stat_p], dim=-1)
        stat_f = static_attrs.repeat(1, future_precip.size(1), 1)
            
        B = y_norm.size(0)

        # 1) sample a diffusion time
        t = torch.rand(B, device=cfg['DEVICE'])  # continuous [0,1]

        # 2) noise and diffuse
        eps = torch.randn_like(y_norm)
        _, alpha, sigma = diffusion_params(t)
        x_t = (alpha * y_norm + sigma * eps).unsqueeze(-1)  # (B, H, 1)
        # 3) predict
        out = model(x_past, x_t, t, future_precip, stat_f)
        
        # 4) velocity loss
        target = (alpha * eps - sigma * y_norm).unsqueeze(-1)
    
        loss = F.mse_loss(out, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_value'])
        
        optimizer.step()
        scheduler.step() 
        
        ema.update()
        
        total_loss += loss.item() * B
        total_n    += B
        
    print(f"Epoch {epoch} TRAIN lo: {total_loss / total_n:.6f}")
    return total_loss / total_n

@torch.no_grad()
def validate_diffusion_epoch(cfg, model, loader, epoch, ema):
    model.eval()
    total_loss, total_n = 0.0, 0
    fh = cfg['forecast_horizon']
    pbar = tqdm(loader,
            desc=f"Epoch {epoch} [Diffusion]",
            dynamic_ncols=True)
    
    with ema.average_parameters():
        for batch in pbar:
            # 1) unpack & move to device
            if cfg['no_static']:
                x_d, y_norm, *_ = batch
                static_attrs = None
            else:
                x_d, static_attrs, y_norm, *_ = batch
                static_attrs = static_attrs.unsqueeze(1).to(cfg['DEVICE'])
    
            x_d = x_d.to(cfg['DEVICE'])
            y_norm = y_norm.to(cfg['DEVICE'])
            
            nldas_idx  = [0,  3,  6,  9, 12]
            maurer_idx = [1,  4,  7, 10, 13]
            daymet_idx = [2,  5,  8, 11, 14]
            idx_map = {
                'nldas':  nldas_idx,
                'maurer': maurer_idx,
                'daymet': daymet_idx
            }
            
            which = cfg['forcing_source']
            
            if cfg['forcing_source'] != 'all':
                x_d = x_d[:, :, idx_map[which]] 
            
            # 2) split
            x_past = x_d[:, :-fh, :]
                    
            if cfg['model_name'] in ['decoder_only_ssm','decoder_only_lstm']:
                future_precip = x_d[:, -fh+1:, :]
            else:
                future_precip = x_d[:, -fh:, :]
                
            # ======================================= 
            # todo, slice out the nowcast y_norm
            #y_norm = y_norm[:,1:]
            # ======================================= 
                
            if (not cfg['no_static']) and cfg['concat_static'] and cfg['model_name'] not in ['decoder_only_ssm','decoder_only_lstm']:
                stat_p = static_attrs.expand(-1, x_past.size(1), static_attrs.size(-1))  # [batch, seq_len, 27]
                x_past = torch.cat([x_past, stat_p], dim=-1)
            stat_f = static_attrs.repeat(1, future_precip.size(1), 1)
                
            B = y_norm.size(0)
    
            # 3) sample continuous t and noise exactly as in training
            t    = torch.rand(B, device=cfg['DEVICE'])
            eps  = torch.randn_like(y_norm)
            _, alpha, sigma = diffusion_params(t)
    
            # 4) form the noised target
            x_t = (alpha * y_norm + sigma * eps).unsqueeze(-1)   # (B, H, 1)
    
            # 5) forward pa feed x_t into the model
            out = model(x_past, x_t, t, future_precip, stat_f)
    
            # 6) compute the same velocity-target lo
            v_target = alpha * eps - sigma * y_norm
            v_target = v_target.unsqueeze(-1)
            
            if cfg.get('predict_mode','velocity') == 'velocity':
                target = v_target
            else:
                target = eps.unsqueeze(-1)
        
            loss = F.mse_loss(out, target)
    
            total_loss += loss.item() * B
            total_n += B

    avg_loss = total_loss / total_n
    print(f"Epoch {epoch} VALIDATION lo: {avg_loss:.6f}")
    return avg_loss
    
    
    
# ------------------------------------------------------------------ #
#  TRAINING LOOP (deterministic models)                              #
# ------------------------------------------------------------------ #
def train_epoch(cfg, model, optimizer, scheduler, loss_fn, loader, epoch, ema):
    """
    One epoch of training for deterministic models.
    Returns the average loss over the entire epoch.
    """
    import torch, sys
    model.train()
    fh = cfg['forecast_horizon']

    total_loss, total_n = 0.0, 0    
    pbar = tqdm(loader,
        desc=f"Epoch {epoch} [Deterministic]",
        dynamic_ncols=True)

    for batch in pbar:
        # -------- flexible unpack ---------------------------------------
        # batch order: x, [attrs], y, q_means, q_stds, basin, date
        if cfg['no_static']:
            x, y, *rest = batch
            static_attrs = None
        else:
            x, static_attrs, y, *rest = batch
        # rest[0]=q_means, rest[1]=q_stds
        q_stds = rest[1] if (len(rest) > 1 and torch.is_tensor(rest[1])) else None

        # -------- move tensors to device --------------------------------
        to_dev = lambda t: t.to(cfg['DEVICE']) if torch.is_tensor(t) else t
        x, y = map(to_dev, (x, y))
        if static_attrs is not None: static_attrs = to_dev(static_attrs)
        if q_stds is not None: q_stds = to_dev(q_stds)
        
        
        nldas_idx  = [0,  3,  6,  9, 12]
        maurer_idx = [1,  4,  7, 10, 13]
        daymet_idx = [2,  5,  8, 11, 14]
        idx_map = {
            'nldas':  nldas_idx,
            'maurer': maurer_idx,
            'daymet': daymet_idx
        }
        
        which = cfg['forcing_source']
        
        if cfg['forcing_source'] != 'all':
            x = x[:, :, idx_map[which]] 
        
        # fallback stds for NSE
        #if (not cfg['use_mse']) and (q_stds is None):
        #    q_stds = torch.ones_like(y, device=cfg['DEVICE'])

        # -------- forward pa ------------------------------------------
        optimizer.zero_grad()

        # ---- model-specific inputs -------------------------------------
        x_past = x[:, :-fh, :]
        if cfg['model_name'] == 'encdec_lstm':
            future_prec = x[:, -fh:, :]
        elif cfg['model_name'] in ['seq2seq_ssm', 'seq2seq_lstm']:
            future_prec = x[:, -fh+1:, :]
        if (not cfg['no_static']) and cfg['concat_static']:
            stat_p = static_attrs.unsqueeze(1).repeat(1, x_past.size(1),     1)
            stat_f = static_attrs.unsqueeze(1).repeat(1, future_prec.size(1), 1)
            x_past = torch.cat([x_past, stat_p], dim=-1)
            future_prec = torch.cat([future_prec, stat_f], dim=-1)
        preds = model(x_past, future_prec,
                      None if cfg['concat_static'] else static_attrs)
    
        # -------- align targets to preds -------------------------------
        if preds.dim() == 3:                         # (B,S,1)
            S = preds.size(1)
            y_trg = y[:, -S:].unsqueeze(-1)          # (B,S,1)
        else:                                        # (B,1)
            y_trg = y[:, -1].unsqueeze(-1)

        # -------- loss_fn / back-prop --------------------------------------
        # Targets are z-scored (std≈1 per basin) so NSELoss weights should
        # all equal 1. Pass ones so NSELoss reduces to MSE in z-score space.
        loss = (loss_fn(preds, y_trg) if cfg['use_mse']
                else loss_fn(preds, y_trg, torch.ones_like(y_trg)))
        loss.backward()
        if cfg['clip_norm']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_value'])
        optimizer.step()
        
        if cfg['model_name'] in ['seq2seq_ssm']:
            scheduler.step() # per-batch scheduler
        
        ema.update()
        # accumulate for epoch average
        B = y_trg.size(0)
        total_loss += loss.item() * B
        total_n += B
    avg_loss = total_loss / total_n
    print(f"Epoch {epoch} TRAIN lo: {avg_loss:.6f}")
    return avg_loss


# ------------------------------------------------------------------ #
#  VALIDATION LOOP (deterministic models)                            #
# ------------------------------------------------------------------ #
@torch.no_grad()
def validate_epoch(cfg, model, loader, loss_fn, epoch, ema):
    import torch
    model.eval()
    fh = cfg['forecast_horizon']
    total_loss, total_n = 0.0, 0
    pbar = tqdm(loader,
        desc=f"Epoch {epoch} [Deterministic]",
        dynamic_ncols=True)

    with ema.average_parameters():
        for batch in pbar:
            # -------- flexible unpack ---------------------------------------
            # batch order: x, [attrs], y, q_means, q_stds, basin, date
            if cfg['no_static']:
                x, y, *rest = batch
                static_attrs = None
            else:
                x, static_attrs, y, *rest = batch
            # rest[0]=q_means, rest[1]=q_stds
            q_stds = rest[1] if (len(rest) > 1 and torch.is_tensor(rest[1])) else None

            to_dev = lambda t: t.to(cfg['DEVICE']) if torch.is_tensor(t) else t
            x, y = map(to_dev, (x, y))
            if static_attrs is not None: static_attrs = to_dev(static_attrs)
            if q_stds is not None: q_stds = to_dev(q_stds)
            if (not cfg['use_mse']) and (q_stds is None):
                q_stds = torch.ones_like(y, device=cfg['DEVICE'])
            
            nldas_idx  = [0,  3,  6,  9, 12]
            maurer_idx = [1,  4,  7, 10, 13]
            daymet_idx = [2,  5,  8, 11, 14]
            idx_map = {
                'nldas':  nldas_idx,
                'maurer': maurer_idx,
                'daymet': daymet_idx
            }
            
            which = cfg['forcing_source']
            
            if cfg['forcing_source'] != 'all':
                x = x[:, :, idx_map[which]] 
    
            # -------- forward ------------------------------------------------
            x_past = x[:, :-fh, :]
            if cfg['model_name'] == 'encdec_lstm':
                future_prec = x[:, -fh:, :]
            elif cfg['model_name'] in ['seq2seq_ssm','seq2seq_lstm']:
                future_prec = x[:, -fh+1:, :]
            if (not cfg['no_static']) and cfg['concat_static']:
                stat_p = static_attrs.unsqueeze(1).repeat(1, x_past.size(1),     1)
                stat_f = static_attrs.unsqueeze(1).repeat(1, future_prec.size(1), 1)
                x_past = torch.cat([x_past, stat_p], dim=-1)
                future_prec = torch.cat([future_prec, stat_f], dim=-1)
            preds = model(x_past, future_prec,
                          None if cfg['concat_static'] else static_attrs)
    
            # -------- align targets -----------------------------------------
            if preds.dim() == 3:                        # (B,S,1)
                S = preds.size(1)
                y_trg = y[:, -S:].unsqueeze(-1)
            else:                                       # (B,1)
                y_trg = y[:, -1].unsqueeze(-1)

            # Targets are z-scored so pass unit weights to NSELoss
            loss = (loss_fn(preds, y_trg) if cfg['use_mse']
                    else loss_fn(preds, y_trg, torch.ones_like(y_trg)))
            B = y.size(0)
            total_loss += loss.item() * B
            total_n += B

    avg = total_loss / total_n
    print(f"Epoch {epoch} VALIDATION loss: {avg:.6f}")
    return avg


def _build_model(cfg: Dict):
    if cfg['forcing_source'] == 'all':
        dyn_in = 15 
        input_size_dyn = dyn_in if (cfg['no_static'] or not cfg['concat_static']) else 42
    else:
        dyn_in = 5
        input_size_dyn = dyn_in if (cfg['no_static'] or not cfg['concat_static']) else 32
    static_size = 0 if cfg['no_static'] else (input_size_dyn - dyn_in)

    if cfg['model_name'] == 'seq2seq_lstm':   
        return Seq2SeqLSTM(
            input_size = input_size_dyn,
            hidden = cfg['hidden_size'],
            horizon = cfg['forecast_horizon'],
            dropout = cfg['dropout'],
        ).to(cfg['DEVICE'])

    elif cfg['model_name'] == 'encdec_lstm':
        return EncoderDecoderDetLSTM(
            past_features = dyn_in,
            future_features = dyn_in, 
            static_size = static_size,
            horizon = cfg['forecast_horizon'],
            hidden = cfg['hidden_size'],
            dropout = cfg['dropout']
        ).to(cfg['DEVICE'])
        
    elif cfg['model_name'] == 'seq2seq_ssm':
        model = seq2seq_ssm(
            d_input      = input_size_dyn, 
            d_model      = cfg['d_model'],     
            n_layers     = cfg['n_layers'],      
            cfg          = cfg,                 
            horizon      = cfg['forecast_horizon'],
            time_emb_dim = cfg['time_emb_dim'],
            static_dim   = cfg.get('static_dim', 27),
            dropout      = cfg['ssm_dropout']
        ).to(cfg['DEVICE'])
        return model


    elif cfg['model_name'] == 'diffusion_lstm':
        encoder = GenericLSTM(
            input_size = input_size_dyn,
            hidden_size = cfg['hidden_size'],
            dropout = cfg['dropout'],
            init_forget_bias=cfg['initial_forget_gate_bias'],
            batch_first=True
        )
        decoder = GenericLSTM(
            input_size = 1+dyn_in+27, # x_t is scalar per time-step # todo 2 for x_t+futurue prcp 1 source, 4 for x_t+3 future prcp!!
            hidden_size = cfg['hidden_size'],
            dropout = cfg['dropout'],
            init_forget_bias=cfg['initial_forget_gate_bias'],
            batch_first =True
        )
        model = EncoderDecoderDiffusionWrapper(
            encoder = encoder,
            decoder = decoder,
            hidden_size = cfg['hidden_size'],
            time_emb_dim = cfg['time_emb_dim'],
            decoder_name = 'lstm',
            prediction_type =  cfg['predict_mode']
        ).to(cfg['DEVICE'])
        return model
        
        
        
    elif cfg['model_name'] == 'diffusion_unet':
        encoder = GenericLSTM(
            input_size = input_size_dyn,
            hidden_size = cfg['hidden_size'],
            num_layers = cfg.get('lstm_nlayers',1),
            dropout = cfg['dropout'],
            initial_forget_bias=cfg['initial_forget_gate_bias'],
            batch_first = True
        )
        
        '''
        decoder = unet_film_v3(
            input_dim   = 2, # todo, for concatenation test                   
            hidden_dim  = cfg['unet_nfeat'],     
            static_dim  = cfg.get('static_dim', 27),
            h_lstm_dim  = cfg['hidden_size']         
        ) # unet, unet_film, unet_film_v2, unet_film_v3# todo
        
        '''
        decoder = unet_attention_film_v3(
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
            decoder_name = 'unet',
            prediction_type =  cfg['predict_mode']
        ).to(cfg['DEVICE'])
        return model
        
    elif cfg['model_name'] == 'diffusion_ssm':
        '''
        encoder = GenericLSTM(
            input_size = input_size_dyn,
            hidden_size = cfg['hidden_size'],
            num_layers = cfg.get('lstm_nlayers',1),
            dropout = cfg['dropout'],
            init_forget_bias=cfg['initial_forget_gate_bias'],
            batch_first = True
        )
        '''
        # todo, m encoder instead of lstm
        encoder = ssm_encoder(
            d_input    = input_size_dyn, 
            d_model    = cfg['d_model'],
            n_layers   = cfg['n_layers'],
            cfg        = cfg,
            static_dim = cfg.get('static_dim', 27),
            dropout    = cfg['ssm_dropout'],
            pool_type  = cfg['pool_type'],
            horizon = cfg['forecast_horizon'] # todo, test without nowcast!
        )
        
        decoder = ssm_v2(
            d_input = 1+dyn_in+27, # todo!! x_t concat all [future metr, static]
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
        )    
        model = EncoderDecoderDiffusionWrapper(
            encoder = encoder,
            decoder = decoder,
            hidden_size = cfg['hidden_size'],
            time_emb_dim = cfg['time_emb_dim'],
            decoder_name = 'ssm',
            prediction_type =  cfg['predict_mode']
        ).to(cfg['DEVICE'])        
        return model
        
    elif cfg['model_name'] == 'decoder_only_ssm':
        model = decoder_only_ssm(
            d_input      = input_size_dyn, 
            d_model      = cfg['d_model'],       
            n_layers     = cfg['n_layers'],      
            cfg          = cfg,                  # lr, d_state, dt_min, etc.
            horizon      = cfg['forecast_horizon'],
            time_emb_dim = cfg['time_emb_dim'],
            static_dim   = cfg.get('static_dim', 27),
            dropout      = cfg['ssm_dropout'],
            time_full    = True
        ).to(cfg['DEVICE'])
        return model
        
    elif cfg['model_name'] == 'decoder_only_lstm':
        model = decoder_only_lstm(
            d_input      = input_size_dyn, 
            hidden_size  = cfg['hidden_size'],
            cfg          = cfg,                
            horizon      = cfg['forecast_horizon'],
            time_emb_dim = cfg['time_emb_dim'],
            static_dim   = cfg.get('static_dim', 27),
            #time_full    = True
        ).to(cfg['DEVICE'])
        return model
        
        
    elif cfg['model_name'] == 'diffusion_ssm_unet':
        encoder = ssm_encoder(
            d_input    = input_size_dyn, 
            d_model    = cfg['d_model'],
            n_layers   = cfg['n_layers'],
            cfg        = cfg,
            static_dim = cfg.get('static_dim', 27),
            dropout    = cfg['ssm_dropout'],
            pool_type  = cfg['pool_type '],
            horizon = cfg['forecast_horizon']
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
            decoder_name = 'unet',
            prediction_type =  cfg['predict_mode']
        ).to(cfg['DEVICE'])        
        return model
        
        
    elif cfg['model_name'] == 'diffusion_ssm_lstm':
        encoder = ssm_encoder(
            d_input    = input_size_dyn, 
            d_model    = cfg['d_model'],
            n_layers   = cfg['n_layers'],
            cfg        = cfg,
            static_dim = cfg.get('static_dim', 27),
            dropout    = cfg['ssm_dropout'],
            pool_type  = cfg['pool_type '],
            horizon = cfg['forecast_horizon']
        )
        decoder = GenericLSTM(
            input_size = 16, # x_t is scalar per time-step # todo
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
            decoder_name = 'lstm',
            prediction_type =  cfg['predict_mode']
        ).to(cfg['DEVICE'])        
        return model




