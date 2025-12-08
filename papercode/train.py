import torch
import numpy as np
import json
import pickle
from pathlib import Path, PosixPath
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from typing import Dict, List
import random
import torch.nn.functional as F
import torch.nn as nn
import pdb
import sys

from papercode.datasets import CamelsH5
from papercode.nseloss import NSELoss
from papercode.lstm import Seq2SeqLSTM, EncoderDecoderDetLSTM
from papercode.SSM_test import HOPE, setup_optimizer, Seq2SeqSSM
from papercode.diffusion_lstm import EncoderDecoderDiffusionLSTM, EncoderDiffusionLSTM, DecoderDiffusionLSTM
from papercode.diffusion_utils import diffusion_params
from papercode.utils import get_basin_list, create_h5_files
from papercode.datautils import add_camels_attributes

'''
def debug_diffusion_step(x_past, y_norm, noise, eps_theta, t, x_t):
    print("\n===== DEBUG DIFFUSION STEP =====")
    print(f"Loss (MSE): {F.mse_loss(eps_theta, noise).item():.6f}")
    print(f"eps mean: {eps_theta.mean().item():.4f}, std: {eps_theta.std().item():.4f}")
    print(f"noise mean: {noise.mean().item():.4f}, std: {noise.std().item():.4f}")
    print(f"x_t mean: {x_t.mean().item():.4f}, std: {x_t.std().item():.4f}")
    print(f"y_norm mean: {y_norm.mean().item():.4f}, std: {y_norm.std().item():.4f}")
    print(f"x_past mean: {x_past.mean().item():.4f}, std: {x_past.std().item():.4f}")
    print(f"t[:5]: {t[:5].tolist()}")
    print("y_norm[0]:", y_norm[0].detach().cpu().numpy())
    print("noise[0]:", noise[0].detach().cpu().numpy())
    print("eps_theta[0]:", eps_theta[0].detach().cpu().numpy())
    print("x_past[0]:", x_past[0].detach().cpu().numpy())
    print("===============================\n")
    
def print_gradient_norms(model):
    print("\n===== GRADIENT NORMS =====")
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            print(f"{name} grad norm: {norm:.6f}")
    print("=========================\n")
'''

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

def train(cfg):
    import json
    from pathlib import Path

    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])

    print("Preparing training basins...")
    if cfg.get('split_file'):
        with open(cfg['split_file'], 'rb') as f:
            splits = pickle.load(f)
        basins = splits[cfg['split']]['train']
    else:
        basins = get_basin_list(cfg['camels_root'])

    print(f"Loaded {len(basins)} basins.")
    cfg = _setup_run(cfg)
    cfg = _prepare_data(cfg, basins)
                        
    train_ds = CamelsH5(
        h5_file          = cfg['train_file'],
        basins           = basins,
        db_path          = cfg['db_path'],
        concat_static    = cfg['concat_static'],
        no_static        = cfg['no_static'],
        model_name       = cfg['model_name'],
        forecast_horizon = cfg['forecast_horizon'],
        include_dates    = False,
        cache            = True
    )
    
    val_ds = CamelsH5(
        h5_file          = cfg['val_file'],
        basins           = basins,
        db_path          = cfg['db_path'],
        concat_static    = cfg['concat_static'],
        no_static        = cfg['no_static'],
        model_name       = cfg['model_name'],
        forecast_horizon = cfg['forecast_horizon'],
        include_dates    = False,
        cache            = True
    )

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

    model = _build_model(cfg)
    model.apply(init_weights)
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=1e-4)
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    optimizer, scheduler = setup_optimizer(model, lr=cfg["lr"], weight_decay=cfg["weight_decay"], epochs=cfg["epochs_scheduler"], warmup_epochs=cfg["warmup"]) 
    
    loss_fn = torch.nn.MSELoss() if cfg['use_mse'] else NSELoss()

    best_val = float('inf')
    patience_ctr = 0
    train_losses, val_losses = [], []

    loss_log_path = cfg['run_dir'] / "loss_history.json"
    loss_log = {"train_loss": [], "val_loss": []}

    for epoch in range(1, cfg['epochs'] + 1):
        if cfg['model_name'] == 'diffusion_lstm':
            train_loss = train_diffusion_epoch(cfg, model, optimizer, train_loader, epoch)
        else:
            train_loss = train_epoch(cfg, model, optimizer, loss_fn, train_loader, epoch)

        train_losses.append(train_loss)
        loss_log["train_loss"].append(train_loss)
        print(f"Epoch {epoch} TRAINING loss: {train_loss:.6f}")

        torch.save(model.state_dict(), cfg['run_dir'] / f"model_epoch{epoch}.pt")

        if epoch % 1 == 0 or epoch == cfg['epochs']:    
            if cfg['model_name'] == "diffusion_lstm":
                val_loss = validate_diffusion_epoch(cfg, model, val_loader, epoch)
            else:
                val_loss = validate_epoch(cfg, model, val_loader, loss_fn, epoch)
            val_losses.append(val_loss)
            loss_log["val_loss"].append(val_loss)
            print(f"Validation loss: {val_loss:.6f}")
            
            scheduler.step(val_loss)
            print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6g}")

            if val_loss < best_val:
                best_val = val_loss
                patience_ctr = 0
                torch.save(model.state_dict(), cfg['run_dir'] / "best_model.pt")
                print("New best model saved.")
            else:
                patience_ctr += 1
                print(f"No improvement. Patience: {patience_ctr}/10")
                if patience_ctr >= 10 : # todo
                    print("Early stopping triggered.")
                    break

        # Dynamically write log after each epoch
        with open(loss_log_path, "w") as f:
            json.dump(loss_log, f, indent=2)

    print(f"Final loss log saved to: {loss_log_path}")

def train_diffusion_epoch(cfg, model, optimizer, loader, epoch):
    model.train()
    total_loss, total_n = 0.0, 0
    fh = cfg['forecast_horizon']
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Diffusion]", 
                disable=not sys.stdout.isatty())

    for batch in pbar:
        # unpack
        if cfg['no_static']:
            x_d, y_norm, *_     = batch
            static_attrs        = None
        else:
            x_d, static_attrs, y_norm, *_ = batch
            static_attrs        = static_attrs.to(cfg['DEVICE'])

        x_d, y_norm = x_d.to(cfg['DEVICE']), y_norm.to(cfg['DEVICE'])

        x_past        = x_d[:, :-fh, :]
        future_precip = x_d[:, -fh:, 0:1]  # (B, H, 1)
        B             = y_norm.size(0)

        # 1) sample a diffusion time
        t = torch.rand(B, device=cfg['DEVICE'])  # continuous [0,1]

        # 2) noise and diffuse
        eps        = torch.randn_like(y_norm)
        _, alpha, sigma = diffusion_params(t)
        x_t        = (alpha * y_norm + sigma * eps).unsqueeze(-1)  # (B, H, 1)

        # 3) predict velocity
        v_pred     = model(x_past, x_t, t, static_attrs, future_precip)

        # 4) velocity target & loss
        v_target = (alpha * eps - sigma * y_norm)
        v_target = v_target.unsqueeze(-1)
        loss       = F.mse_loss(v_pred, v_target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_value'])
        optimizer.step()

        total_loss += loss.item() * B
        total_n    += B
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    return total_loss / total_n

@torch.no_grad()
def validate_diffusion_epoch(cfg, model, loader, epoch):
    model.eval()
    total_loss, total_n = 0.0, 0
    fh = cfg['forecast_horizon']
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Validation]", 
                disable=not sys.stdout.isatty())

    for batch in pbar:
        # 1) unpack & move to device
        if cfg['no_static']:
            x_d, y_norm, *_ = batch
            static_attrs   = None
        else:
            x_d, static_attrs, y_norm, *_ = batch
            static_attrs = static_attrs.to(cfg['DEVICE'])

        x_d     = x_d.to(cfg['DEVICE'])
        y_norm  = y_norm.to(cfg['DEVICE'])

        # 2) split context vs. forcing
        x_past       = x_d[:, :-fh, :]
        future_precip= x_d[:, -fh:, 0:1]
        B            = y_norm.size(0)

        # 3) sample continuous t and noise exactly as in training
        t    = torch.rand(B, device=cfg['DEVICE'])
        eps  = torch.randn_like(y_norm)
        _, alpha, sigma = diffusion_params(t)

        # 4) form the noised target
        x_t  = (alpha * y_norm + sigma * eps).unsqueeze(-1)   # (B, H, 1)

        # 5) forward pass—feed x_t into the model
        v_pred = model(x_past, x_t, t, static_attrs, future_precip)

        # 6) compute the same velocity-target loss
        v_target = alpha * eps - sigma * y_norm
        v_target = v_target.unsqueeze(-1)
        
        loss     = F.mse_loss(v_pred, v_target)

        total_loss += loss.item() * B
        total_n    += B

    avg_loss = total_loss / total_n
    print(f"Epoch {epoch} VALIDATION loss: {avg_loss:.6f}")
    return avg_loss
    
    
    
# ------------------------------------------------------------------ #
#  TRAINING LOOP (deterministic models)                              #
# ------------------------------------------------------------------ #
def train_epoch(cfg, model, optimizer, loss_fn, loader, epoch):
    """
    One epoch of training for deterministic models.
    Returns the average loss over the entire epoch.
    """
    import torch, sys
    model.train()
    fh = cfg['forecast_horizon']

    total_loss, total_n = 0.0, 0    
    pbar = tqdm(loader, desc=f"Epoch {epoch}", disable=not sys.stdout.isatty())

    for batch in pbar:
        # -------- flexible unpack ---------------------------------------
        if cfg['no_static']:
            x, y, *rest = batch
            static_attrs = None
        else:
            x, static_attrs, y, *rest = batch
        q_stds = rest[0] if (rest and torch.is_tensor(rest[0])) else None

        # -------- move tensors to device --------------------------------
        to_dev = lambda t: t.to(cfg['DEVICE']) if torch.is_tensor(t) else t
        x, y   = map(to_dev, (x, y))
        if static_attrs is not None: static_attrs = to_dev(static_attrs)
        if q_stds is not None:       q_stds       = to_dev(q_stds)

        # fallback stds for NSE
        if (not cfg['use_mse']) and (q_stds is None):
            q_stds = torch.ones_like(y, device=cfg['DEVICE'])

        # -------- forward pass ------------------------------------------
        optimizer.zero_grad()

        # ---- model-specific inputs -------------------------------------
        if cfg['model_name'] == 'seq2seq_lstm':
            x_full = x                            
            if (not cfg['no_static']) and cfg['concat_static']:
                stat_exp = static_attrs.unsqueeze(1).repeat(1, x_full.size(1), 1)
                x_full   = torch.cat([x_full, stat_exp], dim=-1)
        
            preds = model(x_full)                        # (B, H, 1)

        elif cfg['model_name'] == 'encdec_lstm':
            x_past      = x[:, :-fh, :]
            future_prec = x[:, -fh:, 0:1]
            if (not cfg['no_static']) and cfg['concat_static']:
                stat_p = static_attrs.unsqueeze(1).repeat(1, x_past.size(1),     1)
                stat_f = static_attrs.unsqueeze(1).repeat(1, future_prec.size(1), 1)
                x_past      = torch.cat([x_past,      stat_p], dim=-1)
                future_prec = torch.cat([future_prec, stat_f], dim=-1)
            preds = model(x_past, future_prec,
                          None if cfg['concat_static'] else static_attrs)

        elif cfg['model_name'] == 'seq2seq_ssm':
            # 1) keep *all* 365 + H rows
            x_full = x                      # shape (B, 365+H, Din)
            # 2) optionally concatenate statics to every time-step
            if (not cfg['no_static']) and cfg['concat_static']:
                stat_exp  = static_attrs.unsqueeze(1)          # (B,1,S)
                stat_exp  = stat_exp.repeat(1, x_full.size(1), 1)
                x_full    = torch.cat([x_full, stat_exp], dim=-1)
        
            # 3) forward pass through SSM
            preds_all = model(x_full)        # (B, 365+H, 1)
        
            # 4) keep only the horizon slice
            preds = preds_all[:, -fh:, :]    # (B, fh, 1)
    
        # -------- align targets to preds -------------------------------
        if preds.dim() == 3:                         # (B,S,1)
            S     = preds.size(1)
            y_trg = y[:, -S:].unsqueeze(-1)          # (B,S,1)
            q_trg = (q_stds[:, -S:].unsqueeze(-1)
                     if q_stds is not None else None)
        else:                                        # (B,1)
            y_trg = y[:, -1].unsqueeze(-1)
            q_trg = (q_stds[:, -1].unsqueeze(-1)
                     if q_stds is not None else None)

        # -------- loss / back-prop --------------------------------------
        loss = (loss_fn(preds, y_trg) if cfg['use_mse']
                else loss_fn(preds, y_trg, q_trg))
        loss.backward()
        if cfg['clip_norm']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_value'])
        optimizer.step()

        # accumulate for epoch average
        B = y_trg.size(0)
        total_loss += loss.item() * B
        total_n    += B
        pbar.set_postfix_str(f"Loss: {loss.item():.6f}")

    avg_loss = total_loss / total_n
    return avg_loss



# ------------------------------------------------------------------ #
#  VALIDATION LOOP (deterministic models)                            #
# ------------------------------------------------------------------ #
@torch.no_grad()
def validate_epoch(cfg, model, loader, loss_fn, epoch):
    import torch
    model.eval()
    fh = cfg['forecast_horizon']
    total_loss, total_n = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Validation]", disable=not sys.stdout.isatty())

    for batch in pbar:
        # -------- flexible unpack ---------------------------------------
        if cfg['no_static']:
            x, y, *rest = batch
            static_attrs = None
        else:
            x, static_attrs, y, *rest = batch
        q_stds = rest[0] if (rest and torch.is_tensor(rest[0])) else None

        to_dev = lambda t: t.to(cfg['DEVICE']) if torch.is_tensor(t) else t
        x, y   = map(to_dev, (x, y))
        if static_attrs is not None: static_attrs = to_dev(static_attrs)
        if q_stds is not None:       q_stds       = to_dev(q_stds)
        if (not cfg['use_mse']) and (q_stds is None):
            q_stds = torch.ones_like(y, device=cfg['DEVICE'])

        # -------- forward ------------------------------------------------
        if cfg['model_name'] == 'seq2seq_lstm':
            x_full = x                            
            if (not cfg['no_static']) and cfg['concat_static']:
                stat_exp = static_attrs.unsqueeze(1).repeat(1, x_full.size(1), 1)
                x_full   = torch.cat([x_full, stat_exp], dim=-1)
            preds = model(x_full) 

        elif cfg['model_name'] == 'encdec_lstm':
            x_past      = x[:, :-fh, :]
            future_prec = x[:, -fh:, 0:1]
            if (not cfg['no_static']) and cfg['concat_static']:
                stat_p = static_attrs.unsqueeze(1).repeat(1, x_past.size(1),     1)
                stat_f = static_attrs.unsqueeze(1).repeat(1, future_prec.size(1), 1)
                x_past      = torch.cat([x_past,      stat_p], dim=-1)
                future_prec = torch.cat([future_prec, stat_f], dim=-1)
            preds = model(x_past, future_prec,
                          None if cfg['concat_static'] else static_attrs)

        elif cfg['model_name'] == 'seq2seq_ssm':
            x_full = x                               # (B, 365+H, Din)
            # option-ally concatenate statics
            if (not cfg['no_static']) and cfg['concat_static']:
                stat_exp = static_attrs.unsqueeze(1)          # (B,1,S)
                stat_exp = stat_exp.repeat(1, x_full.size(1), 1)
                x_full   = torch.cat([x_full, stat_exp], dim=-1)
        
            # forward pass through the SSM
            preds_all = model(x_full)                # (B, 365+H, 1)
        
            # keep only horizon predictions
            preds = preds_all[:, -fh:, :]            # (B, H, 1)

        # -------- align targets -----------------------------------------
        if preds.dim() == 3:                        # (B,S,1)
            S     = preds.size(1)
            y_trg = y[:, -S:].unsqueeze(-1)
            q_trg = (q_stds[:, -S:].unsqueeze(-1)
                     if q_stds is not None else None)
        else:                                       # (B,1)
            y_trg = y[:, -1].unsqueeze(-1)
            q_trg = (q_stds[:, -1].unsqueeze(-1)
                     if q_stds is not None else None)

        loss = (loss_fn(preds, y_trg) if cfg['use_mse']
                else loss_fn(preds, y_trg, q_trg))
        B = y.size(0)
        total_loss += loss.item() * B
        total_n    += B

    avg = total_loss / total_n
    print(f"Epoch {epoch} VALIDATION loss: {avg:.6f}")
    return avg


def _build_model(cfg: Dict):
    dyn_in = 15 # todo, depend on whether it's multisources. If single source, set it to 5. 
    input_size_dyn = dyn_in if (cfg['no_static'] or not cfg['concat_static']) else 42
    static_size = 0 if cfg['no_static'] else (input_size_dyn - dyn_in)

    if cfg['model_name'] == 'seq2seq_lstm':   
        return Seq2SeqLSTM(
            in_size   = input_size_dyn,
            hidden    = cfg['hidden_size'],
            horizon   = cfg['forecast_horizon'],
            n_layers  = cfg.get('n_layers', 1),
            dropout   = cfg['dropout'],
        ).to(cfg['DEVICE'])

    elif cfg['model_name'] == 'encdec_lstm':
        return EncoderDecoderDetLSTM(
            past_features   = dyn_in,
            future_features = 1, 
            static_size     = static_size,
            hidden          = cfg['hidden_size'],
            horizon         = cfg['forecast_horizon'],
            dropout         = cfg['dropout']
        ).to(cfg['DEVICE'])

    elif cfg['model_name'] == 'seq2seq_ssm':
        hope = HOPE(
            d_input   = input_size_dyn,
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


    elif cfg['model_name'] == 'diffusion_lstm':
        return EncoderDecoderDiffusionLSTM(
            past_features    = input_size_dyn,
            static_size      = static_size,
            hidden_size      = cfg['hidden_size'],
            time_emb_dim     = cfg.get('time_emb_dim', 16),
            forecast_horizon = cfg['forecast_horizon'],
            dropout          = cfg['dropout']
        ).to(cfg['DEVICE'])


def _setup_run(cfg: Dict) -> Dict:
    now = datetime.now().strftime("%d%m_%H%M")
    run_name = f"run_{now}_seed{cfg['seed']}"
    base = Path(__file__).resolve().parent.parent / "runs" / run_name

    (base / "data" / "train").mkdir(parents=True, exist_ok=False)
    (base / "data" / "val").mkdir(parents=True, exist_ok=False)
    (base / "data" / "test").mkdir(parents=True, exist_ok=False)

    cfg["run_dir"]   = base
    cfg["train_dir"] = base / "data" / "train"
    cfg["val_dir"]   = base / "data" / "val"
    cfg["test_dir"]  = base / "data" / "test"

    with open(base / "cfg.json", "w") as f:
        json.dump({k: str(v) for k,v in cfg.items()}, f, indent=4)

    return cfg


def _prepare_data(cfg: Dict, basins: List[str]) -> Dict:
    # === 1) Set up paths
    run_dir = Path(cfg['run_dir'])
    cfg['db_path'] = str(run_dir / "attributes.db")

    if cfg.get('h5_dir') is not None:
        shared_dir = Path(cfg['h5_dir']) / "data"
        shared_db = Path(cfg['h5_dir']) / "attributes.db"
    else:
        shared_dir = None
        shared_db = None

    cfg['train_file'] = run_dir / 'data/train/train_data.h5'
    cfg['val_file']   = run_dir / 'data/val/val_data.h5'
    cfg['test_file']  = run_dir / 'data/test/test_data.h5'

    # === 2) Use shared attribute DB if available
    if shared_db and shared_db.exists():
        print("Using shared attribute DB from:", shared_db)
        cfg['db_path'] = str(shared_db)
        print(f"Run directory is: {run_dir}")
    else:
        if not Path(cfg['db_path']).exists():
            print("Creating new attribute DB...")
            add_camels_attributes(cfg['camels_root'], db_path=cfg['db_path'])
        else:
            print("Attribute DB already exists at:", cfg['db_path'])

    # === 3) Use shared HDF5 files if they exist
    if shared_dir:
        shared_train = shared_dir / "train/train_data.h5"
        shared_val   = shared_dir / "val/val_data.h5"
        shared_test  = shared_dir / "test/test_data.h5"

        if shared_train.exists() and shared_val.exists() and shared_test.exists():
            print("Using shared preprocessed HDF5 files from:", shared_dir)
            cfg['train_file'] = shared_train
            cfg['val_file']   = shared_val
            cfg['test_file']  = shared_test
            return cfg

    # === 4) Otherwise, create new local HDF5 files
    print("Shared HDF5 files not found. Generating new datasets...")
    create_h5_files(
        camels_root=cfg['camels_root'],
        out_file=cfg['train_file'],
        basins=basins,
        dates=[cfg['train_start'], cfg['train_end']],
        db_path=cfg['db_path'],
        model_name=cfg['model_name'],
        is_train=True,
        with_basin_str=True,
        seq_length=cfg['seq_length'],
        forecast_horizon=cfg['forecast_horizon']
    )
    create_h5_files(
        camels_root=cfg['camels_root'],
        out_file=cfg['val_file'],
        basins=basins,
        dates=[cfg['val_start'], cfg['val_end']],
        db_path=cfg['db_path'],
        model_name=cfg['model_name'],
        is_train=True,
        with_basin_str=True,
        seq_length=cfg['seq_length'],
        forecast_horizon=cfg['forecast_horizon']
    )
    create_h5_files(
        camels_root=cfg['camels_root'],
        out_file=cfg['test_file'],
        basins=basins,
        dates=[cfg['test_start'], cfg['test_end']],
        db_path=cfg['db_path'],
        model_name=cfg['model_name'],
        is_train=False,
        with_basin_str=True,
        seq_length=cfg['seq_length'],
        forecast_horizon=cfg['forecast_horizon'],
        include_dates = True
    )

    return cfg

