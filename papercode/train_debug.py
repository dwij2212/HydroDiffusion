import torch
import numpy as np
import json
import pickle
from pathlib import Path, PosixPath
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List
import random
import torch.nn.functional as F
import pdb

from papercode.datasets import CamelsH5
from papercode.nseloss import NSELoss
from papercode.lstm import LSTM_Model
from papercode.mclstm_modifiedhydrology import MassConservingLSTM
from papercode.mclstm_mr_modifiedhydrology import MassConservingLSTM_MR
from papercode.diffusion_lstm import EncoderDecoderDiffusionLSTM
from papercode.diffusion_utils import get_beta_schedule, q_sample, compute_posterior_mean
from papercode.utils import get_basin_list, create_h5_files
from papercode.datautils import add_camels_attributes


def debug_diffusion_step(x_past, y_norm, noise, eps_theta, t, x_t):
    print("\n===== DEBUG DIFFUSION STEP =====")
    print(f"Loss (MSE): {F.mse_loss(eps_theta, noise).item():.6f}")
    print(f"eps mean: {eps_theta.mean().item():.4f}, std: {eps_theta.std().item():.4f}")
    print(f"noise mean: {noise.mean().item():.4f}, std: {noise.std().item():.4f}")
    print(f"x_t mean: {x_t.mean().item():.4f}, std: {x_t.std().item():.4f}")
    print(f"t[:5]: {t[:5].tolist()}")
    print("y_norm[0]:", y_norm[0].detach().cpu().numpy())
    print("noise[0]:", noise[0].detach().cpu().numpy())
    print("eps_theta[0]:", eps_theta[0].detach().cpu().numpy())
    print("===============================\n")

def train(cfg):
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

    train_ds = CamelsH5(cfg['train_file'], basins, cfg['db_path'],
                        cfg['concat_static'], cfg['cache_data'],
                        cfg['no_static'], cfg['model_name'],
                        cfg['forecast_horizon'])

    val_ds = CamelsH5(cfg['val_file'], basins, cfg['db_path'],
                      cfg['concat_static'], cfg['cache_data'],
                      cfg['no_static'], cfg['model_name'],
                      cfg['forecast_horizon'])

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

    model = _build_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    loss_fn = torch.nn.MSELoss() if cfg['use_mse'] else NSELoss()
    lr_schedule = {1: 1e-4, 101: 5e-5, 141: 1e-5, 181: 5e-6}
    betas = get_beta_schedule().to(cfg['DEVICE']) if cfg['model_name'] == 'diffusion_lstm' else None

    best_val = float('inf')
    patience_ctr = 0
    train_losses, val_losses = [], []

    for epoch in range(1, cfg['epochs'] + 1):
        if cfg['model_name'] == 'diffusion_lstm':
            train_loss = train_diffusion_epoch(cfg, model, optimizer, train_loader, epoch, betas)
            train_losses.append(train_loss)
        else:
            train_epoch(cfg, model, optimizer, loss_fn, train_loader, epoch)

        torch.save(model.state_dict(), cfg['run_dir'] / f"model_epoch{epoch}.pt")

        if epoch % 20 == 0 or epoch == cfg['epochs']:
            val_loss = validate_epoch(cfg, model, loss_fn, val_loader, betas, epoch)
            val_losses.append(val_loss)
            print(f"Validation loss: {val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                patience_ctr = 0
                torch.save(model.state_dict(), cfg['run_dir'] / "best_model.pt")
                print("New best model saved.")
            else:
                patience_ctr += 1
                print(f"No improvement. Patience: {patience_ctr}/{cfg['patience']}")
                if patience_ctr >= cfg['patience']:
                    print("Early stopping triggered.")
                    break
    # Save loss history to file
    loss_log = {
        "train_loss": train_losses,
        "val_loss": val_losses
    }
    with open(cfg['run_dir'] / "loss_history.json", "w") as f:
        json.dump(loss_log, f, indent=2)
    print(f"Saved training and validation loss history to: {cfg['run_dir']}/loss_history.json")

def train_diffusion_epoch(cfg, model, optimizer, loader, epoch, betas):
    model.train()
    total_loss, total_n = 0.0, 0
    fh = cfg['forecast_horizon']

    pbar = tqdm(loader, desc=f"Epoch {epoch} [DRUM Loss]")

    for batch_idx, batch in enumerate(pbar):
        if cfg['no_static']:
            x_d, y_norm, q_stds, *_ = batch
            static_attrs = None
        else:
            x_d, static_attrs, y_norm, q_stds, *_ = batch
            static_attrs = static_attrs.to(cfg['DEVICE'])

        x_d, y_norm, q_stds = [t.to(cfg['DEVICE']) for t in [x_d, y_norm, q_stds]]
        x_past = x_d[:, :-fh, :]
        future_precip = x_d[:, -fh:, 0:1]

        B, T = y_norm.size(0), betas.size(0)
        t = torch.randint(0, T, (B,), device=x_past.device)
        noise = torch.randn_like(y_norm)
        x_t = q_sample(y_norm, t, noise, betas)

        eps_theta = model(x_past, t, static_attrs, future_precip)
        loss = F.mse_loss(eps_theta, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        total_n += B

        # Live loss update
        pbar.set_postfix({'Loss': f'{loss.item():.6f}'})

        debug_diffusion_step(x_past, y_norm, noise, eps_theta, t, x_t)

    return total_loss / total_n

@torch.no_grad()
def validate_epoch(cfg, model, loss_fn, loader, betas, epoch):
    model.eval()
    total_loss, total_n = 0.0, 0
    fh = cfg['forecast_horizon']
    T = betas.size(0) if betas is not None else None

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Validation]", leave=False)

    for batch in pbar:
        if cfg['model_name'] == 'diffusion_lstm':
            if cfg['no_static']:
                x_d, y_norm, q_stds, *_ = batch
                static_attrs = None
            else:
                x_d, static_attrs, y_norm, q_stds, *_ = batch
                static_attrs = static_attrs.to(cfg['DEVICE'])

            x_d, y_norm, q_stds = [t.to(cfg['DEVICE']) for t in [x_d, y_norm, q_stds]]
            x_past = x_d[:, :-fh, :]
            future_precip = x_d[:, -fh:, 0:1]

            B = x_past.size(0)
            t = torch.randint(0, T, (B,), device=x_past.device)
            noise = torch.randn_like(y_norm)
            x_t = q_sample(y_norm, t, noise, betas)

            eps_theta = model(x_past, t, static_attrs, future_precip)
            loss = F.mse_loss(eps_theta, noise)
        else:
            if cfg['no_static']:
                x, y, q_stds, _ = batch
                static_attrs = None
            else:
                x, static_attrs, y, q_stds, _ = batch
                static_attrs = static_attrs.to(cfg['DEVICE'])
            x, y, q_stds = [t.to(cfg['DEVICE']) for t in [x, y, q_stds]]
            preds = model(x)[0] if cfg['model_name'] == 'lstm' else model(x[..., :1], x[..., 1:])[0][:, :, 1:].sum(-1)[:, -1:]
            loss = loss_fn(preds, y) if cfg['use_mse'] else loss_fn(preds, y, q_stds)

        total_loss += loss.item() * x_d.size(0)
        total_n += x_d.size(0)
        pbar.set_postfix({'ValLoss': f'{loss.item():.6f}'})

    avg_loss = total_loss / total_n
    print(f"Epoch {epoch} VALIDATION loss: {avg_loss:.6f}")
    return avg_loss

def train_epoch(cfg, model, optimizer, loss_fn, loader, epoch):
    # for lstm, mclstm, mcrlstm
    model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        if cfg['no_static']:
            x, y, q_stds, basins = batch
        else:
            x, static_attrs, y, q_stds, basins = batch

        tensors = [x, y, q_stds]
        if not cfg['no_static']:
            tensors.insert(1, static_attrs)

        tensors = [t.to(cfg['DEVICE']) if isinstance(t, torch.Tensor) else t for t in tensors]

        if cfg['no_static']:
            x, y, q_stds = tensors
        else:
            x, static_attrs, y, q_stds = tensors

        optimizer.zero_grad()

        if cfg['model_name'] == 'lstm':
            preds = model(x)[0]
        else:
            xm, xa = x[..., :1], x[..., 1:]
            m_out, *_ = model(xm, xa)
            preds = m_out[:, :, 1:].sum(-1, keepdim=True)[:, -1, :]

        loss = loss_fn(preds, y) if cfg['use_mse'] else loss_fn(preds, y, q_stds)
        loss.backward()
        if cfg['clip_norm']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_value'])
        
        optimizer.step()
        pbar.set_postfix_str(f"Loss: {loss.item():.6f}")


def _build_model(cfg: Dict):
    dyn_in = 5
    input_size_dyn = dyn_in if (cfg['no_static'] or not cfg['concat_static']) else 32
    static_size = 0 if cfg['no_static'] else (input_size_dyn - dyn_in)

    if cfg['model_name'] == 'lstm':
        return LSTM_Model(
            input_size_dyn, cfg['hidden_size'],
            cfg['initial_forget_gate_bias'], cfg['dropout'],
            cfg['concat_static'], cfg['no_static']
        ).to(cfg['DEVICE'])

    elif cfg['model_name'] == 'mclstm':
        return MassConservingLSTM(
            1, input_size_dyn - 1, cfg['hidden_size'], False, True
        ).to(cfg['DEVICE'])

    elif cfg['model_name'] == 'mcrlstm':
        return MassConservingLSTM_MR(
            1, input_size_dyn - 1, cfg['hidden_size'], False, True
        ).to(cfg['DEVICE'])

    elif cfg['model_name'] == 'diffusion_lstm':
        return EncoderDecoderDiffusionLSTM(
            past_features    = dyn_in,
            future_precip    = 1,
            static_size      = static_size,
            hidden_size      = cfg['hidden_size'],
            time_emb_dim     = cfg.get('time_emb_dim', 16),
            forecast_horizon = cfg['forecast_horizon']
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
        is_train=False,
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
        forecast_horizon=cfg['forecast_horizon']
    )

    return cfg

