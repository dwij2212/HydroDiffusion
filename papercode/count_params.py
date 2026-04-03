#!/usr/bin/env python3
"""
Count parameters in the decoder_only_ssm model with config from train_slurm.sh
"""

import sys
import torch
import torch.nn as nn

# Add paths
sys.path.insert(0, '/users/6/mehta423/projects/HydroDiffusion')

from papercode.decoder_only_ssm import decoder_only_ssm

# Config from train_slurm.sh
cfg = {
    'd_model': 256,
    'd_state': 256,
    'n_layers': 6,
    'lr': 3e-5,
    'lr_min': 3e-6,
    'weight_decay': 0.00,
    'wd': 4e-5,
    'lr_dt': 0.001,
    'min_dt': 0.01,
    'max_dt': 0.1,
    'cfi': 10,
    'cfr': 10,
}

# Model hyperparameters
d_input = 6  # meteorological forcing dimensions (typical for hydrology)
d_model = cfg['d_model']
n_layers = cfg['n_layers']
horizon = 8
time_emb_dim = 256
static_dim = 27
dropout = 0.2

# Create model
model = decoder_only_ssm(
    d_input=d_input,
    d_model=d_model,
    n_layers=n_layers,
    cfg=cfg,
    horizon=horizon,
    time_emb_dim=time_emb_dim,
    static_dim=static_dim,
    dropout=dropout,
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n{'='*60}")
print(f"Model: decoder_only_ssm")
print(f"{'='*60}")
print(f"\nConfiguration:")
print(f"  d_model: {d_model}")
print(f"  d_state: {cfg['d_state']}")
print(f"  n_layers: {n_layers}")
print(f"  d_input: {d_input}")
print(f"  horizon: {horizon}")
print(f"  dropout: {dropout}")
print(f"\nParameter Counts:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Non-trainable parameters: {total_params - trainable_params:,}")

# Breakdown by component
print(f"\n{'='*60}")
print(f"Breakdown by Component:")
print(f"{'='*60}")

components = {
    'input_proj': model.input_proj,
    'time_mlp': model.time_mlp,
    'head': model.head,
}

for name, module in components.items():
    params = sum(p.numel() for p in module.parameters())
    print(f"  {name}: {params:,}")

print(f"\n  S4D blocks (6 blocks):")
for i, block in enumerate(model.blocks):
    params = sum(p.numel() for p in block.parameters())
    print(f"    block_{i}: {params:,}")

total_s4d = sum(sum(p.numel() for p in block.parameters()) for block in model.blocks)
print(f"    Total S4D: {total_s4d:,}")

print(f"\n  BatchNorm1d layers (6 layers):")
total_bn = sum(sum(p.numel() for p in norm.parameters()) for norm in model.norms)
print(f"    Total BatchNorm: {total_bn:,}")

print(f"\n{'='*60}")
print(f"Total: {total_params:,} parameters")
print(f"{'='*60}\n")
