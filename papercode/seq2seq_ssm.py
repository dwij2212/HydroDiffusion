import torch
import torch.nn as nn
from models.s4.s4d import S4D as LTI
import pdb
from diffusion_utils import diffusion_params
import math

dropout_fn = nn.Dropout1d if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12) else nn.Dropout

class seq2seq_ssm(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_model: int,
        n_layers: int,
        cfg: dict,
        *,
        horizon: int = 8,
        time_emb_dim: int = 256,
        static_dim: int = 27,
        dropout: float = 0.15
    ):
        super().__init__()
        self.d_model   = d_model
        self.H         = horizon
        self.static_dim = static_dim
        
        # 1) project raw input (met + noisy flow + static) to state dim
        self.input_proj = nn.Linear(d_input, d_model)

        # 3) S4D residual stack
        self.blocks, self.norms, self.drops = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(
                LTI(
                    d_model,
                    dropout    = dropout,
                    transposed = True,
                    lr         = min(cfg["lr_min"], cfg["lr"]),
                    d_state    = cfg["d_state"],
                    dt_min     = cfg["min_dt"],
                    dt_max     = cfg["max_dt"],
                    lr_dt      = cfg["lr_dt"],
                    cfr        = cfg["cfr"],
                    cfi        = cfg["cfi"],
                    wd         = cfg["wd"],
                )
            )
            self.norms.append(nn.BatchNorm1d(d_model))
            self.drops.append(dropout_fn(dropout))

        # 4) final head: map state to noise prediction
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        x_past:       torch.Tensor,  # (B, L, d_input)
        x_future:     torch.Tensor,  # (B, H-1, d_input), -1 is for excluding the nowcast day
        static_attr:  torch.Tensor,  # (B, static_dim)
    ) -> torch.Tensor:
        B, L, _ = x_past.shape
        H        = self.H
        device   = x_past.device


        # Build feature sequence: met & flow & static
        all_met    = torch.cat([x_past, x_future], dim=1)               # (B, L+H-1, d_input)
        #static_seq = static_attr[:,0,:].unsqueeze(1).expand(-1, L+H-1, -1)   
        if static_attr:
          static_seq = static_attr.unsqueeze(1).expand(-1, L+H-1, -1)
          feats = torch.cat([all_met, static_seq], dim=-1) # (B, L+H-1, d_input)
        else:
          feats = all_met

        # Project to state dimension
        h = self.input_proj(feats)  # (B, L+H-1, d_model)
        
        # Residual S4D stack
        h = h.transpose(1, 2)  # (B, d_model, L+H-1)
        for blk, norm, drop in zip(self.blocks, self.norms, self.drops):
            z, _ = blk(h)
            h    = norm(h + drop(z))
        h = h.transpose(1, 2)  # (B, L+H-1, d_model)

        # Predict only the future window
        out = self.head(h)          # (B, L+H-1, 1)
        return out[:, -H:, :]       # (B, H, 1)