import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.s4.s4d import S4D as LTI
import pdb

# ---------------------------------------------------------------------------
# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d

class ssm_v3(nn.Module):
    def __init__(self, d_input, d_model, n_layers, cfg, *,
                 time_emb_dim=256, enc_hidden_dim=256, static_dim=27, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # --- your existing projektion & bias modules ---
        self.input_proj = nn.Linear(d_input, d_model)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim*2),
            nn.SiLU(),
            nn.Linear(time_emb_dim*2, d_model),
        )
        self.enc_mlp   = nn.Linear(enc_hidden_dim, d_model)
        self.static_fc = nn.Linear(static_dim,    d_model)

        # we no longer need a pcp-to-token linear; we’ll just broadcast the raw prcp
        # self.pcp_mlp = nn.Linear(1, d_model)   ? REMOVE this line

        # residual S4D stack (unchanged)
        self.blocks, self.norms, self.drops = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(
                LTI(d_model,
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
            # choose the right dropout version
            self.drops.append(dropout_fn(dropout))

        # read-out head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model//2, 1),
        )

    def forward(self, x_past, t, h_enc, future_pcp, static_attr):
        # x_past:   (B, L, d_input)
        # future_pcp:(B, H, 1)
        B, L, _ = x_past.shape
        H       = future_pcp.shape[1]
        # (in your setup H==L; if not you may need to up/down-sample)

        # 1) project your history into model-space
        h = self.input_proj(x_past)  # (B, L, d_model)

        # 2) build all of your additive biases (shape ? (B, L, d_model))
        #  2a) time, for ini step only
        t_emb     = self.time_mlp(t)              # (B, d_model)
        time_bias = t_emb.unsqueeze(1).expand(-1, L, -1)

        #  2b) encoder state
        if h_enc.dim() == 3:  # take top-layer if stacked
            h_enc = h_enc[0]
        enc_emb   = self.enc_mlp(h_enc)           # (B, d_model)
        enc_bias  = enc_emb.unsqueeze(1).expand(-1, L, -1)

        #  2c) static
        static_emb  = self.static_fc(static_attr) # (B, d_model)
        static_bias = static_emb.unsqueeze(1).expand(-1, L, -1)

        #  2d) precipitation: just repeat the raw prcp along the model dim
        #     future_pcp is (B, H, 1) ? we broadcast to (B, H, d_model)
        pcp_bias = future_pcp.expand(-1, -1, self.d_model)

        # 3) add them all in one go
        h = h + time_bias + enc_bias + static_bias + pcp_bias

        # 4) feed through your S4D residual stack
        h = h.transpose(1, 2)  # (B, d_model, L)
        for blk, norm, drop in zip(self.blocks, self.norms, self.drops):
            z, _ = blk(h)       # (B, d_model, L)
            h    = norm(h + drop(z))

        # 5) read out one velocity per time step
        data_tok = h.transpose(1, 2)  # (B, L, d_model)
        v_pred   = self.head(data_tok) # (B, L, 1)
        return v_pred
