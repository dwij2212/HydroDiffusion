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
    
class ssm_v1(nn.Module):
    """
    S4D-based decoder that consumes
        • past history          x_past      : (B,L,d_input)
        • diffusion timestep    t           : (B,)
        • encoder hidden state  h_enc       : (B, enc_hidden_dim)  or (num_layers,B,enc_hidden_dim)
        • future precip         future_pcp  : (B,H,1)
        • static attributes     static_attr : (B,static_dim)
    and predicts a velocity sequence (B,1).
    """
    def __init__(
        self,
        d_input          : int,
        d_model          : int,
        n_layers         : int,
        cfg              : dict,      # S4D hyper-params
        *,
        time_emb_dim     : int  = 256,
        enc_hidden_dim   : int  = 256,
        static_dim       : int  = 27,
        dropout          : float = 0.10,
    ):
        super().__init__()

        # 1) projections / embeddings -------------------------------------------------
        self.input_proj = nn.Linear(d_input, d_model)

        self.time_mlp = nn.Sequential(                  # R^C → R^{d_model}
            nn.Linear(time_emb_dim, time_emb_dim*2),
            nn.SiLU(),
            nn.Linear(time_emb_dim*2, d_model),
        )

        self.pcp_mlp   = nn.Linear(1,            d_model)   # per-step precip token
        self.enc_mlp   = nn.Linear(enc_hidden_dim, d_model) # encoder-state token
        self.static_fc = nn.Linear(static_dim,     d_model) # global static bias

        # 2) residual S4D stack -------------------------------------------------------
        self.blocks, self.norms, self.drops = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(
                LTI(d_model,
                    dropout     = dropout,
                    transposed  = True,
                    lr          = min(cfg["lr_min"], cfg["lr"]),
                    d_state     = cfg["d_state"],
                    dt_min      = cfg["min_dt"],
                    dt_max      = cfg["max_dt"],
                    lr_dt       = cfg["lr_dt"],
                    cfr         = cfg["cfr"],
                    cfi         = cfg["cfi"],
                    wd          = cfg["wd"])
            )
            self.norms.append(nn.BatchNorm1d(d_model))
            self.drops.append(dropout_fn(dropout))

        # 3) read-out head ------------------------------------------------------------
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model//2, 1),
        )

    # -------------------------------------------------------------------------
    def forward(
        self,
        x_past      : torch.Tensor,   # (B,L,d_input)
        t           : torch.Tensor,   # (B,)
        h_enc       : torch.Tensor,   # (B,enc_hidden) or (layers,B,enc_hidden)
        future_pcp  : torch.Tensor,   # (B,H,1)
        static_attr : torch.Tensor,   # (B,static_dim)
    ) -> torch.Tensor:

        B, L, _ = x_past.shape
        H       = future_pcp.size(1)
        assert future_pcp.shape == (B,H,1),        "future_pcp must be (B,H,1)"
        assert static_attr.shape[0] == B,          "static_attr batch mismatch"

        # 1) embed past history  -----------------------------------------------------
        h = self.input_proj(x_past).transpose(1,2)               # (B,d_model,L)

        # 2) build *tokens* ----------------------------------------------------------
        t_token   = self.time_mlp(t).unsqueeze(2)       # (B,d_model,1)

        if h_enc.dim() == 3:                                     # (layers,B,H)->take top layer
            h_enc = h_enc[0]
        enc_token = self.enc_mlp(h_enc).unsqueeze(2)             # (B,d_model,1)

        p_tokens  = self.pcp_mlp(future_pcp).transpose(1,2)      # (B,d_model,H)

        # 3) full sequence  [time | enc | precip(H) | past(L)] -----------------------
        h = torch.cat([t_token, enc_token, p_tokens, h], dim=2)  # (B,d_model,2+H+L)

        # 4) add static bias everywhere ---------------------------------------------
        h = h + self.static_fc(static_attr).unsqueeze(-1)        # broadcast over sequence

        # 5) residual S4D stack ------------------------------------------------------
        for blk, norm, drop in zip(self.blocks, self.norms, self.drops):
            z, _ = blk(h)          # (B,d_model,seq_len)
            h    = norm(h + drop(z))

        # 6) drop non-data tokens & pool --------------------------------------------
        h = h[:,:, (2+H): ]        # keep only the past-tokens  -> (B,d_model,L)
        #h = h.mean(dim=2)          # global average pool       -> (B,d_model)
        #return self.head(h)        # (B,1)
        data_tok = h.transpose(1, 2)  # (B, H, d_model) , H=L in this case
        v_pred   = self.head(data_tok)       # (B, H, 1)
        return v_pred
