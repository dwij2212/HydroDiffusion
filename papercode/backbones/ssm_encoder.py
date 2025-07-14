import torch
import torch.nn as nn
from models.s4.s4d import S4D as LTI
import pdb

# ---------------------------------------------------------------------------
# Dropout selection (as before)
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
elif tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d

class ssm_encoder(nn.Module):
    """
    Acts like your GenericLSTM encoder:
      * forward(x, None, static_attr) ? (outputs, (h_enc, None))
    where
      - x:        (B, 365, d_input)
      - static:   (B, static_dim)
      - outputs:  full sequence embedding (B, 365, d_model) [optional downstream]
      - h_enc:    final-step embedding (1, B, d_model)
    """
    def __init__(
        self,
        d_input: int,
        d_model: int,
        n_layers: int,
        cfg: dict,
        *,
        static_dim: int = 27,
        dropout: float = 0.1,
        pool_type: str = 'avg',        # 'avg', 'power', or 'attn'
        power_exponent: float = 2.0     # used only if pool_type=='power'
    ):
        super().__init__()
        self.d_model = d_model
        
        self.pool_type = pool_type
        self.power_exponent = power_exponent

        # If using attention pooling, define a scorer head
        if pool_type == 'attn':
            self.attn_scorer = nn.Linear(d_model, 1)
        

        # 1) project dynamic inputs
        self.input_proj = nn.Linear(d_input, d_model)

        # 2) build S4D residual stack
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
            self.drops.append(dropout_fn(dropout))

    def forward(
        self,
        x: torch.Tensor,                # (B, 365, d_input)
        init_state: None = None,        # unused—kept for API consistency
        static_attr: torch.Tensor = None  # (B, static_dim)
    ):
        B, L, _ = x.shape
        device  = x.device

        # 1) concatenate static to every time-step if provided
        if static_attr is not None:
            S = static_attr.size(1)
            stat = static_attr.unsqueeze(1).expand(-1, L, -1)  # (B,365,S)
            x    = torch.cat([x, stat], dim=-1)                # (B,365,d_in+S)
        # 2) project into model space
        h = self.input_proj(x)  # (B, 365, d_model)

        # 3) S4D residual stack
        h = h.transpose(1,2)     # ? (B, d_model, 365)
        for blk, norm, drop in zip(self.blocks, self.norms, self.drops):
            z, _ = blk(h)        # (B, d_model, 365)
            h    = norm(h + drop(z))
        h = h.transpose(1,2)     # (B, 365, d_model)
        
        # pooling
        if self.pool_type == 'avg':
            pooled = h.mean(dim=1)               # (B, d_model)

        elif self.pool_type == 'power':
            # weights ? (t/L)^p
            idx = torch.arange(1, L+1, device=device, dtype=h.dtype)
            w   = (idx / L)**self.power_exponent
            w   = w / w.sum()
            pooled = (h * w.view(1, L, 1)).sum(dim=1)

        elif self.pool_type == 'attn':
            # compute raw scores for each step
            # h: (B, L, d_model) ? scores: (B, L, 1)
            scores  = self.attn_scorer(h)           # (B, L, 1)
            weights = torch.softmax(scores, dim=1)  # (B, L, 1)
            pooled  = (h * weights).sum(dim=1)      # (B, d_model)

        else:
            raise ValueError(f"Unsupported pool_type: {self.pool_type}")

        # package as LSTM-style hidden/cell states
        h_enc = pooled.unsqueeze(0)                # (1, B, d_model)
        c_enc = torch.zeros_like(h_enc)            # no cell state for SSM

        return h, (h_enc, c_enc)
