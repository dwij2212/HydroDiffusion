import torch
import torch.nn as nn
from models.s4.s4d import S4D as LTI
import pdb
from diffusion_utils import diffusion_params
import math

dropout_fn = nn.Dropout1d if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12) else nn.Dropout

class MPFourier(nn.Module):
    def __init__(self, num_channels, bandwidth=1.0):
        super().__init__()
        self.register_buffer('freqs', 2*math.pi*torch.randn(num_channels))
        self.register_buffer('phases', 2*math.pi*torch.rand(num_channels))

    def forward(self, t):
        t = t.to(torch.float32)
        x = t[:, None]*self.freqs[None,:] + self.phases[None,:]
        return (x.cos() * math.sqrt(2)).to(t.dtype)

class decoder_only_ssm(nn.Module):
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
        emb_dim: int = 32,
        variant: str = "static",
        dropout: float = 0.15,
        time_full: bool = False,
    ):
        super().__init__()
        self.d_model    = d_model
        self.H          = horizon
        self.static_dim = static_dim
        self.emb_dim    = emb_dim
        self.variant    = variant

        if variant == "static":
            ctx_dim = static_dim
        elif variant == "z":
            ctx_dim = emb_dim
        elif variant == "z+static":
            ctx_dim = static_dim + emb_dim
        else:
            raise ValueError(f"Unknown variant '{variant}'. Choose from: static, z, z+static")

        self.input_proj = nn.Linear(d_input + 1 + ctx_dim, d_model)

        self.mp = MPFourier(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, d_model),
        )
        self.time_full = True

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

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        x_past:       torch.Tensor,
        noisy_future: torch.Tensor,
        t:            torch.Tensor,
        x_future:     torch.Tensor,
        static_attr:  torch.Tensor,
        z_emb:        torch.Tensor = None,
    ) -> torch.Tensor:
        B, L, _ = x_past.shape
        H        = self.H
        device   = x_past.device

        t_feats = self.mp(t)

        all_met  = torch.cat([x_past, x_future], dim=1)
        pad_flow = torch.zeros(B, L-1, 1, device=device)
        all_flow = torch.cat([pad_flow, noisy_future], dim=1)

        if self.variant == "static":
            ctx = static_attr[:, 0, :]
        elif self.variant == "z":
            assert z_emb is not None, "z_emb required for variant='z'"
            ctx = z_emb
        else:
            assert z_emb is not None, "z_emb required for variant='z+static'"
            ctx = torch.cat([static_attr[:, 0, :], z_emb], dim=-1)

        ctx_seq = ctx.unsqueeze(1).expand(-1, L+H-1, -1)
        feats = torch.cat([all_met, all_flow, ctx_seq], dim=-1)

        h = self.input_proj(feats)

        t_b = self.time_mlp(t_feats)
        time_bias = torch.zeros_like(h)
        if self.time_full:
            time_bias[:, L-1:, :] = t_b.unsqueeze(1).expand(-1, self.H, -1)
        else:
            time_bias[:, L-1, :] = t_b
        h = h + time_bias

        h = h.transpose(1, 2)
        for blk, norm, drop in zip(self.blocks, self.norms, self.drops):
            z, _ = blk(h)
            h    = norm(h + drop(z))
        h = h.transpose(1, 2)

        out = self.head(h)
        return out[:, -H:, :]

    @torch.no_grad()
    def sample_ddim(self,
                    x_past:            torch.Tensor,
                    static_attributes: torch.Tensor,
                    future_pcp:        torch.Tensor,
                    num_steps:         int = 10,
                    eta:               float = 0.0,
                    z_emb:             torch.Tensor = None,
                   ) -> torch.Tensor:
        device = x_past.device
        B, L, _ = x_past.shape
        H = future_pcp.size(1) + 1

        ts = torch.linspace(0., 1., num_steps, device=device)
        x  = torch.randn(B, H, 1, device=device)

        for i in range(num_steps - 1, -1, -1):
            t      = ts[i].repeat(B)
            t_prev = ts[i - 1].repeat(B) if i > 0 else ts[0].repeat(B)

            pred = self.forward(
                x_past       = x_past,
                noisy_future = x,
                t            = t,
                x_future     = future_pcp,
                static_attr  = static_attributes,
                z_emb        = z_emb,
            )

            _, alpha_t,  sigma_t  = diffusion_params(t)
            _, alpha_tp, sigma_tp = diffusion_params(t_prev)

            alpha_t  = alpha_t.view(B, 1, 1)
            sigma_t  = sigma_t.view(B, 1, 1)
            alpha_tp = alpha_tp.view(B, 1, 1)
            sigma_tp = sigma_tp.view(B, 1, 1)

            x0  = alpha_t * x - sigma_t * pred
            eps = (pred + sigma_t * x0) / alpha_t

            if i > 0:
                sigma = eta * torch.sqrt(
                    torch.clamp((sigma_tp**2) * (1 - (alpha_t**2 / alpha_tp**2)),
                                min=1e-12)
                )
                noise = torch.randn_like(x) if eta > 0 else 0.0
                x = alpha_tp * x0 + sigma_tp * eps + sigma * noise
            else:
                x = x0

        return x.squeeze(-1)
