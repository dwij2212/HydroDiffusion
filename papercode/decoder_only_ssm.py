import torch
import torch.nn as nn
from models.s4.s4d import S4D as LTI
import pdb
from diffusion_utils import diffusion_params
import math

# Determine correct Dropout function based on PyTorch version
dropout_fn = nn.Dropout1d if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12) else nn.Dropout

class MPFourier(nn.Module):
    def __init__(self, num_channels, bandwidth=1.0):
        super().__init__()
        self.register_buffer('freqs', 2*math.pi*torch.randn(num_channels))
        self.register_buffer('phases', 2*math.pi*torch.rand(num_channels))

    def forward(self, t):
        # t: (B,) long/int or float tensor of diffusion step indices
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
        dropout: float = 0.15,
        time_full: bool = False,
    ):
        super().__init__()
        self.d_model   = d_model
        self.H         = horizon
        self.static_dim = static_dim
        
        # 1) project raw input (met + noisy flow + static) to state dim
        self.input_proj = nn.Linear(d_input + 1, d_model)

        # 2) time embedding projector -> bias
        self.mp = MPFourier(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, d_model),
        )
        self.time_full = True

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
        noisy_future: torch.Tensor,  # (B, H, 1)
        t:     torch.Tensor,  # (B, H, 1)
        x_future:     torch.Tensor,  # (B, H-1, d_input), -1 is for excluding the nowcast day
        static_attr:  torch.Tensor,  # (B, static_dim)
    ) -> torch.Tensor:
        B, L, _ = x_past.shape
        H        = self.H
        device   = x_past.device
        
        t_feats = self.mp(t) # (B, time_emb_dim)

        # Build feature sequence: met & flow & static
        all_met    = torch.cat([x_past, x_future], dim=1)               # (B, L+H-1, d_input)
        pad_flow   = torch.zeros(B, L-1, 1, device=device)
        all_flow   = torch.cat([pad_flow, noisy_future], dim=1)         # (B, L+H-1, 1)
        static_seq = static_attr[:,0,:].unsqueeze(1).expand(-1, L+H-1, -1)   
        feats = torch.cat([all_met, all_flow, static_seq], dim=-1) # (B, L+H-1, d_input+1)

        # Project to state dimension
        h = self.input_proj(feats)  # (B, L+H, d_model)
        
        # Time bias: apply embedding at nowcast index L-1 or full horizon
        t_b = self.time_mlp(t_feats)  # (B, d_model)
        time_bias = torch.zeros_like(h)  # (B, L+H, d_model)
        if self.time_full:
            # broadcast the time embedding across all forecast steps (positions L-1 to L+H-1)
            time_bias[:, L-1:, :] = t_b.unsqueeze(1).expand(-1, self.H, -1)
        else:
            time_bias[:, L-1, :] = t_b
        h = h + time_bias
        
        # Residual S4D stack
        h = h.transpose(1, 2)  # (B, d_model, L+H)
        for blk, norm, drop in zip(self.blocks, self.norms, self.drops):
            z, _ = blk(h)
            h    = norm(h + drop(z))
        h = h.transpose(1, 2)  # (B, L+H, d_model)

        # Predict only the future window
        out = self.head(h)          # (B, L+H, 1)
        return out[:, -H:, :]       # (B, H, 1)
        
    @torch.no_grad()
    def sample_ddim(self,
                    x_past: torch.Tensor,           # (B, L, d_input)
                    static_attributes: torch.Tensor,      # (B, S)
                    future_pcp: torch.Tensor,       # (B, H-1, forcing_dim[5, 15])
                    num_steps: int = 10,
                    eta: float = 0.0,
                   ) -> torch.Tensor:
        """
        DDIM sampling for the decoder_only_ssm model.
        Returns x_0 of shape (B, H).
        """
        device = x_past.device
        B, L, _ = x_past.shape
        H = future_pcp.size(1) + 1

        # create the uniform time grid [0..1]
        ts = torch.linspace(0., 1., num_steps, device=device)

        # initial noisy sample (we sample the entire H horizon at once)
        x = torch.randn(B, H, 1, device=device)

        for i in range(num_steps - 1, -1, -1):
            # current and next time
            t      = ts[i].repeat(B) 
            t_prev = ts[i - 1].repeat(B) if i > 0 else ts[0].repeat(B)

            # call your SSM:
            #   x_past:     (B, L, d_input)
            #   noisy_future: x        (B, H, 1)
            #   t:          t     (B, H, 1)
            #   x_future:   future forcings (B, H-1, d_input), where H-1 is to exclude the nowcasting day
            #   static_attr: static_msg  (B, S)
            # note: SSM signature is forward(x_past, noisy_future, t, x_future, static_attr)
            pred = self.forward(
                x_past = x_past,
                noisy_future = x, 
                t = t,        # (B, H, 1), broadcasting is performed internally in forward
                x_future = future_pcp,  # (B, H-1, d_input)
                static_attr = static_attributes  # (B, S)
            )  # (B, H, 1)

            # get diffusion parameters at t and t_prev
            _, alpha_t,  sigma_t  = diffusion_params(t)      
            _, alpha_tp, sigma_tp = diffusion_params(t_prev)

            # reshape
            alpha_t  = alpha_t.view(B, 1, 1)
            sigma_t  = sigma_t.view(B, 1, 1)
            alpha_tp = alpha_tp.view(B, 1, 1)
            sigma_tp = sigma_tp.view(B, 1, 1)

            # the prediction type is velocity. 
            x0  = alpha_t * x - sigma_t * pred
            eps = (pred + sigma_t * x0) / alpha_t

            # DDIM update
            if i > 0:
                # deterministic or stochastic step
                sigma = eta * torch.sqrt(
                    torch.clamp((sigma_tp**2) * (1 - (alpha_t**2 / alpha_tp**2)),
                                min=1e-12)
                )
                noise = torch.randn_like(x) if eta > 0 else 0.0
                x = alpha_tp * x0 + sigma_tp * eps + sigma * noise
            else:
                # final step
                x = x0

        # squeeze off the last dim → (B, H)
        return x.squeeze(-1)
