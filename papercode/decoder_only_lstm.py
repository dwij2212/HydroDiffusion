import math
import torch
import torch.nn as nn
from papercode.diffusion_utils import diffusion_params
from papercode.lstm import Seq2SeqLSTM, EncoderDecoderDetLSTM
from papercode.SSM_test import HOPE, setup_optimizer
from papercode.backbones.lstm import GenericLSTM
import pdb

# PyTorch 1.12+ naming
dropout_fn = nn.Dropout1d if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12) else nn.Dropout

class MPFourier(nn.Module):
    def __init__(self, num_channels, bandwidth=1.0):
        super().__init__()
        self.register_buffer('freqs', 2*math.pi*torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2*math.pi*torch.rand(num_channels))

    def forward(self, t):
        # Accept (B,), (B,1), (B,H) or (B,H,1) and reduce to (B,)
        if t.dim() >= 2:
            t = t[..., 0]
            if t.dim() >= 2:
                t = t[:, 0]
        t = t.to(torch.float32)  # (B,)
        x = t[:, None]*self.freqs[None, :] + self.phases[None, :]
        return (x.cos() * math.sqrt(2.)).to(t.dtype)  # (B, C)
'''
class decoder_only_lstm(nn.Module):
    """
    Encoder-only LSTM that injects BOTH time and static as a bias ONLY at the nowcast step (L-1).
    """
    def __init__(
        self,
        d_input: int, # dynamic forcing + static
        hidden_size: int,
        cfg: dict,
        *,
        horizon: int = 8,
        static_dim: int = 27,
        time_emb_dim: int = 256,
        time_full: bool = True,
    ):
        super().__init__()
        self.hidden_size=hidden_size
        self.H         = horizon
        self.static_dim= static_dim
        self.time_full = time_full

        # feed met + flow + static through the sequence projector.
        self.input_proj = nn.Linear(d_input + 1, hidden_size)

        # Time embedding -> bias
        self.mp = MPFourier(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, hidden_size),
        )
        
        
        # LSTM encoder
        self.lstm=GenericLSTM(
            input_size = cfg['hidden_size'],
            hidden_size = cfg['hidden_size'],
            dropout = cfg['dropout'],
            init_forget_bias=cfg['initial_forget_gate_bias'],
            batch_first =True
        )

        # read out
        self.head = nn.Linear(hidden_size, 1)

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

        # Predict only the future window
        h_seq, _ = self.lstm(h)
        out = self.head(h_seq)                                        # (B, H, 1)

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
            t = ts[i].repeat(B) 
            t_prev = ts[i - 1].repeat(B) if i > 0 else ts[0].repeat(B)

            # call your LSTM:
            #   x_past:     (B, L, d_input)
            #   noisy_future: x        (B, H, 1)
            #   t:          t     (B, H, 1)
            #   x_future:   future forcings (B, H-1, d_input), where H-1 is to exclude the nowcasting day
            #   static_attr: static_msg  (B, S)
            # note: your SSM signature is forward(x_past, noisy_future, t, x_future, static_attr)
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

            # the prediction type is just velocity. 
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

        # squeeze off the last dim  (B, H)
        return x.squeeze(-1)
'''

class decoder_only_lstm(nn.Module):
    """
    Encoder-only LSTM with GenericLSTM:
      - LSTM input_size = d_input + 1 + static_dim (e.g., 33)
      - Run past (0..L-2) to get (h_p, c_p)
      - Add time-embedding to (h_p, c_p) at the nowcast boundary (bottleneck)
      - Run future segment (L-1..L+H-2) from (h0, c0)
    """
    def __init__(
        self,
        d_input: int,                 # dynamic forcings ONLY (no statics here)
        hidden_size: int,
        cfg: dict,
        *,
        horizon: int = 8,
        static_dim: int = 27,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        self.H           = horizon
        self.hidden_size = hidden_size
        self.static_dim  = static_dim
        self.in_dim      = d_input + 1 # keep raw input size (e.g., 33)

        # GenericLSTM over raw features (used for both past and future segments)
        self.lstm = GenericLSTM(
            input_size        = self.in_dim,
            hidden_size       = hidden_size,
            dropout           = cfg.get('dropout', 0.0),
            init_forget_bias  = cfg.get('initial_forget_gate_bias', 3.0),
            batch_first       = True
        )

        # Time embedding -> hidden-size bias (added to BOTH h and c at bottleneck)
        self.mp = MPFourier(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, hidden_size),
        )

        # Readout
        self.head = nn.Linear(hidden_size, 1)

    def _build_sequence(self, x_past, noisy_future, x_future, static_attr):
        """
        x_past:      (B, L, d_input)
        noisy_future:(B, H, 1)
        x_future:    (B, H-1, d_input)   # excludes nowcast day
        static_attr: (B, static_dim) or (B, 1, static_dim)

        Returns feats: (B, L+H-1, d_input+1+static_dim), L
        """
        B, L, _ = x_past.shape
        H = self.H
        device = x_past.device

        all_met  = torch.cat([x_past, x_future], dim=1)           # (B, L+H-1, d_input)
        pad_flow = torch.zeros(B, L-1, 1, device=device)
        all_flow = torch.cat([pad_flow, noisy_future], dim=1)     # (B, L+H-1, 1)

        if static_attr.dim() == 2:
            static_seq = static_attr.unsqueeze(1).expand(-1, L+H-1, -1)  # (B, L+H-1, S)
        else:
            static_seq = static_attr[:, 0, :].unsqueeze(1).expand(-1, L+H-1, -1)

        feats = torch.cat([all_met, all_flow, static_seq], dim=-1).contiguous()
        return feats, L

    def forward(
        self,
        x_past:       torch.Tensor,  # (B, L, d_input)
        noisy_future: torch.Tensor,  # (B, H, 1)
        t:            torch.Tensor,  # (B,) or (B,H,1) etc.
        x_future:     torch.Tensor,  # (B, H-1, d_input)
        static_attr:  torch.Tensor,  # (B, static_dim) or (B,1,static_dim)
    ) -> torch.Tensor:
        B, L, _ = x_past.shape
        H = self.H

        feats, L = self._build_sequence(x_past, noisy_future, x_future, static_attr)  # (B, T, in_dim)
        T = feats.size(1)
        assert T == L + H - 1, "Sequence length mismatch."

        # 1) Encode past (0..L-2) with GenericLSTM (stateless call)
        if L - 1 > 0:
            past = feats[:, :L-1, :]                               # (B, L-1, in_dim)
            _, (h_p, c_p) = self.lstm(past)                        # h_p, c_p: (B, hidden)
        else:
            h_p = feats.new_zeros(B, self.hidden_size)
            c_p = feats.new_zeros(B, self.hidden_size)

        # 2) Bottleneck time conditioning: add t-embedding to (h, c) once
        t_emb = self.time_mlp(self.mp(t))                          # (B, hidden)
        h0 = h_p + t_emb
        c0 = c_p + t_emb

        # 3) Decode future segment starting from (h0, c0)
        fut_in = feats[:, L-1:, :]                                 # (B, H, in_dim)
        fut_out, _ = self.lstm(fut_in, init_state=(h0, c0))        # (B, H, hidden)

        # 4) Project to target space
        return self.head(fut_out)                                  # (B, H, 1)

    @torch.no_grad()
    def sample_ddim(
        self,
        x_past: torch.Tensor,                 # (B, L, d_input)
        static_attributes: torch.Tensor,      # (B, static_dim) or (B,1,static_dim)
        future_pcp: torch.Tensor,             # (B, H-1, d_input)
        num_steps: int = 10,
        eta: float = 0.0,
    ) -> torch.Tensor:
        device = x_past.device
        B, L, _ = x_past.shape
        H = future_pcp.size(1) + 1

        ts = torch.linspace(0., 1., num_steps, device=device)
        x = torch.randn(B, H, 1, device=device)

        for i in range(num_steps - 1, -1, -1):
            t_cur   = ts[i].repeat(B)
            t_prev  = ts[i - 1].repeat(B) if i > 0 else ts[0].repeat(B)

            pred = self.forward(
                x_past       = x_past,
                noisy_future = x,
                t            = t_cur,
                x_future     = future_pcp,
                static_attr  = static_attributes
            )  # (B, H, 1)

            _, alpha_t,  sigma_t  = diffusion_params(t_cur)
            _, alpha_tp, sigma_tp = diffusion_params(t_prev)

            alpha_t  = alpha_t.view(B, 1, 1)
            sigma_t  = sigma_t.view(B, 1, 1)
            alpha_tp = alpha_tp.view(B, 1, 1)
            sigma_tp = sigma_tp.view(B, 1, 1)

            # velocity parameterization
            x0  = alpha_t * x - sigma_t * pred
            eps = (pred + sigma_t * x0) / alpha_t

            if i > 0:
                sigma = eta * torch.sqrt(
                    torch.clamp((sigma_tp**2) * (1 - (alpha_t**2 / alpha_tp**2)), min=1e-12)
                )
                noise = torch.randn_like(x) if eta > 0 else 0.0
                x = alpha_tp * x0 + sigma_tp * eps + sigma * noise
            else:
                x = x0

        return x.squeeze(-1)

