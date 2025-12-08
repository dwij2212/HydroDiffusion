
import torch
import torch.nn as nn
import math

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

class EncoderDecoderDiffusionLSTM(nn.Module):
    '''
    Given:
       x_past:       (B, P, past_features) 
       x_t:          (B, H, 1)            
       t:            (B,)                   
       static_attrs: (B, S) or None         
       future_precip:(B, H, 1) or None
    Predicts:
       v_pred: (B, H, 1)  the velocity/denoising output
    '''
    def __init__(self,
                 past_features:int,
                 static_size:int,
                 hidden_size:int=256,
                 time_emb_dim:int=16,
                 forecast_horizon:int=8,
                 dropout: float=0.3):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.static_size = static_size
        self.time_emb_dim = time_emb_dim

        # 1) encoder for past context
        self.encoder = nn.LSTM(past_features, hidden_size,
                               num_layers=1, batch_first=True)

        # 2) time embedding via random Fourier + MLP
        self.mp = MPFourier(time_emb_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim*2),
            nn.SiLU(),
            nn.Linear(time_emb_dim*2, time_emb_dim),
        )

        # 3) decoder: now takes [x_t | precip | t_emb | static]
        dec_in_dim = 1 + 1 + time_emb_dim + static_size
        self.decoder = nn.LSTM(dec_in_dim, hidden_size,
                               num_layers=1, batch_first=True)

        self.dropout     = nn.Dropout(p=dropout)
        self.predict_out = nn.Linear(hidden_size, 1)

    def forward(self,
                x_past:       torch.Tensor,
                x_t:          torch.Tensor,
                t:            torch.Tensor,
                static_attrs: torch.Tensor=None,
                future_precip:torch.Tensor=None
               ) -> torch.Tensor:
               
        #x_t = x_t.unsqueeze(-1)
        B, H, _ = x_t.shape

        # Encode past context once
        _, (h_enc, c_enc) = self.encoder(x_past)

        # Time embedding
        t_feat = self.time_embed(self.mp(t))      # (B, time_emb_dim)
        t_seq  = t_feat.unsqueeze(1).repeat(1, H, 1)

        # Static features
        if static_attrs is None:
            s_seq = torch.zeros(B, H, self.static_size,
                                device=x_t.device, dtype=x_t.dtype)
        else:
            s_seq = static_attrs.unsqueeze(1).repeat(1, H, 1)

        # Precip forcing (optional)
        if future_precip is None:
            p_seq = torch.zeros(B, H, 1,
                                device=x_t.device, dtype=x_t.dtype)
        else:
            p_seq = future_precip

        # Build decoder input: [noisy-target | precip | t-emb | static]
        dec_in = torch.cat([x_t, p_seq, t_seq, s_seq], dim=-1)  # (B,H,1+1+T+S)
        dec_out, _ = self.decoder(dec_in, (h_enc, c_enc))       # (B,H,hidden)
        dec_out    = self.dropout(dec_out)
        v_pred     = self.predict_out(dec_out)                  # (B,H,1)
        return v_pred


class EncoderDiffusionLSTM(nn.Module):
    '''
    Given:
       x_past:       (B, P, past_features) 
    Produce:
       hidden: (B, H, hidden_size)
    '''
    def __init__(self,
                 past_features:int,
                 hidden_size:int=256,
                 forecast_horizon:int=8,
                 dropout: float=0.3):
        super().__init__()
        self.forecast_horizon = forecast_horizon

        # 1) encoder for past context
        self.encoder = nn.LSTM(past_features, hidden_size,
                               num_layers=1, batch_first=True)

    def forward(self,
                x_past:       torch.Tensor
               ) -> torch.Tensor:
               
        # Encode past context once
        _, (h_enc, c_enc) = self.encoder(x_past)
        
        return h_enc, c_enc
        
        
class DecoderDiffusionLSTM(nn.Module):
    '''
    Decoder for diffusion model (autoregressive or parallel, as you want).

    Given:
      - decoder_inputs: (B, H, in_features)  # e.g., noise, static attrs, time emb, etc.
      - h0, c0: initial hidden/cell from encoder (num_layers, B, hidden_size)

    Produces:
      - velocity: (B, H, 1)
    '''
    def __init__(self, 
                 in_features: int, 
                 hidden_size: int = 256,
                 out_features: int = 1,
                 num_layers: int = 1,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.output_proj = nn.Linear(hidden_size, out_features)

    def forward(self, decoder_inputs, h0, c0):
        # decoder_inputs: (B, H, in_features)
        outputs, (hn, cn) = self.lstm(decoder_inputs, (h0, c0))   # h0, c0 shapes: (num_layers, B, hidden_size)
        outputs    = self.dropout(outputs)
        v_pred = self.output_proj(outputs)  # (B, H, out_features)
        return v_pred

