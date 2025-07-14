# papercode/lstm.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2SeqLSTM(nn.Module):
    """
    Single-stream sequence-to-sequence LSTM
      • consumes the whole (365 + H) input sequence in one go
      • emits a prediction at every time step
      • we keep only the last  H  rows during training / validation
    """
    def __init__(
        self,
        in_size:   int,       # feature dimension of each time step
        hidden:    int = 256, # LSTM hidden size
        horizon:   int = 8,   # forecast horizon H
        n_layers:  int = 1,
        dropout:   float = 0.0,
    ):
        super().__init__()
        self.horizon = horizon

        self.lstm  = nn.LSTM(
            input_size   = in_size,
            hidden_size  = hidden,
            num_layers   = n_layers,
            batch_first  = True,
            dropout      = dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, 1)

    # ------------------------------------------------------------------ #
    def forward(self, x):
        """
        x : (B, 365+H, Din)
        returns (B, H, 1)
        """
        h, _ = self.lstm(x)        # (B, 365+H, hidden)
        y_all = self.head(h)       # (B, 365+H, 1)
        return y_all[:, -self.horizon :, :]   # keep only the last H steps
   
'''
class EncoderDecoderDetLSTM(nn.Module):
    """
    Deterministic ED-LSTM similar to your diffusion version.
    - x_past:        (B, T_past,  F_past)
    - future_forc:   (B, H,     F_future)   # e.g. precip forecasts inc. t0
    - static_attrs:  (B, S) or None
    -> y_pred:       (B, H,     1)
    """
    def __init__(self, past_features, future_features, static_size,
                 hidden, horizon, dropout):
        super().__init__()
        self.horizon = horizon

        self.enc = nn.LSTM(past_features + static_size, hidden, batch_first=True)
        self.dec = nn.LSTM(future_features + static_size, hidden, batch_first=True)
        self.readout = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_past, future_forc, static=None):
        if static is not None:
            static_exp = static.unsqueeze(1).repeat(1, x_past.size(1), 1)
            x_past     = torch.cat([x_past, static_exp], -1)
        h_enc, (h0, c0) = self.enc(x_past)

        if static is not None:
            static_dec = static.unsqueeze(1).repeat(1, future_forc.size(1), 1)
            future_forc = torch.cat([future_forc, static_dec], -1)

        h_dec, _ = self.dec(future_forc, (h0, c0))
        h_dec    = self.drop(h_dec)
        y_pred   = self.readout(h_dec)         # (B,H,1)
        return y_pred
'''

from typing import Optional, Tuple
import torch
import torch.nn as nn

# ————————————————————————————————————————————————————————————————————————————————
# 1) Your custom cell and wrapper (from your “specific one”):
# ————————————————————————————————————————————————————————————————————————————————
class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        batch_first: bool = True,
        init_forget_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.init_forget_bias = init_forget_bias

        self.weight_ih = nn.Parameter(torch.empty(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, 4 * hidden_size))
        self.bias = nn.Parameter(torch.empty(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight_ih)
        self.weight_hh.data = torch.eye(self.hidden_size).repeat(1, 4)
        nn.init.zeros_(self.bias)
        if self.init_forget_bias:
            self.bias.data[: self.hidden_size].fill_(self.init_forget_bias)

    def forward(
        self,
        x: torch.Tensor,  # [B,T,D] if batch_first else [T,B,D]
        init_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.batch_first:
            x = x.transpose(0, 1)  # → [T,B,D]

        T, B, _ = x.size()
        if init_state is None:
            h_t = x.new_zeros(B, self.hidden_size)
            c_t = x.new_zeros(B, self.hidden_size)
        else:
            h_t, c_t = init_state
            if h_t.dim() == 3:
                h_t = h_t.squeeze(0)
                c_t = c_t.squeeze(0)

        h_seq = []
        bias = self.bias.unsqueeze(0)  # [1,4H]
        for t in range(T):
            gates = (
                torch.addmm(bias.expand(B, -1), h_t, self.weight_hh)
                + x[t] @ self.weight_ih
            )
            f, i, o, g = gates.chunk(4, dim=1)
            c_t = torch.sigmoid(f) * c_t + torch.sigmoid(i) * torch.tanh(g)
            h_t = torch.sigmoid(o) * torch.tanh(c_t)
            h_seq.append(h_t)

        h_seq = torch.stack(h_seq, dim=0)  # [T,B,H]
        if self.batch_first:
            h_seq = h_seq.transpose(0, 1)  # [B,T,H]
        return h_seq, (h_t, c_t)


class GenericLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        init_forget_bias: float = 0.0,
        dropout: float = 0.0,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.lstm = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            init_forget_bias=init_forget_bias,
            batch_first=batch_first,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # [B,T,D]
        *,
        init_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        static_attr: Optional[torch.Tensor] = None,  # [B,S]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # concatenate static if provided
        if static_attr is not None:
            seq_len = x.size(1)
            static_exp = static_attr.unsqueeze(1).expand(-1, seq_len, -1)
            x = torch.cat([x, static_exp], dim=-1)

        h_seq, (h_n, c_n) = self.lstm(x, init_state=init_state)
        h_seq = self.dropout(h_seq)  # apply dropout to all time‐steps
        return h_seq, (h_n, c_n)


# ————————————————————————————————————————————————————————————————————————————————
# 2) Your updated Encoder–Decoder using GenericLSTM
# ————————————————————————————————————————————————————————————————————————————————
class EncoderDecoderDetLSTM(nn.Module):
    """
    Deterministic ED-LSTM, but now with built-in dropout after each time step,
    using the same pattern as your GenericLSTM.
    """
    def __init__(
        self,
        past_features: int,
        future_features: int,
        static_size: int,
        hidden: int,
        horizon: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.horizon = horizon

        # encoder LSTM with dropout
        self.enc = GenericLSTM(
            input_size=past_features + static_size,
            hidden_size=hidden,
            init_forget_bias=3.0,
            dropout=dropout,
            batch_first=True,
        )

        # decoder LSTM with dropout
        self.dec = GenericLSTM(
            input_size=future_features + static_size,
            hidden_size=hidden,
            init_forget_bias=3.0,
            dropout=dropout,
            batch_first=True,
        )

        self.readout = nn.Linear(hidden, 1)

    def forward(
        self,
        x_past: torch.Tensor,           # [B, T_past, F_past]
        future_forc: torch.Tensor,      # [B, T_future, F_future]
        static: Optional[torch.Tensor]  # [B, static_size] or None
    ) -> torch.Tensor:
        # encode
        h_enc_seq, (h0, c0) = self.enc(x_past, static_attr=static)

        # decode, seeding with encoder final state
        h_dec_seq, _ = self.dec(
            future_forc,
            init_state=(h0, c0),
            static_attr=static,
        )

        # final readout
        y_pred = self.readout(h_dec_seq)  # [B, T_future, 1]
        return y_pred
      


