import torch
import torch.nn as nn
from typing import Tuple, Optional
import pdb

'''
class GenericLSTM(nn.Module):
    """
    A drop‑in LSTM that can act as encoder or decoder.

    Parameters
    ----------
    input_size : int
    hidden_size : int
    num_layers : int
    dropout : float
    batch_first : bool
    init_forget_bias : float
        Value assigned to the forget‑gate bias b_f for **every layer
        and (if bidirectional) every direction**. Typical hydrology /
        seq‑model values are 1 → 5.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = True,
        init_forget_bias: float = 3.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=batch_first,
            bidirectional=False,              # edit if you need bi‑LSTM
        )

        # ---------- set forget‑gate bias ----------------------------------
        if init_forget_bias != 0.0:
            self._set_forget_gate_bias(init_forget_bias)
        # ------------------------------------------------------------------

    # ---------------------------------------------------------------
    # private helper so we don’t clutter __init__
    # ---------------------------------------------------------------
    def _set_forget_gate_bias(self, bias_val: float):
        """
        Fill the forget‑gate bias slice with `bias_val`.
        Works for any num_layers and (optionally) bidirectional setup.
        """
        for name, param in self.lstm.named_parameters():
            if "bias_ih" in name or "bias_hh" in name:
                hidden = param.size(0) // 4
                with torch.no_grad():
                    param[hidden : 2 * hidden].fill_(bias_val)

    # ---------------------------------------------------------------
    # forward pass (unchanged)
    # ---------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                      # (B, H, D_in)
        init_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        static_attr: torch.Tensor | None = None,  # (B, S)
    ):
        B, H, D = x.shape

        # 1) concatenate static attributes if provided
        if static_attr is not None:
            stat = static_attr.unsqueeze(1).expand(-1, H, -1)
            x = torch.cat([x, stat], dim=-1)

        # 2) run LSTM
        if init_state is None:
            outputs, (h, c) = self.lstm(x)
        else:
            outputs, (h, c) = self.lstm(x, init_state)

        return outputs, (h, c)
'''


from typing import Optional, Tuple

import torch
import torch.nn as nn

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

        # Parameter layout (input @ W_ih   +   h_{t-1} @ W_hh  +  bias)
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

    # ---------------------------------------------------------------------
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
        init_forget_bias: float = 3.0,
        dropout: float = 0.0,
        batch_first: bool = True
    ) -> None:
        super().__init__()

        self.lstm = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            init_forget_bias=init_forget_bias,
            batch_first=batch_first,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    # ---------------------------------------------------------------------
    def forward(
        self,
        x_dyn: torch.Tensor,                  # [B,T,D]
        *,
        init_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        static_attr: Optional[torch.Tensor] = None,  # [B,S]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Optional static concatenation -----------------------------------
        if static_attr is not None:
            seq_len = x_dyn.size(1)
            static_exp = static_attr.unsqueeze(1).expand(-1, seq_len, -1)
            x_dyn = torch.cat([x_dyn, static_exp], dim=-1)

        h_seq, (h_n, c_n) = self.lstm(x_dyn, init_state=init_state)

        out = self.dropout(h_seq)  # [B,T,H]

        return out, (h_n, c_n)

