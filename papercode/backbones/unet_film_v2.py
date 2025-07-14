import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class FiLMBlock(nn.Module):
    def __init__(self, channels, emb_dim, dropout=0.1):
        super().__init__()
        self.norm    = nn.GroupNorm(8, channels)
        self.fc      = nn.Linear(emb_dim, 2 * channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, emb):
        # x: (B, C, T), emb: (B, emb_dim)
        h = self.norm(x)
        scale, shift = self.fc(emb).chunk(2, dim=1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        h = h * (1 + scale) + shift
        return self.dropout(h)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim,
                 kernel_size=3, dilation=1, dropout=0.0):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=pad, dilation=dilation)
        self.gn1   = nn.GroupNorm(8, out_ch)
        self.act1  = nn.SiLU()
        self.film  = FiLMBlock(out_ch, emb_dim, dropout=dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=pad, dilation=dilation)
        self.gn2   = nn.GroupNorm(8, out_ch)
        self.act2  = nn.SiLU()
        self.drop2 = nn.Dropout(dropout)

        if in_ch != out_ch:
            self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, emb):
        h = self.act1(self.gn1(self.conv1(x)))
        h = self.film(h, emb)
        h = self.drop2(self.act2(self.gn2(self.conv2(h))))
        return self.skip(x) + h


class unet_film_v2(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64,
                 static_dim=27, h_lstm_dim=256, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        self.time_proj = nn.Linear(h_lstm_dim, hidden_dim)
        self.lstm_proj = nn.Linear(h_lstm_dim, hidden_dim)
        
        self.cond_proj = nn.Linear(h_lstm_dim, hidden_dim*4)

        self.down1 = ResBlock(hidden_dim,     hidden_dim,   emb_dim=hidden_dim, dropout=dropout)
        self.down2 = ResBlock(hidden_dim,     hidden_dim*2, emb_dim=hidden_dim, dropout=dropout)
        self.down3 = ResBlock(hidden_dim*2,   hidden_dim*4, emb_dim=hidden_dim, dropout=dropout)
        self.pool  = nn.AvgPool1d(2, 2)

        self.bottleneck = ResBlock(hidden_dim*4, hidden_dim*4, emb_dim=hidden_dim,
                                   dilation=2, dropout=dropout)
        self.bot_drop   = nn.Dropout(dropout * 1.5)

        self.static_proj = nn.Linear(static_dim, hidden_dim*4)
        self.precip_proj = nn.Conv1d(1, hidden_dim*4, kernel_size=3, padding=1)

        self.up3 = ResBlock(hidden_dim*8, hidden_dim*2, emb_dim=hidden_dim, dropout=dropout)
        self.up2 = ResBlock(hidden_dim*4, hidden_dim,   emb_dim=hidden_dim, dropout=dropout)
        self.up1 = ResBlock(hidden_dim*2, hidden_dim,   emb_dim=hidden_dim, dropout=dropout)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)

        self.final_proj = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def _make_emb(self, temb, h_lstm):
        return self.time_proj(temb) + self.lstm_proj(h_lstm)

    def forward(self, x, temb, h_lstm=None, future_pcp=None, static_attr=None):
        x = x.permute(0, 2, 1)               # (B, 1, H)
        
        ## concatenate test ################
        if future_pcp is not None:
            x = torch.cat([x, future_pcp.permute(0, 2, 1)], dim=1)
        ####################################
        
        h = self.input_proj(x)              # (B, hidden_dim, H)

        h_l = h_lstm.squeeze(0) if h_lstm is not None else torch.zeros_like(temb)

        # Down path
        d1 = self.down1(h,    self._make_emb(temb, h_l))
        d2 = self.down2(self.pool(d1), self._make_emb(temb, h_l))
        d3 = self.down3(self.pool(d2), self._make_emb(temb, h_l))

        # Bottleneck
        bi   = self.pool(d3)
        bott = self.bot_drop(self.bottleneck(bi, self._make_emb(temb, h_l)))

        # ---- only this block changed ----
        # project temb into channel space, then do out-of-place adds:
        cond = self.cond_proj(temb).unsqueeze(-1)              # (B, hidden_dim, 1)
        if static_attr is not None:
            s = static_attr.mean(dim=1)                       # (B, static_dim)
            cond = cond + self.static_proj(s).unsqueeze(-1)
        ''' comment for concat test
        if future_pcp is not None:
            pcp_emb = self.precip_proj(future_pcp.permute(0, 2, 1))
            pcp_emb = self.pool(self.pool(self.pool(pcp_emb)))
            cond = cond + pcp_emb
        '''    
            
        bott = bott + cond
        # ----------------------------------

        # Up path
        u3 = self.up3(torch.cat([self.upsample(bott), d3], dim=1), self._make_emb(temb, h_l))
        u2 = self.up2(torch.cat([self.upsample(u3), d2], dim=1), self._make_emb(temb, h_l))
        u1 = self.up1(torch.cat([self.upsample(u2), d1], dim=1), self._make_emb(temb, h_l))

        out = self.final_proj(u1)         # (B, 1, H)
        return out.permute(0, 2, 1)       # (B, H, 1)
