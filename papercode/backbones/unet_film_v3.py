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
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        return self.dropout(h)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim,
                 kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=pad, dilation=dilation)
        self.gn1   = nn.GroupNorm(8, out_ch)
        self.act1  = nn.SiLU()
        self.film  = FiLMBlock(out_ch, emb_dim, dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=pad, dilation=dilation)
        self.gn2   = nn.GroupNorm(8, out_ch)
        self.act2  = nn.SiLU()
        self.drop2 = nn.Dropout(dropout)

        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.act1(self.gn1(self.conv1(x)))
        h = self.film(h, emb)
        h = self.drop2(self.act2(self.gn2(self.conv2(h))))
        return self.skip(x) + h

class unet_film_v3(nn.Module):
    def __init__(
        self,
        input_dim:  int = 1,
        hidden_dim: int = 64,
        static_dim: int = 27,
        h_lstm_dim: int = 256,
        dropout:    float = 0.1,
    ):
        super().__init__()
        # initial 1×1 conv
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        # project time embedding → channels for FiLM
        self.time_proj = nn.Linear(h_lstm_dim, hidden_dim)

        # down / bottleneck / up
        self.down1 = ResBlock(hidden_dim,     hidden_dim,   emb_dim=hidden_dim, dropout=dropout)
        self.down2 = ResBlock(hidden_dim,     hidden_dim*2, emb_dim=hidden_dim, dropout=dropout)
        self.down3 = ResBlock(hidden_dim*2,   hidden_dim*4, emb_dim=hidden_dim, dropout=dropout)
        self.pool  = nn.AvgPool1d(2,2)

        self.bottleneck = ResBlock(hidden_dim*4, hidden_dim*4,
                                   emb_dim=hidden_dim,
                                   dilation=2, dropout=dropout)
        self.bot_drop   = nn.Dropout(dropout * 1.5)

        self.up3 = ResBlock(hidden_dim*8, hidden_dim*2, emb_dim=hidden_dim, dropout=dropout)
        self.up2 = ResBlock(hidden_dim*4, hidden_dim,   emb_dim=hidden_dim, dropout=dropout)
        self.up1 = ResBlock(hidden_dim*2, hidden_dim,   emb_dim=hidden_dim, dropout=dropout)

        self.upsample   = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.final_proj = nn.Conv1d(hidden_dim, 1, kernel_size=1)

        # ---- at bottleneck, we’ll additionally inject:
        #  • time_cond (same emb_dim*4)
        #  • static attributes
        #  • encoder hidden state
        #  • precip
        self.cond_time  = nn.Linear(h_lstm_dim, hidden_dim*4)
        self.static_proj = nn.Linear(static_dim, hidden_dim*4)
        self.hidden_proj = nn.Linear(h_lstm_dim, hidden_dim*4)
        self.precip_proj = nn.Conv1d(1, hidden_dim*4, kernel_size=3, padding=1)

    def forward(
        self,
        x,                   # (B, H, 1)
        temb,                # (B, h_lstm_dim)
        h_lstm=None,         # (1, B, h_lstm_dim)
        future_pcp=None,     # (B, H_fut, 1)
        static_attr=None     # (B, H, static_dim)
    ):
        B, H, _ = x.shape
        # 1) initial conv
        x = x.permute(0,2,1)           # → (B,1,H)
        
        ''' concatenate test '''
        if future_pcp is not None:
            x = torch.cat([x, future_pcp.permute(0, 2, 1)], dim=1)
        
        h = self.input_proj(x)         # → (B,hidden_dim,H)

        # 2) build *only* time-based FiLM embedding
        time_emb = self.time_proj(temb)   # (B, hidden_dim)

        # 3) down path
        d1 = self.down1(h,    time_emb)
        d2 = self.down2(self.pool(d1), time_emb)
        d3 = self.down3(self.pool(d2), time_emb)

        # 4) bottleneck FiLM
        bi   = self.pool(d3)
        bot  = self.bot_drop(self.bottleneck(bi, time_emb))

        # 5) explicit conditioning at bottleneck
        #    (time, static, hidden, precip)
        cond = self.cond_time(temb).unsqueeze(-1)  # (B, C4, 1)
        if static_attr is not None:
            s = static_attr.mean(dim=1)            # (B, static_dim)
            cond = cond + self.static_proj(s).unsqueeze(-1)
        if h_lstm is not None:
            h_enc = h_lstm.squeeze(0)              # (B, h_lstm_dim)
            cond = cond + self.hidden_proj(h_enc).unsqueeze(-1)
        ''' comment for concat test
        if future_pcp is not None:
            p = self.precip_proj(future_pcp.permute(0,2,1))
            p = self.pool(self.pool(self.pool(p)))
            cond = cond + p
        '''
        bot = bot + cond

        # 6) up path
        u3 = self.up3(torch.cat([self.upsample(bot), d3], dim=1), time_emb)
        u2 = self.up2(torch.cat([self.upsample(u3), d2], dim=1), time_emb)
        u1 = self.up1(torch.cat([self.upsample(u2), d1], dim=1), time_emb)

        # 7) final conv → back to (B, H, 1)
        out = self.final_proj(u1)       # (B,1,H)
        return out.permute(0,2,1)       # (B,H,1)
