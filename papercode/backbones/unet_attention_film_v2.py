import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class FiLMBlock(nn.Module):
    def __init__(self, channels, emb_dim, dropout=0.1):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.norm    = nn.GroupNorm(groups, channels)
        self.fc      = nn.Linear(emb_dim, 2 * channels)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, emb):
        h = self.norm(x)
        scale, shift = self.fc(emb).chunk(2, dim=1)
        return self.dropout(h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1))

class CrossAttn1d(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1, max_len=1024):
        super().__init__()
        self.pos_enc = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        self.attn    = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln      = nn.LayerNorm(dim)
        self.drop    = nn.Dropout(dropout)
        self.alpha   = nn.Parameter(torch.tensor(0.1))
    def forward(self, x, kv):
        # x, kv: (B,C,L)
        B, C, L = x.shape
        assert L <= self.pos_enc.size(1), f"Sequence length {L} > {self.pos_enc.size(1)}"
        q = x.permute(0,2,1) + self.pos_enc[:, :L, :]
        k = kv.permute(0,2,1) + self.pos_enc[:, :L, :]
        v = k
        a, _ = self.attn(q, k, v)
        a = self.drop(self.ln(a))
        return x + self.alpha * a.permute(0,2,1)

class ResBlockAttn(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim,
                 attn_heads=4, dropout=0.1, dilation=1,
                 kv_in_ch=None):
        super().__init__()
        pad = (3 - 1)//2 * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=pad, dilation=dilation)
        self.gn1   = nn.GroupNorm(8 if out_ch%8==0 else 1, out_ch)
        self.act1  = nn.SiLU()
        self.film  = FiLMBlock(out_ch, emb_dim, dropout)

        # if we’re going to attend, we need to project incoming kv to out_ch
        self.kv_proj = None
        if kv_in_ch is not None:
            self.kv_proj = nn.Conv1d(kv_in_ch, out_ch, 1)

        self.attn  = CrossAttn1d(out_ch, heads=attn_heads, dropout=dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=pad, dilation=dilation)
        self.gn2   = nn.GroupNorm(8 if out_ch%8==0 else 1, out_ch)
        self.act2  = nn.SiLU()
        self.drop2 = nn.Dropout(dropout)

        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb, pcp_kv=None):
        h = self.act1(self.gn1(self.conv1(x)))
        h = self.film(h, emb)
        if pcp_kv is not None:
            kv = self.kv_proj(pcp_kv) if self.kv_proj is not None else pcp_kv
            h = self.attn(h, kv)
        h = self.drop2(self.act2(self.gn2(self.conv2(h))))
        return self.skip(x) + h

class unet_attention_film_v2(nn.Module):
    def __init__(self,
                 input_dim=1,
                 hidden_dim=64,
                 static_dim=27,
                 h_lstm_dim=256,
                 dropout=0.1,
                 attn_heads=4,
                 max_len=1024):
        super().__init__()
        self.input_proj  = nn.Conv1d(input_dim, hidden_dim, 1)
        self.time_proj   = nn.Linear(h_lstm_dim, hidden_dim)
        self.lstm_proj   = nn.Linear(h_lstm_dim, hidden_dim)
        self.static_proj = nn.Linear(static_dim, hidden_dim * 4)
        self.precip_proj = nn.Conv1d(1, hidden_dim, 3, padding=1)

        # Encoder: tell each block that kv_in_ch=hidden_dim
        self.down1 = ResBlockAttn(hidden_dim, hidden_dim, emb_dim=hidden_dim,
                                  attn_heads=attn_heads, dropout=dropout,
                                  kv_in_ch=hidden_dim)
        self.down2 = ResBlockAttn(hidden_dim, hidden_dim*2, emb_dim=hidden_dim,
                                  attn_heads=attn_heads, dropout=dropout,
                                  kv_in_ch=hidden_dim)
        self.down3 = ResBlockAttn(hidden_dim*2, hidden_dim*4, emb_dim=hidden_dim,
                                  attn_heads=attn_heads, dropout=dropout,
                                  kv_in_ch=hidden_dim)

        self.pool  = nn.AvgPool1d(2,2)

        # Bottleneck also attends:
        self.bottleneck = ResBlockAttn(hidden_dim*4, hidden_dim*4,
                                       emb_dim=hidden_dim,
                                       attn_heads=attn_heads,
                                       dropout=dropout,
                                       dilation=2,
                                       kv_in_ch=hidden_dim)

        # Decoder:
        self.up3 = ResBlockAttn(hidden_dim*8, hidden_dim*2, emb_dim=hidden_dim,
                                attn_heads=attn_heads, dropout=dropout,
                                kv_in_ch=hidden_dim)
        self.up2 = ResBlockAttn(hidden_dim*4, hidden_dim,   emb_dim=hidden_dim,
                                attn_heads=attn_heads, dropout=dropout,
                                kv_in_ch=hidden_dim)
        self.up1 = ResBlockAttn(hidden_dim*2, hidden_dim,   emb_dim=hidden_dim,
                                attn_heads=attn_heads, dropout=dropout,
                                kv_in_ch=hidden_dim)

        self.upsample   = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.final_proj = nn.Conv1d(hidden_dim, 1, 1)
        self.max_len    = max_len

    def _make_emb(self, temb, h_lstm):
        if h_lstm.dim() == 3:
            h = h_lstm[0]
        else:
            h = h_lstm
        return self.time_proj(temb) + self.lstm_proj(h)

    def forward(self, x, temb, h_lstm=None, future_pcp=None, static_attr=None):
        B, H, _ = x.shape
        assert H <= self.max_len

        x = x.permute(0,2,1)
        h = self.input_proj(x)
        emb = self._make_emb(temb, h_lstm)

        # full‑res precip K/V
        pcp_kv_full = None
        if future_pcp is not None:
            pcp_kv_full = self.precip_proj(future_pcp.permute(0,2,1))

        # encoder
        d1 = self.down1(h, emb, pcp_kv_full)
        d2 = self.down2(self.pool(d1), emb,
                        None if pcp_kv_full is None else self.pool(pcp_kv_full))
        d3 = self.down3(self.pool(d2), emb,
                        None if pcp_kv_full is None else self.pool(self.pool(pcp_kv_full)))

        # bottleneck
        bott = self.bottleneck(self.pool(d3), emb,
                               None if pcp_kv_full is None else self.pool(self.pool(self.pool(pcp_kv_full))))

        # static conditioning
        if static_attr is not None:
            s_emb = self.static_proj(static_attr.mean(dim=1)).unsqueeze(-1)
            bott = bott + s_emb

        # decoder
        u3 = self.up3(torch.cat([self.upsample(bott), d3],1), emb,
                      None if pcp_kv_full is None else self.pool(self.pool(pcp_kv_full)))
        u2 = self.up2(torch.cat([self.upsample(u3), d2],1), emb,
                      None if pcp_kv_full is None else self.pool(pcp_kv_full))
        u1 = self.up1(torch.cat([self.upsample(u2), d1],1), emb, pcp_kv_full)

        out = self.final_proj(u1)
        return out.permute(0,2,1)
