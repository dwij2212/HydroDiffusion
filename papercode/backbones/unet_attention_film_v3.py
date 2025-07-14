import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMBlock(nn.Module):
    def __init__(self, channels, emb_dim, dropout=0.1):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.norm    = nn.GroupNorm(groups, channels)
        self.fc      = nn.Linear(emb_dim, 2 * channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, temb):
        # temb is only the time embedding now
        h = self.norm(x)
        scale, shift = self.fc(temb).chunk(2, dim=1)
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
        B, C, L = x.shape
        assert L <= self.pos_enc.size(1), f"Sequence length {L} > {self.pos_enc.size(1)}"
        q = x.permute(0,2,1) + self.pos_enc[:, :L, :]
        k = kv.permute(0,2,1) + self.pos_enc[:, :L, :]
        v = k
        a, _ = self.attn(q, k, v)
        a = self.drop(self.ln(a))
        return x + self.alpha * a.permute(0,2,1)


class ResBlockAttn(nn.Module):
    def __init__(self, in_ch, out_ch, film_emb_dim,
                 attn_heads=4, dropout=0.1, dilation=1,
                 kv_in_ch=None):
        super().__init__()
        pad = (3 - 1)//2 * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=pad, dilation=dilation)
        self.gn1   = nn.GroupNorm(8 if out_ch%8==0 else 1, out_ch)
        self.act1  = nn.SiLU()
        self.film  = FiLMBlock(out_ch, film_emb_dim, dropout)

        self.kv_proj = None
        if kv_in_ch is not None:
            self.kv_proj = nn.Conv1d(kv_in_ch, out_ch, 1)

        self.attn  = CrossAttn1d(out_ch, heads=attn_heads, dropout=dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=pad, dilation=dilation)
        self.gn2   = nn.GroupNorm(8 if out_ch%8==0 else 1, out_ch)
        self.act2  = nn.SiLU()
        self.drop2 = nn.Dropout(dropout)

        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb, pcp_kv=None):
        # temb: (B, film_emb_dim)
        h = self.act1(self.gn1(self.conv1(x)))
        h = self.film(h, temb)

        if pcp_kv is not None:
            kv = self.kv_proj(pcp_kv) if self.kv_proj else pcp_kv
            h = self.attn(h, kv)

        h = self.drop2(self.act2(self.gn2(self.conv2(h))))
        return self.skip(x) + h


class unet_attention_film_v3(nn.Module):
    def __init__(self,
                 input_dim=1,
                 hidden_dim=64,
                 static_dim=27,
                 h_lstm_dim=256,
                 dropout=0.1,
                 attn_heads=4,
                 max_len=1024):
        super().__init__()
        # projections
        self.input_proj  = nn.Conv1d(input_dim, hidden_dim, 1)
        self.time_proj   = nn.Linear(h_lstm_dim, hidden_dim)
        self.lstm_proj   = nn.Linear(h_lstm_dim, hidden_dim * 4) 
        self.static_proj = nn.Linear(static_dim, hidden_dim * 4)
        self.precip_proj = nn.Conv1d(1, hidden_dim, 3, padding=1)

        # Encoder blocks (film_emb_dim = hidden_dim)
        self.down1 = ResBlockAttn(hidden_dim, hidden_dim, hidden_dim,
                                  attn_heads=attn_heads, dropout=dropout,
                                  kv_in_ch=hidden_dim)
        self.down2 = ResBlockAttn(hidden_dim, hidden_dim*2, hidden_dim,
                                  attn_heads=attn_heads, dropout=dropout,
                                  kv_in_ch=hidden_dim)
        self.down3 = ResBlockAttn(hidden_dim*2, hidden_dim*4, hidden_dim,
                                  attn_heads=attn_heads, dropout=dropout,
                                  kv_in_ch=hidden_dim)
        self.pool  = nn.AvgPool1d(2,2)

        # Bottleneck block
        self.bottleneck = ResBlockAttn(hidden_dim*4, hidden_dim*4, hidden_dim,
                                       attn_heads=attn_heads, dropout=dropout,
                                       dilation=2,
                                       kv_in_ch=hidden_dim)

        # Decoder blocks
        self.up3 = ResBlockAttn(hidden_dim*8, hidden_dim*2, hidden_dim,
                                attn_heads=attn_heads, dropout=dropout,
                                kv_in_ch=hidden_dim)
        self.up2 = ResBlockAttn(hidden_dim*4, hidden_dim,   hidden_dim,
                                attn_heads=attn_heads, dropout=dropout,
                                kv_in_ch=hidden_dim)
        self.up1 = ResBlockAttn(hidden_dim*2, hidden_dim,   hidden_dim,
                                attn_heads=attn_heads, dropout=dropout,
                                kv_in_ch=hidden_dim)

        self.upsample   = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.final_proj = nn.Conv1d(hidden_dim, 1, 1)
        self.max_len    = max_len

    def forward(self, x, temb, h_lstm=None, future_pcp=None, static_attr=None):
        """
        x:          (B, H, 1)
        temb:       (B, h_lstm_dim)
        h_lstm:     either (1, B, h_lstm_dim) or (B, h_lstm_dim)
        future_pcp: (B, H, 1)
        static_attr:(B, seq, static_dim)
        """
        B, H, _ = x.shape
        assert H <= self.max_len

        # reshape & project inputs
        x = x.permute(0,2,1)       # ? (B,1,H)
        h = self.input_proj(x)     # ? (B,hidden,H)

        # build film embedding (time-only)
        # both film and attention blocks use this
        film_emb = self.time_proj(temb)

        # build lstm embedding (for bottleneck only)
        if h_lstm is not None:
            h_vec = h_lstm[0] if h_lstm.dim()==3 else h_lstm
            lstm_emb = self.lstm_proj(h_vec)  # (B, hidden_dim)
        else:
            lstm_emb = None

        # precip key/value (full res)
        pcp_kv_full = None
        if future_pcp is not None:
            pcp_kv_full = self.precip_proj(future_pcp.permute(0,2,1))

        # --- Encoder ----------------------------------------------------------
        d1 = self.down1(h, film_emb, pcp_kv_full)                                                    # H
        d2 = self.down2(self.pool(d1), film_emb,
                        None if pcp_kv_full is None else self.pool(pcp_kv_full))                    # H/2
        d3 = self.down3(self.pool(d2), film_emb,
                        None if pcp_kv_full is None else self.pool(self.pool(pcp_kv_full)))         # H/4

        # --- Bottleneck ------------------------------------------------------
        bott = self.bottleneck(self.pool(d3), film_emb,
                               None if pcp_kv_full is None else self.pool(self.pool(self.pool(pcp_kv_full))) )  # H/8

        # add LSTM hidden embedding once, here in the bottleneck
        if lstm_emb is not None:
            bott = bott + lstm_emb.unsqueeze(-1)  # broadcast across the H/8 length

        # static conditioning
        if static_attr is not None:
            s_emb = self.static_proj(static_attr.mean(dim=1)).unsqueeze(-1)
            bott = bott + s_emb

        # --- Decoder ----------------------------------------------------------
        u3 = self.up3(torch.cat([self.upsample(bott), d3],1), film_emb,
                      None if pcp_kv_full is None else self.pool(self.pool(pcp_kv_full)))
        u2 = self.up2(torch.cat([self.upsample(u3), d2],1), film_emb,
                      None if pcp_kv_full is None else self.pool(pcp_kv_full))
        u1 = self.up1(torch.cat([self.upsample(u2), d1],1), film_emb, pcp_kv_full)

        out = self.final_proj(u1)    # (B,1,H)
        return out.permute(0,2,1)    # ? (B,H,1)
