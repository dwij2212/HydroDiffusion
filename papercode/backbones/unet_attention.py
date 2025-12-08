import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

# --------------------------------------------------------------------------
# Utility: 1-D ConvBlock
# --------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.0):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)

# --------------------------------------------------------------------------
# Unet with **early cross-attention** (after d1)
# --------------------------------------------------------------------------
class unet_attention(nn.Module):
    def __init__(
        self,
        input_dim:  int = 1,
        hidden_dim: int = 64,
        static_dim: int = 27,
        h_lstm_dim: int = 256,
        dropout:    float = 0.0,
        attn_heads: int = 4,
    ):
        super().__init__()

        # -------- stem ------------------------------------------------------
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        # -------- first encoder block (will host attention) -----------------
        self.down1 = ConvBlock(hidden_dim, hidden_dim, dropout=dropout)

        # Cross-attention at full horizon H
        self.pcp_embed1  = nn.Conv1d(1, hidden_dim, kernel_size=1)  # K,V embed
        self.cross_attn1 = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=attn_heads, batch_first=True
        )
        
        self.pos_enc    = nn.Parameter(torch.randn(1, 1024, hidden_dim) * 0.02)
        self.alpha_attn = nn.Parameter(torch.tensor(0.1))
        self.ln_attn    = nn.LayerNorm(hidden_dim)
        self.attn_drop  = nn.Dropout(0.1)
        
        # -------- remaining encoder ----------------------------------------
        self.pool = nn.AvgPool1d(2, 2)
        self.down2 = ConvBlock(hidden_dim, hidden_dim * 2, dropout=dropout)
        self.down3 = ConvBlock(hidden_dim * 2, hidden_dim * 4, dropout=dropout)

        # -------- bottleneck (no precip attention here) --------------------
        self.bottleneck = ConvBlock(
            hidden_dim * 4, hidden_dim * 4, dilation=2, dropout=dropout
        )

        # Conditioning projections (temporal & static)
        self.temb_proj   = nn.Linear(h_lstm_dim, hidden_dim * 4)
        self.static_proj = nn.Linear(static_dim, hidden_dim * 4)

        # -------- decoder ---------------------------------------------------
        self.up3 = ConvBlock(hidden_dim * 8, hidden_dim * 2, dropout=dropout)
        self.up2 = ConvBlock(hidden_dim * 4, hidden_dim,     dropout=dropout)
        self.up1 = ConvBlock(hidden_dim * 2, hidden_dim,     dropout=dropout)
        self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.final_proj = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    # ---- helper: align precip length to target_len (linear interp) ---------
    @staticmethod
    def _align_seq(seq, target_len: int):
        B, L, C = seq.shape
        if L == target_len:
            return seq
        return F.interpolate(seq.permute(0, 2, 1), size=target_len,
                             mode="linear", align_corners=True).permute(0, 2, 1)

    # ----------------------------------------------------------------------
    def forward(
        self,
        x,                   # (B, H, 1)
        temb,                # (B, h_lstm_dim)
        h_lstm=None,         # optional extra temporal cond
        future_pcp=None,     # (B, H_fut, 1)
        static_attr=None,    # (B, H, static_dim)
    ):
        B, H, _ = x.shape

        # ---- stem ---------------------------------------------------------
        x = self.input_proj(x.permute(0, 2, 1))   # (B, hidden_dim, H)

        # ---- first conv block --------------------------------------------
        d1 = self.down1(x)                        # (B, hidden_dim, H)

        # ---- EARLY cross-attention ---------------------------------------
        if future_pcp is not None:
            # Align precip length to H (if different)
            pcp = self._align_seq(future_pcp, H)          # (B, H, 1)
            kv  = self.pcp_embed1(pcp.permute(0, 2, 1))   # (B, hidden_dim, H)
            
            pos = self.pos_enc[:, :H, :]                            # (1,H,C)
            q   = (d1.permute(0,2,1) + pos)                         # (B,H,C)
            kv  = (kv.permute(0,2,1) + pos)
            
            #print("q+posenc std:", q.std().item(), "kv+posenc std:", kv.std().item())
            
            attn_out, _ = self.cross_attn1(q, kv, kv)
            attn_out = self.attn_drop(self.ln_attn(attn_out))
            d1 = d1 + self.alpha_attn * attn_out.permute(0,2,1)
            
            '''
            # (B, L, C) format for MHA
            q  = d1.permute(0, 2, 1)   # (B, H, hidden_dim)
            kv = kv.permute(0, 2, 1)   # (B, H, hidden_dim)

            attn_out, _ = self.cross_attn1(q, kv, kv)     # (B, H, hidden_dim)
            d1 = d1 + attn_out.permute(0, 2, 1)           # residual add
            '''
            
        # ---- rest of the encoder -----------------------------------------
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))

        # ---- bottleneck ---------------------------------------------------
        bot = self.bottleneck(self.pool(d3))       # (B, C4, H/8)

        # ---- explicit (scalar) conditioning ------------------------------
        cond = self.temb_proj(temb).unsqueeze(-1)  # (B, C4, 1)
        if h_lstm is not None:
            cond += self.temb_proj(h_lstm.squeeze(0)).unsqueeze(-1)
        if static_attr is not None:
            static_mean = static_attr.mean(dim=1)  # (B, static_dim)
            cond += self.static_proj(static_mean).unsqueeze(-1)
        bot = bot + cond

        # ---- decoder ------------------------------------------------------
        u3 = self.upsample(bot)
        u3 = self.up3(torch.cat([u3, d3], dim=1))

        u2 = self.upsample(u3)
        u2 = self.up2(torch.cat([u2, d2], dim=1))

        u1 = self.upsample(u2)
        u1 = self.up1(torch.cat([u1, d1], dim=1))

        out = self.final_proj(u1).permute(0, 2, 1)  # (B, H, 1)
        return out
