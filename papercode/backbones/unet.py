import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.0):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.block = nn.Sequential(
            nn.Conv1d(in_channels,  out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.block(x)

class unet(nn.Module):
    def __init__(
        self,
        input_dim:   int = 1,
        hidden_dim:  int = 32,
        static_dim:  int = 27,
        h_lstm_dim:  int = 256,
        dropout:   float = 0.0,
    ):
        super().__init__()

        # initial 1×1 projection
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        # Downsampling path with dropout in each ConvBlock
        self.down1 = ConvBlock(hidden_dim,           hidden_dim,     dropout=dropout)
        self.down2 = ConvBlock(hidden_dim,           hidden_dim * 2, dropout=dropout)
        self.down3 = ConvBlock(hidden_dim * 2,       hidden_dim * 4, dropout=dropout)

        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

        # Bottleneck (with dropout baked into ConvBlock)
        self.bottleneck = ConvBlock(hidden_dim * 4, hidden_dim * 4, dilation=2, dropout=dropout)

        # Conditioning projections
        self.temb_proj = nn.Linear(h_lstm_dim, hidden_dim * 4)
        self.hproj = nn.Linear(h_lstm_dim, hidden_dim * 4)
        self.static_proj = nn.Linear(static_dim,    hidden_dim * 4)
        self.precip_proj = nn.Conv1d(1,            hidden_dim * 4, kernel_size=1)

        # Upsampling path
        self.up3 = ConvBlock(hidden_dim * 8, hidden_dim * 2, dropout=dropout)
        self.up2 = ConvBlock(hidden_dim * 4, hidden_dim,     dropout=dropout)
        self.up1 = ConvBlock(hidden_dim * 2, hidden_dim,     dropout=dropout)

        self.up_sample  = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.final_proj = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, x, temb, h_lstm=None, future_pcp=None, static_attr=None):
        """
        Args:
            x:          (B, H, 1)
            temb:       (B, h_lstm_dim)
            h_lstm:     (1, B, h_lstm_dim)
            future_pcp: (B, H, 1)
            static_attr:(B, H, static_dim)
        """
        B, H, _ = x.shape

        # (B, 1, H) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)  # (B, hidden_dim, H)

        # Down path
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(d3))

        # Conditioning at bottleneck
        cond = self.temb_proj(temb).unsqueeze(-1)  # (B, hidden_dim*4, 1)
        if h_lstm is not None:
            cond += self.hproj(h_lstm.squeeze(0)).unsqueeze(-1)
        if static_attr is not None:
            static_mean = static_attr.mean(dim=1)
            cond += self.static_proj(static_mean).unsqueeze(-1)
        if future_pcp is not None:
            pcp_emb = self.precip_proj(future_pcp.permute(0, 2, 1))
            # down-sample to match (B, hidden_dim*4, H/8)
            pcp_emb = self.pool(self.pool(self.pool(pcp_emb)))
            cond += pcp_emb
        bottleneck = bottleneck + cond

        # Up path
        u3 = self.up_sample(bottleneck)
        u3 = self.up3(torch.cat([u3, d3], dim=1))
        u2 = self.up_sample(u3)
        u2 = self.up2(torch.cat([u2, d2], dim=1))
        u1 = self.up_sample(u2)
        u1 = self.up1(torch.cat([u1, d1], dim=1))

        # Final projection back to (B, H, 1)
        out = self.final_proj(u1)    # (B, 1, H)
        out = out.permute(0, 2, 1)   # (B, H, 1)
        return out
