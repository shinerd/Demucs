import torch
import torch.nn as nn

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, channels, max_len=2048):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(1, channels, max_len))

    def forward(self, x):
        B, C, T = x.size()
        pos_emb = self.positional_embedding[:, :, :T]
        return x + pos_emb

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=8, stride=4, padding=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)               # (B, C, T)
        x = self.activation(x)
        x = x.transpose(1, 2)          # (B, C, T) → (B, T, C)
        x = self.norm(x)               # Normalize over channels
        x = x.transpose(1, 2)          # (B, T, C) → (B, C, T)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model=1536, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.pos_enc = LearnablePositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True  # (B, T, C) 형식으로 입력 가능
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = x.transpose(1, 2)            # (B, C, T) → (B, T, C)
        x = self.pos_enc(x.transpose(1, 2)).transpose(1, 2)  # positional encoding
        x = self.transformer(x)
        x = x.transpose(1, 2)            # (B, T, C) → (B, C, T)
        return x

class HTDemucs(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [48, 96, 192, 384, 768, 1536]
        self.encoder = nn.ModuleList()
        in_ch = 1
        for out_ch in channels:
            self.encoder.append(EncoderBlock(in_ch, out_ch))
            in_ch = out_ch

        self.transformer = TransformerBlock(d_model=1536, nhead=8, num_layers=6)

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x
