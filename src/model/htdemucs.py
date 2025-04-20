import torch
import torch.nn as nn

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


class HTDemucs(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [48, 96, 192, 384, 768, 1536]
        self.encoder = nn.ModuleList()
        in_ch = 1
        for out_ch in channels:
            self.encoder.append(EncoderBlock(in_ch, out_ch))
            in_ch = out_ch

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x
