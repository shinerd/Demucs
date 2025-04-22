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
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=8, stride=4, padding=2, output_padding=0):    # 일단 output_padding을 0으로 두고 skip 때 조정
        super().__init__()
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.deconv(x)              # (B, C_in, T) → (B, C_out, T_up)
        x = self.activation(x)
        x = x.transpose(1, 2)           # (B, C, T) → (B, T, C)
        x = self.norm(x)
        x = x.transpose(1, 2)           # (B, T, C) → (B, C, T)
        return x

class HTDemucs(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config["model"]

        self.encoder_channels = model_cfg["encoder_channels"]
        self.decoder_channels = list(reversed(self.encoder_channels[:-1])) # 마지막 1536은 그대로 유지

        self.kernel_size = model_cfg["kernel_size"]
        self.stride = model_cfg["stride"]
        self.padding = model_cfg["padding"]

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        in_ch = 1
        for out_ch in self.encoder_channels:
            self.encoder.append(EncoderBlock(in_ch, out_ch, self.kernel_size, self.stride, self.padding))
            in_ch = out_ch

        tf_cfg = model_cfg["transformer"]
        self.transformer = TransformerBlock(d_model=tf_cfg["d_model"], nhead=tf_cfg["nhead"], num_layers=tf_cfg["num_layers"], dropout=tf_cfg["dropout"])

        in_ch = self.encoder_channels[-1]
        for out_ch in self.decoder_channels:
            self.decoder.append(DecoderBlock(in_ch + out_ch, out_ch, self.kernel_size, self.stride, self.padding)) # skip connection으로 채널 2배
            in_ch = out_ch
        
        self.final_conv = nn.Conv1d(self.decoder_channels[-1], 1, kernel_size=1)

    def forward(self, x):
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)

        x = self.transformer(x)
        skips = skips[:-1][::-1]  # 맨 마지막 encoder output은 transformer input이므로 제외하고 역순

        for i, layer in enumerate(self.decoder):
            skip = skips[i]
            if skip.shape[-1] != x.shape[-1]:
                # 시간축 맞추기
                diff = skip.shape[-1] - x.shape[-1]
                if diff > 0:
                    skip = skip[:, :, :-diff]  # 중심 crop
                elif diff < 0:
                    skip = nn.functional.interpolate(skip, size=x.shape[-1], mode='linear', align_corners=False) #interpolation
            x = torch.cat([x, skip], dim=1)
            x = layer(x)

        return self.final_conv(x)