# src/model/htdemucs.py

import torch
import torch.nn as nn

class HTDemucs(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: add layers based on the paper
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=8, stride=4, padding=2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 1, kernel_size=8, stride=4, padding=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x