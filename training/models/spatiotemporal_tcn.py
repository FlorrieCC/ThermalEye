import torch
import torch.nn as nn

class SpatioTemporalTCN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.tcn = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.classifier = nn.Linear(8, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = self.flatten(x)  # [B*T, 8]
        x = x.view(B, T, -1).permute(0, 2, 1)  # [B, 8, T]
        x = self.tcn(x).permute(0, 2, 1)       # [B, T, 8]
        y = self.classifier(x).squeeze(-1)     # [B, T]
        return y
