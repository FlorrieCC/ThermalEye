import torch
import torch.nn as nn

class threeDCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=(3, 3, 3), padding=1),  # -> [16, T, H, W]
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),                            # -> [16, T, H/2, W/2]

            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),           # -> [32, T, H/2, W/2]
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),                            # -> [32, T, H/4, W/4]
        )

        # Global pooling + dense classification per time step
        self.temporal_conv = nn.Conv3d(32, 32, kernel_size=(3, 1, 1), padding=(1, 0, 0))  # preserve [T]

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),  # -> [32, T, 1, 1]
            nn.Flatten(start_dim=2),             # -> [B, 32, T]
            nn.Conv1d(32, num_classes, kernel_size=1),  # -> [B, 1, T]
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: [B, 1, T, H, W] - Note: the input dimension order is different from TCN model
        Returns:
            y: [B, T] - Frame level binary probability output
        """
        x = self.encoder(x)            # -> [B, 32, T, H', W']
        x = self.temporal_conv(x)     # -> [B, 32, T, H', W']
        x = self.classifier(x)        # -> [B, 1, T]
        return x.squeeze(1)           # -> [B, T]
