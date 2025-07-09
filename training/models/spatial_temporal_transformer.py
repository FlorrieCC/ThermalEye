import torch
import torch.nn as nn

class SpatialTemporalTransformer(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()

        # CNN encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Output shape: [B*T, 16, 1, 1]
        )

        # Flatten & Project to Transformer input
        self.embedding = nn.Linear(16, embed_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn(x).view(B, T, -1)        # -> [B, T, 16]
        x = self.embedding(x)                # -> [B, T, embed_dim]
        x = self.transformer(x)              # -> [B, T, embed_dim]
        logits = self.classifier(x).squeeze(-1)  # -> [B, T]
        return logits
