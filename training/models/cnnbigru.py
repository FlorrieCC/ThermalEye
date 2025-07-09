import torch
import torch.nn as nn

class CNNBiGRUBlinkModel(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=128, num_layers=1, dropout=0.3, num_classes=1):
        super().__init__()

        # 🧠 CNN backbone（帧级特征提取器）
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),  # (B, 16, H, W)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 16, H/2, W/2)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (B, 32, H/2, W/2)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 32, H/4, W/4)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B, 64, H/4, W/4)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # (B, 64, 1, 1)
        )

        self.feature_dim = 64  # 输出通道数

        # 📦 BiGRU (时间序列建模)
        self.bigru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 🔮 全连接分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)  # 最终输出 logits
        )

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # (B*T, C, H, W)
        feat = self.cnn(x)  # (B*T, F, 1, 1)
        feat = feat.view(B, T, self.feature_dim)  # (B, T, F)

        output, _ = self.bigru(feat)  # (B, T, 2*H)
        logits = self.classifier(output)  # (B, T, 1)
        return logits.squeeze(-1)  # → (B, T)
