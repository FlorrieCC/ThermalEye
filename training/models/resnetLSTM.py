import torch
import torch.nn as nn
import torchvision.models as models
from constants import FRAME_STACK_SIZE

class ResNetLSTM(nn.Module):
    def __init__(self, in_channels=1, hidden_size=64, num_layers=1, num_classes=1):
        super().__init__()

        # 使用ResNet18作为特征提取器，只用前两层
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2
        )

        # 每帧的空间特征扁平化维度
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = resnet.layer2[-1].conv2.out_channels

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=False)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # 输入 x: [B, C=T, H, W]，我们假设 C 是堆叠的 T 帧
        B, C, H, W = x.shape
        T = C  # 每个通道代表一帧
        x = x.view(B, T, 1, H, W)  # 恢复为 [B, T, 1, H, W]

        x = x.view(B * T, 1, H, W)
        x = self.backbone(x)
        x = self.pool(x).view(B, T, -1)
        lstm_out, _ = self.lstm(x)
        out = self.classifier(lstm_out).squeeze(-1)
        return out

