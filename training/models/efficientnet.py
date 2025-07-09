import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision.models.efficientnet import EfficientNet_B0_Weights
from constants import FRAME_STACK_SIZE


class EfficientNetBlinkModel(nn.Module):
    def __init__(self, in_channels=FRAME_STACK_SIZE, num_classes=1, dropout_prob=0.3):
        super().__init__()

        # 加载预训练 EfficientNetB0
        self.base_model = efficientnet_b0(weights=None)  # 如果需要预训练则改为 EfficientNet_B0_Weights.IMAGENET1K_V1

        # 修改第一层以适应自定义通道数
        out_channels = self.base_model.features[0][0].out_channels
        self.base_model.features[0][0] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        # 获取全连接前的特征维度
        in_features = self.base_model.classifier[1].in_features

        # 替换分类器
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.base_model(x).squeeze(-1)  # 输出为 [B]

