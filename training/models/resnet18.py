import torch
import torch.nn as nn
import torchvision.models as models
from constants import FRAME_STACK_SIZE

class ResNet18BlinkModel(nn.Module):
    def __init__(self, in_channels=FRAME_STACK_SIZE, num_classes=1, dropout_prob=0.3):
        super().__init__()
        # 加载标准 resnet18 结构
        self.base_model = models.resnet18(pretrained=False)

        # 替换第一层以适应输入通道（默认是3通道RGB）
        self.base_model.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.base_model.bn1 = nn.BatchNorm2d(64)

        # 添加 Dropout 用于全连接层前的 regularization
        self.dropout = nn.Dropout(dropout_prob)

        # 修改最后分类层（fc）
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.base_model(x) # 输出为 [B, 1]
