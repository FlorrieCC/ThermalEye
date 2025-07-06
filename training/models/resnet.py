import torch
import torch.nn as nn
import torchvision.models as models
from constants import FRAME_STACK_SIZE  

class ResNetBlink(nn.Module):
    def __init__(self, in_channels=FRAME_STACK_SIZE, num_classes=1):
        super().__init__()
        self.base_model = models.resnet18(pretrained=False)
        self._modify_first_conv(in_channels)
        self.base_model.maxpool = nn.Identity()
        self.global_dropout = nn.Dropout(0.2)

        # Only keep layer1 and layer2
        # Dynamically get the output channels of layer2
        num_ftrs = self.base_model.layer2[-1].conv2.out_channels  # 128 for resnet18 after layer2

        # Define new fc
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        self.global_dropout = nn.Dropout(0.2)
        # Use only layer1 and layer2, so fc input features should match layer2 output
        num_ftrs = self.base_model.layer2[-1].conv2.out_channels  # 128 for resnet18 after layer2
        self.fc = nn.Linear(num_ftrs, num_classes)

    def _modify_first_conv(self, in_channels):
        self.base_model.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.base_model.bn1 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.global_dropout(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        # Skip layer3/layer4
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.global_dropout(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        # Skip layer3 and layer4
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x