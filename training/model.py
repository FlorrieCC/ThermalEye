# import torch
# import torch.nn as nn
# import pytorch_lightning as pl
# from constants import *
# from models.get_model import get_model  # 👈 使用统一接口加载模型


# class BlinkClassifier(pl.LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.model = get_model(MODEL_NAME)  # 👈 通过字符串选择模型
#         self.criterion = nn.CrossEntropyLoss()  

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         x, y = batch["x"], batch["y"]
#         pred = self(x)
#         targets = y.view(-1)
#         loss = self.criterion(pred, targets)
#         self.log("train_loss", loss)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch["x"], batch["y"]
        
#         targets = y.view(-1)
#         preds = self(x)
#         loss = self.criterion(preds, targets)
#         self.log("val_loss", loss, prog_bar=True)
#         return loss

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
from constants import *  

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce)
        loss = self.alpha * (1 - p_t) ** self.gamma * ce
        return loss.mean()

class OptimizedBlinkClassifier(pl.LightningModule):
    def __init__(self, model_name='resnet18', num_classes=3, lr=LEARNING_RATE):
        super().__init__()
        self.save_hyperparameters()
        # Backbone: pretrained ResNet18 (drop last fc)
        backbone = getattr(torchvision.models, model_name)(pretrained=True)
        # adapt first conv to single-channel input if needed
        if hasattr(backbone, 'conv1') and backbone.conv1.in_channels != 1:
            old_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )
            with torch.no_grad():
                backbone.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
        layers = list(backbone.children())[:-2]  # remove avgpool & fc
        self.cnn = nn.Sequential(*layers)
        # Project to feature dim
        self.feat_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(backbone.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # Temporal model: bidirectional GRU
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        # Classifier
        self.classifier = nn.Linear(64*2, num_classes)
        # Loss
        self.criterion = FocalLoss(gamma=2.0, alpha=0.25)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.size()
        x = x.view(B*T, C, H, W)
        feat = self.cnn(x)
        feat = self.feat_proj(feat)       # [B*T, 128]
        feat = feat.view(B, T, -1)        # [B, T, 128]
        # GRU expects [B, T, feat]
        out, _ = self.gru(feat)           # [B, T, 128]
        logits = self.classifier(out)     # [B, T, num_classes]
        return logits.view(B*T, -1)       # flatten for loss

    def training_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        y = y.view(-1)
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        y = y.view(-1)
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        # 当 val_loss 停滞时，降低学习率
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        # Lightning 要求返回带 'monitor' 的字典
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  # 监控的指标
            }
        }
