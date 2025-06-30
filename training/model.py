import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import F1Score 
from models.get_model import get_model
<<<<<<< Updated upstream
from constants import MODEL_NAME, LEARNING_RATE
=======
from constants import *
>>>>>>> Stashed changes
import numpy as np


class BlinkClassifier(pl.LightningModule):
    def __init__(self, lr=1e-4, pos_weight=POS_WEIGHT):
        super().__init__()
        self.save_hyperparameters(ignore=["train_dataloader"])
<<<<<<< Updated upstream

        self.model = get_model(MODEL_NAME)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))  # 替换为带正样本权重的loss

        # Evaluation metrics
        self.train_mae = MeanAbsoluteError()
        self.train_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()
        self.val_mse = MeanSquaredError()

    def forward(self, x):
        return self.model(x)  # 输出logits，不加sigmoid
=======
        self.model = get_model(MODEL_NAME)

        self.register_buffer("target_device", torch.tensor([]))
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))  # 替换为带正样本权重的loss

        # Evaluation metrics
        self.train_f1 = F1Score(task="binary")  # 二分类任务
        self.val_f1 = F1Score(task="binary")   # 验证集 F1

        self.input_adapter = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1)  # 简单的1x1卷积用于类型转换
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x):
        out = self.model(x)  
        return out

>>>>>>> Stashed changes

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        logits = self(x)
        loss = self.loss_fn(logits, y)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long() 
        self.train_f1.update(preds, y.long())  # y 必须是整数类型（0/1）
        self.log("train_loss", loss, prog_bar=True)
<<<<<<< Updated upstream
        self.train_mae.update(probs, y)
        self.train_mse.update(probs, y)
=======
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", lr, prog_bar=True, on_step=True, on_epoch=True)
>>>>>>> Stashed changes

        # Debug: 预测值统计
        self.print(f"[TRAIN] Mean Pred: {probs.mean():.4f}, Std: {probs.std():.4f}")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        logits = self(x)
        y = y.float().unsqueeze(-1)  # [B] -> [B, 1]
        loss = self.loss_fn(logits, y)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()  # 二分类预测（0/1）
        
        self.val_f1.update(preds, y.long())
        self.log("val_loss", loss, prog_bar=True)
        self.print(f"[VAL] Loss: {loss.item():.4f}, Mean Pred: {probs.mean():.4f}, Std: {probs.std():.4f}")
        return loss

    def on_train_epoch_end(self):
        self.log("train_f1", self.train_f1.compute(), prog_bar=True)
        self.train_f1.reset()  # 重置指标，避免跨 epoch 累积

    def on_validation_epoch_end(self):
        val_f1 = self.val_f1.compute()
        self.log("val_f1", val_f1, prog_bar=True)
        print(f"[VAL] F1 Score: {val_f1:.4f}")  # 打印 F1
        self.val_f1.reset()  # 重置

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # 以StepLR为例，每5个epoch将lr乘以0.5
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # 每个epoch调整一次
                "monitor": "val_loss",  # 可选，某些scheduler需要
            }
        }