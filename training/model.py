import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from models.get_model import get_model
from constants import MODEL_NAME, LEARNING_RATE
import numpy as np


class BlinkClassifier(pl.LightningModule):
    def __init__(self, lr=1e-4, pos_weight=7.5):
        super().__init__()
        self.save_hyperparameters(ignore=["train_dataloader"])

        self.model = get_model(MODEL_NAME)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))  # 替换为带正样本权重的loss

        # Evaluation metrics
        self.train_mae = MeanAbsoluteError()
        self.train_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()
        self.val_mse = MeanSquaredError()

    def forward(self, x):
        return self.model(x)  # 输出logits，不加sigmoid

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        logits = self(x)
        loss = self.loss_fn(logits, y)

        probs = torch.sigmoid(logits)
        self.log("train_loss", loss, prog_bar=True)
        self.train_mae.update(probs, y)
        self.train_mse.update(probs, y)

        # Debug: 预测值统计
        self.print(f"[TRAIN] Mean Pred: {probs.mean():.4f}, Std: {probs.std():.4f}")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        logits = self(x)
        loss = self.loss_fn(logits, y)

        probs = torch.sigmoid(logits)
        self.log("val_loss", loss, prog_bar=True)
        self.val_mae.update(probs, y)
        self.val_mse.update(probs, y)
        self.print(f"[VAL] Loss: {loss.item():.4f}, Mean Pred: {probs.mean():.4f}, Std: {probs.std():.4f}")
        return loss

    def on_train_epoch_end(self):
        self.log("train_mae", self.train_mae.compute())
        self.log("train_mse", self.train_mse.compute())
        self.train_mae.reset()
        self.train_mse.reset()

    def on_validation_epoch_end(self):
        self.log("val_mae", self.val_mae.compute())
        self.log("val_mse", self.val_mse.compute())
        print(f"[VAL] MSE: {self.val_mse.compute():.4f} | MAE: {self.val_mae.compute():.4f}")
        self.val_mae.reset()
        self.val_mse.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
