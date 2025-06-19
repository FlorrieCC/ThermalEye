import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from models.get_model import get_model
from constants import MODEL_NAME, LEARNING_RATE
import numpy as np


class BlinkClassifier(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=["train_dataloader"])

        self.model = get_model(MODEL_NAME)
        self.loss_fn = nn.BCELoss()

        # Evaluation metrics
        self.train_mae = MeanAbsoluteError()
        self.train_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()
        self.val_mse = MeanSquaredError()

    def forward(self, x):
        return torch.sigmoid(self.model(x))  # [B, T] with sigmoid activation

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        self.train_mae.update(pred, y)
        self.train_mse.update(pred, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log("val_loss", loss, prog_bar=True)
        self.val_mae.update(pred, y)
        self.val_mse.update(pred, y)
        self.print(f"[VAL] Loss: {loss.item():.4f}, Mean Pred: {pred.mean():.4f}, Std: {pred.std():.4f}")
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
