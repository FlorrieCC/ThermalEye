import torch
import torch.nn as nn
import pytorch_lightning as pl
from constants import *
from models.get_model import get_model  # 👈 使用统一接口加载模型
from torchmetrics import MeanSquaredError, MeanAbsoluteError


class BlinkClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = get_model(MODEL_NAME)  # 👈 通过字符串选择模型
        self.criterion = nn.MSELoss()  
        self.val_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        preds = self(x)
        loss = self.criterion(preds, y)
        self.val_mse.update(preds, y)
        self.val_mae.update(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)


    def on_validation_epoch_end(self):
        mse = self.val_mse.compute()
        mae = self.val_mae.compute()
        self.log("val/mse", mse)
        self.log("val/mae", mae)
        print(f"[VAL] MSE: {mse:.4f} | MAE: {mae:.4f}")
        self.val_mse.reset()
        self.val_mae.reset()
        
    # def on_after_backward(self):
    #     print("\n🔍 Gradient stats per parameter:")
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             print(f"{name:30s}: grad mean = {param.grad.abs().mean():.6f}")

