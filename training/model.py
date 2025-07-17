import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import F1Score 
from models.get_model import get_model
from constants import *
import numpy as np


class BlinkClassifier(pl.LightningModule):
    def __init__(self, lr=1e-4, pos_weight=POS_WEIGHT):
        super().__init__()
        self.save_hyperparameters(ignore=["train_dataloader"])
        self.model = get_model(MODEL_NAME)

        self.register_buffer("target_device", torch.tensor([]))
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))  # Replace with weighted loss for positive samples

        # Evaluation metrics
        self.train_f1 = F1Score(task="binary")  # Binary classification task
        self.val_f1 = F1Score(task="binary")    # F1 score on validation set

        self.input_adapter = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1)  # Simple 1x1 convolution for type adaptation
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x):
        out = self.model(x)  
        return out


    # def training_step(self, batch, batch_idx):
    #     x, y = batch["x"], batch["y"]
    #     logits = self(x)
    #     y = y.float().unsqueeze(-1)  # [B] -> [B, 1]
    #     loss = self.loss_fn(logits, y)
    #     probs = torch.sigmoid(logits)
    #     preds = (probs > 0.5).long() 
    #     self.train_f1.update(preds, y.long())  # y must be integer type (0/1)
    #     self.log("train_loss", loss, prog_bar=True)
    #     lr = self.trainer.optimizers[0].param_groups[0]['lr']
    #     self.log("lr", lr, prog_bar=True, on_step=True, on_epoch=True)

    #     # Debug: Prediction statistics
    #     self.print(f"[TRAIN] Mean Pred: {probs.mean():.4f}, Std: {probs.std():.4f}")
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch["x"], batch["y"]
    #     logits = self(x)
    #     y = y.float().unsqueeze(-1)  # [B] -> [B, 1]
    #     loss = self.loss_fn(logits, y)
    #     probs = torch.sigmoid(logits)
    #     preds = (probs > 0.5).long()  # Binary classification prediction (0/1)
        
    #     self.val_f1.update(preds, y.long())
    #     self.log("val_loss", loss, prog_bar=True)
    #     self.print(f"[VAL] Loss: {loss.item():.4f}, Mean Pred: {probs.mean():.4f}, Std: {probs.std():.4f}")
    #     return loss

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]          # y: [B]
        logits = self(x)                       # logits: [B]
        # y = y.float()                          # [B] Keep it as-is, no .unsqueeze()
        y = y.float().unsqueeze(-1)  # [B] -> [B, 1]
        loss = self.loss_fn(logits, y)         # [B] vs [B]
        probs = torch.sigmoid(logits)          # [B]
        preds = (probs > 0.5).long()           # [B]
        self.train_f1.update(preds, y.long())  # y must be long type

        self.log("train_loss", loss, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'],
                 prog_bar=True, on_step=True, on_epoch=True)
        self.print(f"[TRAIN] Mean Pred: {probs.mean():.4f}, Std: {probs.std():.4f}")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]          # y: [B]
        logits = self(x)                       # [B]
        # y = y.float()                          # [B]
        y = y.float().unsqueeze(-1)  # [B] -> [B, 1]
        loss = self.loss_fn(logits, y)
        probs = torch.sigmoid(logits)          # [B]
        preds = (probs > 0.5).long()           # [B]
        self.val_f1.update(preds, y.long())    # [B]
        self.log("val_loss", loss, prog_bar=True)
        self.print(f"[VAL] Loss: {loss.item():.4f}, Mean Pred: {probs.mean():.4f}, Std: {probs.std():.4f}")
        return loss

    def on_train_epoch_end(self):
        self.log("train_f1", self.train_f1.compute(), prog_bar=True)
        self.train_f1.reset()  # Reset metric to avoid accumulation across epochs

    def on_validation_epoch_end(self):
        val_f1 = self.val_f1.compute()
        self.log("val_f1", val_f1, prog_bar=True)
        print(f"[VAL] F1 Score: {val_f1:.4f}")  # Print F1 score
        self.val_f1.reset()  # Reset

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # Use ReduceLROnPlateau scheduler to adjust LR dynamically based on validation metric
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",      # Minimize validation loss
            factor=0.5,      # LR reduction factor
            patience=3,      # Wait 3 evaluations without improvement
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Monitor validation loss
            },
        }
