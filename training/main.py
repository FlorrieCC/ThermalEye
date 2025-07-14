import json
import os
import pandas as pd
import torch
from constants import *
from dataset import ThermalBlinkDataset
from evaluate import evaluate_model
from model import BlinkClassifier
from trainer import build_trainer, build_dataloaders
from utils import set_random_seeds
from pytorch_lightning.callbacks import EarlyStopping

def main():
    set_random_seeds()

    print("🛠️ Loading training data...")
    dataset = ThermalBlinkDataset(
        pkl_root=PKL_ROOT,
        csv_root=CSV_ROOT,
        subfolders=SUBFOLDERS,
        split="train",
        center_size=CENTER_SIZE,
    )
    print(f"✅ Dataset loaded successfully, total {len(dataset)} frames")

    print("\n🚀 Initializing model: ", MODEL_NAME)
    model = BlinkClassifier()

    # Build dataloaders
    train_loader, val_loader, train_dataset = build_dataloaders()

    # Configure early stopping
    early_stopping = EarlyStopping(
        monitor="val_f1",
        patience=5,
        mode="max",
        verbose=True
    )

    # Build trainer with reshuffle, early stopping, and learning rate monitor callbacks
    # trainer = build_trainer(train_dataset, callbacks=[early_stopping])
    trainer = build_trainer(train_dataset)
    trainer.fit(model, train_loader, val_loader)

    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    model_save_path = os.path.join(CHECKPOINT_PATH, "res_t1.pth")
    torch.save(model.model.state_dict(), model_save_path)
    print(f"\n✅ Model saved to {model_save_path}")

    print("\n📊 Starting model evaluation...")
    ckpt_path = os.path.join(CHECKPOINT_PATH, "res_t1.pth")
    evaluate_model(ckpt_path)

if __name__ == '__main__':
    main()