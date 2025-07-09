# -*- coding: utf-8 -*-
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


def main():
    set_random_seeds()

    # ====================== [DEBUG] Dataset Validation ======================
    print("🛠️ Loading training data...")
    dataset = ThermalBlinkDataset(
        pkl_root=PKL_ROOT,
        csv_root=CSV_ROOT,
        subfolders=SUBFOLDERS,
        split="train",  # Specify training split
        center_size=CENTER_SIZE,
    )
    print(f"✅ Dataset loaded successfully, total {len(dataset)} frames")

    # ====================== [Training Section] ======================
    print("\n🚀 Initializing model: ", MODEL_NAME)
    model = BlinkClassifier()
    train_loader, val_loader, train_dataset = build_dataloaders()
    trainer = build_trainer(train_dataset)
    trainer.fit(model, train_loader, val_loader)

    # ✅ Save model to checkpoint path
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    model_save_path = os.path.join(CHECKPOINT_PATH, "res_t1.pth")
    torch.save(model.model.state_dict(), model_save_path)
    print(f"\n✅ Model saved to {model_save_path}")

    # ====================== [Evaluation Section] ======================
    print("\n📊 Starting model evaluation...")

    ckpt_path = os.path.join(CHECKPOINT_PATH, "res_t1.pth")
    evaluate_model(ckpt_path)

if __name__ == '__main__':
    main()