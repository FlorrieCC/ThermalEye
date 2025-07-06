import pytorch_lightning as pl
from torch.utils.data import DataLoader
from constants import *
from dataset import ThermalBlinkDataset

def build_dataloaders():
    train = ThermalBlinkDataset(
        pkl_root=PKL_ROOT,
        csv_root=CSV_ROOT,
        subfolders=SUBFOLDERS,
        val_pkl_dir=VAL_PKL_DIR,
        val_csv_dir=VAL_CSV_DIR,
        is_val=False,
        center_size=CENTER_SIZE,
    )
    val = ThermalBlinkDataset(
        pkl_root=PKL_ROOT,
        csv_root=CSV_ROOT,
        subfolders=SUBFOLDERS,
        val_pkl_dir=VAL_PKL_DIR,
        val_csv_dir=VAL_CSV_DIR,
        is_val=True,
        center_size=CENTER_SIZE,
    )

    train_loader = DataLoader(
        train,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader

def build_trainer():
    return pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="cuda",  # 自动选择设备
        devices=1,
        default_root_dir=LOG_DIR,
        log_every_n_steps=10,
        enable_checkpointing=True,
        precision="16-mixed",      # 混合精度训练
        strategy="auto",  # 多GPU支持
        enable_progress_bar=True
    )
