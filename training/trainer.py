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
    return (
        DataLoader(train, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val, batch_size=BATCH_SIZE)
    )

def build_trainer():
    return pl.Trainer(
        max_epochs=EPOCHS,
        accelerator=DEVICE,
        devices=1,
        default_root_dir=LOG_DIR,
        log_every_n_steps=10,
        enable_checkpointing=True,
    )
