import pytorch_lightning as pl
from torch.utils.data import DataLoader
from constants import *
from dataset import ThermalBlinkDataset
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor


class ReshuffleCallback(Callback):
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset

    def on_epoch_start(self, trainer, pl_module):
        self.train_dataset.reshuffle_segments()  # reshuffle before each epoch
        print("[INFO] Training dataset reshuffled for the new epoch.")

def build_dataloaders():
    train = ThermalBlinkDataset(
        pkl_root=PKL_ROOT,
        csv_root=CSV_ROOT,
        subfolders=SUBFOLDERS,
        split="train", 
        center_size=CENTER_SIZE,
    )
    val = ThermalBlinkDataset(
        pkl_root=PKL_ROOT,
        csv_root=CSV_ROOT,
        subfolders=SUBFOLDERS,
        split="val",  
        center_size=CENTER_SIZE,
    )

    train_loader = DataLoader(
        train,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val,
        batch_size=VAL_BATCH_SIZE,
        num_workers=4,
        shuffle=False,  
        pin_memory=True
    )

    return train_loader, val_loader, train



def build_trainer(train, callbacks=None):
    """
    Build PyTorch Lightning Trainer
    Args:
        train: Training dataset
        callbacks: List of callbacks (e.g., EarlyStopping, ReshuffleCallback)
    Returns:
        Trainer object
    """
    reshuffle_callback = ReshuffleCallback(train)

    # if other callbacks are provided, append reshuffle callback
    if callbacks is None:
        callbacks = []
    callbacks.append(reshuffle_callback)

    # Add LearningRateMonitor callback
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    return pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="cuda",
        devices=1,
        default_root_dir=LOG_DIR,
        log_every_n_steps=10,
        enable_checkpointing=True,
        precision="16-mixed",  # mixed precision training
        enable_progress_bar=True,
        callbacks=callbacks,  
    )