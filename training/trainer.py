import pytorch_lightning as pl
from torch.utils.data import DataLoader
from constants import *
from dataset import ThermalBlinkDataset
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

def build_dataloaders():
    train_ds = ThermalBlinkDataset(
        pkl_root=PKL_ROOT,
        csv_root=CSV_ROOT,
        subfolders=SUBFOLDERS,
        val_pkl_dir=VAL_PKL_DIR,
        val_csv_dir=VAL_CSV_DIR,
        is_val=False,
        center_size=CENTER_SIZE,
    )
    val_ds = ThermalBlinkDataset(
        pkl_root=PKL_ROOT,
        csv_root=CSV_ROOT,
        subfolders=SUBFOLDERS,
        val_pkl_dir=VAL_PKL_DIR,
        val_csv_dir=VAL_CSV_DIR,
        is_val=True,
        center_size=CENTER_SIZE,
    )

    # 计算每个类别的采样权重，平衡类别不均衡
    labels = np.array([y for (_, y) in train_ds.data])
    class_counts = np.array([(labels == i).sum() for i in range(NUM_CLASSES)])
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    return train_loader, val_loader

def build_trainer():
    tb_logger = TensorBoardLogger(save_dir=LOG_DIR, name="blink_tb")
    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    return pl.Trainer(
        max_epochs=EPOCHS,
        accelerator=DEVICE,
        devices=1,
        default_root_dir=LOG_DIR,
        log_every_n_steps=10,
        enable_checkpointing=True,
        logger=tb_logger,
        callbacks=[early_stop, lr_monitor],
    )
