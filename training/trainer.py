import pytorch_lightning as pl
from torch.utils.data import DataLoader
from constants import *
from dataset import ThermalBlinkDataset
<<<<<<< Updated upstream
=======
import multiprocessing as mp

def worker_init_fn(worker_id):
    # 获取当前工作进程的信息
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        # 调用数据集的 set_worker_id 方法
        dataset = worker_info.dataset
        if hasattr(dataset, 'set_worker_id'):
            dataset.set_worker_id(worker_id)
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
    return (
        DataLoader(train, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val, batch_size=BATCH_SIZE)
    )

=======

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader

>>>>>>> Stashed changes
def build_trainer():
    return pl.Trainer(
        max_epochs=EPOCHS,
        accelerator=DEVICE,
        devices=1,
        default_root_dir=LOG_DIR,
        log_every_n_steps=10,
        enable_checkpointing=True,
<<<<<<< Updated upstream
    )
=======
        precision="16-mixed",      # 混合精度训练
        strategy="auto",  # 多GPU支持
        enable_progress_bar=True
    )
>>>>>>> Stashed changes
