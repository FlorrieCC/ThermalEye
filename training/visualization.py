# visualize_thermal_dataset.py

import os
import random
from collections import Counter

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from dataset import ThermalBlinkDataset
from constants import (
    PKL_ROOT, CSV_ROOT, SUBFOLDERS,
    VAL_PKL_DIR, VAL_CSV_DIR,
    CENTER_SIZE, BATCH_SIZE
)

def plot_label_distribution(labels, title, save_path=None):
    counts = Counter(labels)
    classes = sorted(counts.keys())
    freqs  = [counts[c] for c in classes]

    plt.figure(figsize=(6,4))
    plt.bar(classes, freqs, tick_label=classes, color='C0')
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title(title)
    for x, y in zip(classes, freqs):
        plt.text(x, y+max(freqs)*0.01, str(y), ha='center')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved label distribution to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_samples_per_class(dataset, num_samples=5, cols=5, save_dir=None):
    # 将所有 idx 按类别分组
    idxs_by_class = {}
    for idx in range(len(dataset)):
        item = dataset[idx]
        y_seq = item["y"]    # [T] 序列标签
        # 取序列中最频繁的标签作为该序列的「主标签」
        label = Counter(y_seq.tolist()).most_common(1)[0][0]
        idxs_by_class.setdefault(label, []).append(idx)

    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    for cls, idxs in idxs_by_class.items():
        chosen = random.sample(idxs, min(len(idxs), num_samples))
        imgs = []
        for i in chosen:
            item = dataset[i]
            x_seq = item["x"]         # [T, C, H, W]
            frame = x_seq[0]          # 取序列中的第一帧 [C,H,W]
            # 单通道取第 0 维
            img = frame[0].unsqueeze(0)  # [1,H,W]
            imgs.append(img)
        grid = make_grid(torch.stack(imgs), nrow=cols, normalize=True, pad_value=1)
        plt.figure(figsize=(cols*2, num_samples*2/cols))
        plt.imshow(grid.permute(1,2,0), cmap='gray')
        plt.axis('off')
        plt.title(f"Class {cls} samples")
        if save_dir:
            path = os.path.join(save_dir, f"samples_class{cls}.png")
            plt.savefig(path, dpi=150)
            print(f"Saved samples for class {cls} to {path}")
        else:
            plt.show()
        plt.close()

def main():
    # 先加载训练集
    train_ds = ThermalBlinkDataset(
        pkl_root    = PKL_ROOT,
        csv_root    = CSV_ROOT,
        subfolders  = SUBFOLDERS,
        val_pkl_dir = VAL_PKL_DIR,
        val_csv_dir = VAL_CSV_DIR,
        is_val      = False,
        center_size = CENTER_SIZE,
        sequence_length = 32
    )
    # 再加载验证集
    val_ds = ThermalBlinkDataset(
        pkl_root    = PKL_ROOT,
        csv_root    = CSV_ROOT,
        subfolders  = SUBFOLDERS,
        val_pkl_dir = VAL_PKL_DIR,
        val_csv_dir = VAL_CSV_DIR,
        is_val      = True,
        center_size = CENTER_SIZE,
        sequence_length = 32
    )

    for name, ds in [("train", train_ds), ("val", val_ds)]:
        print(f"\n=== Processing {name} set, total sequences: {len(ds)} ===")
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        # 收集所有序列「主标签」
        all_labels = []
        for batch in loader:
            y = batch["y"]              # [B, T]
            # 对每条序列，取出现最多的 label 作为该样本的「主标签」
            modes = torch.mode(y, dim=1).values
            all_labels.extend(modes.tolist())

        # 1) 分布
        plot_label_distribution(
            all_labels,
            title=f"{name.capitalize()} Label Distribution",
            save_path=f"{name}_label_distribution.png"
        )

        # 2) 每类示例
        plot_samples_per_class(
            ds,
            num_samples=5,
            cols=5,
            save_dir=f"{name}_samples"
        )

if __name__ == "__main__":
    main()

# frame_level_vis.py

# import os
# import random
# from collections import Counter

# import matplotlib.pyplot as plt
# import torch
# from torch.utils.data import DataLoader
# from torchvision.utils import make_grid

# from dataset import ThermalBlinkDataset
# from constants import (
#     PKL_ROOT, CSV_ROOT, SUBFOLDERS,
#     VAL_PKL_DIR, VAL_CSV_DIR,
#     CENTER_SIZE, BATCH_SIZE
# )

# def plot_frame_distribution(dist, title, save_path=None):
#     labels = sorted(dist.keys())
#     counts = [dist[l] for l in labels]
#     plt.figure(figsize=(5,3))
#     bars = plt.bar(labels, counts, tick_label=labels)
#     for b in bars:
#         h = b.get_height()
#         plt.text(b.get_x()+b.get_width()/2, h+max(counts)*0.01,
#                  str(h), ha='center')
#     plt.title(title)
#     plt.xlabel("Label")
#     plt.ylabel("Frame count")
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=150)
#         print("Saved to", save_path)
#     else:
#         plt.show()
#     plt.close()

# def visualize_some_intermediate(ds, num=16, save_path="intermediate_samples.png"):
#     # 收集所有标记为1（中间态）的帧的 (seq_idx, frame_offset)
#     intermediate = []
#     for seq_idx in range(len(ds)):
#         y_seq = ds[seq_idx]["y"]  # [T]
#         for t, lab in enumerate(y_seq.tolist()):
#             if lab == 1:
#                 intermediate.append((seq_idx, t))
#     print(f"Found {len(intermediate)} intermediate frames.")
#     if not intermediate:
#         return
#     pick = random.sample(intermediate, min(num, len(intermediate)))
#     imgs = []
#     for seq_idx, t in pick:
#         x_seq = ds[seq_idx]["x"]  # [T,C,H,W]
#         frame = x_seq[t,0].unsqueeze(0)  # [1,H,W]
#         imgs.append(frame)
#     grid = make_grid(torch.stack(imgs), nrow=4, normalize=True, pad_value=1)
#     plt.figure(figsize=(8,8))
#     plt.imshow(grid.permute(1,2,0), cmap="gray")
#     plt.title("Some intermediate-state frames")
#     plt.axis("off")
#     plt.savefig(save_path, dpi=150)
#     print("Saved to", save_path)
#     plt.close()

# def frame_level_analysis(is_val=False, prefix="train"):
#     ds = ThermalBlinkDataset(
#         pkl_root    = PKL_ROOT,
#         csv_root    = CSV_ROOT,
#         subfolders  = SUBFOLDERS,
#         val_pkl_dir = VAL_PKL_DIR,
#         val_csv_dir = VAL_CSV_DIR,
#         is_val      = is_val,
#         center_size = CENTER_SIZE,
#         sequence_length = 32,
#     )
#     print(f"{prefix} dataset size (sequences):", len(ds))
#     dist = Counter()
#     for idx in range(len(ds)):
#         y_seq = ds[idx]["y"]  # tensor [T]
#         dist.update(y_seq.tolist())
#     print("Frame-level label distribution:", dist)
#     plot_frame_distribution(
#         dist,
#         title=f"{prefix.capitalize()} Frame-wise Label Dist",
#         save_path=f"{prefix}_frame_dist.png"
#     )
#     # 可视化一些中间态帧
#     visualize_some_intermediate(ds, num=16, save_path=f"{prefix}_intermediate.png")

# if __name__ == "__main__":
#     frame_level_analysis(is_val=False, prefix="train")
#     frame_level_analysis(is_val=True,  prefix="val")