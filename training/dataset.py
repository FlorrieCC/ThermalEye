import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch.utils.data import Dataset
# from utils import discretize_blink_labels
def discretize_blink_labels(cont_labels, low_thr=0.1, high_thr=0.9):
    """
    cont_labels: numpy array or torch.Tensor of shape [N], 连续[0,1]标签
    返回 int 类标: 0=睁眼 (<low_thr), 1=闭眼 (>high_thr), 2=中间态 (low_thr ≤ x ≤ high_thr)
    0=open, 1=closed, 2=intermediate
    """

    # 如果是 tensor，先转 numpy
    x = cont_labels.cpu().numpy() if hasattr(cont_labels, "cpu") else cont_labels
    # intermediate_mask = (x >= low_thr) & (x <= high_thr)
    # if np.any(intermediate_mask):
    #     print("Found intermediate confidence values at indices:",
    #           np.nonzero(intermediate_mask)[0])
    #     print("Their raw confidences are:", x[intermediate_mask])
    # idx = np.zeros_like(x, dtype=np.int64)
    idx = np.full_like(x, 2, dtype=np.int64)

    idx[x < low_thr] = 0      # 0 类: open eye
    idx[x > high_thr] = 1     # 1 类: closed eye
    # idx[(x >= low_thr) & (x <= high_thr)] = 2  # 2 类: intermediate
    
    return idx  # shape [N], dtype=int64
class ThermalBlinkDataset(Dataset):
    def __init__(
        self,
        pkl_root: str,
        csv_root: str,
        subfolders: list,
        val_pkl_dir: str = None,
        val_csv_dir: str = None,
        is_val: bool = False,
        center_size: tuple = (12, 16),
        sequence_length: int = 32
    ):
        """
        Args:
            pkl_root: 热图数据根目录
            csv_root: blink标注csv文件目录
            subfolders: 子文件夹列表，如 ["0503", "0505"]
            val_pkl_dir: 验证集单独pkl路径(is_val=True时启用)
            val_csv_dir: 验证集对应csv路径(is_val=True时启用)
            is_val: 是否是验证集
            center_size: 中心裁剪区域大小，默认(12,16)
        """
        self.center_size = center_size
        self.data = []
        self.sequence_length = sequence_length

        if is_val:
            # 只加载一对 val pkl + csv
            for filename in os.listdir(val_pkl_dir):
                if not filename.endswith(".pkl"):
                    continue
                pkl_path = os.path.join(val_pkl_dir, filename)
                match = re.match(r"(.*_\d{8}_\d{4})", filename)
                if not match:
                    continue
                fuzzy_key = match.group(1)
                matched_csvs = [
                    f for f in os.listdir(val_csv_dir)
                    if f.startswith("blink_offsets_") and fuzzy_key in f and f.endswith(".csv")
                ]
                if not matched_csvs:
                    continue
                csv_path = os.path.join(val_csv_dir, matched_csvs[0])
                X, y, timestamps = self.process_sample(pkl_path, csv_path, return_timestamps=True)
                for i in range(len(X)):
                    self.data.append((X[i], y[i], timestamps[i]))

        else:
            # 批量加载训练集
            for subfolder in subfolders:
                pkl_dir = os.path.join(pkl_root, subfolder)
                for filename in os.listdir(pkl_dir):
                    if not filename.endswith(".pkl"):
                        continue
                    pkl_path = os.path.join(pkl_dir, filename)
                    if val_pkl_dir and pkl_path == val_pkl_dir:
                        continue  # 排除验证集文件
                    match = re.match(r"(.*_\d{8}_\d{4})", filename)
                    if not match:
                        continue
                    fuzzy_key = match.group(1)

                    # 去 gt_output 的子目录中匹配对应 CSV
                    csv_subdir = os.path.join(csv_root, subfolder)
                    if not os.path.exists(csv_subdir):
                        print(f"[WARN] CSV 子目录不存在: {csv_subdir}")
                        continue
                    matched_csvs = [
                        f for f in os.listdir(csv_subdir)
                        if f.startswith("blink_offsets_") and fuzzy_key in f and f.endswith(".csv")
                    ]
                    if not matched_csvs:
                        print(f"[WARN] 未匹配到 CSV: {fuzzy_key} in {csv_subdir}")
                        continue
                    csv_path = os.path.join(csv_subdir, matched_csvs[0])

                    X, y, _ = self.process_sample(pkl_path, csv_path)
                    if X is not None:
                        for i in range(len(X)):
                            self.data.append((X[i], y[i]))


    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx):
        seq_len = self.sequence_length

        if idx + seq_len > len(self.data):
            idx = len(self.data) - seq_len

        frames, labels, timestamps = [], [], []

        for i in range(seq_len):
            item = self.data[idx + i]
            x, y = item[:2]
            frames.append(torch.tensor(x).float().permute(2, 0, 1))  # [C, H, W]
            labels.append(torch.tensor(y).long())
            if len(item) == 3:
                timestamps.append(item[2])  # 只在验证集有

        x_seq = torch.stack(frames, dim=0)   # [T, C, H, W]
        y_seq = torch.stack(labels, dim=0)   # [T]

        if timestamps:
            return {"x": x_seq, "y": y_seq, "timestamp": torch.tensor(timestamps)}
        else:
            return {"x": x_seq, "y": y_seq}



    def process_sample(self, pkl_path, csv_path, return_timestamps=False):
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            if not ('temperature' in data and 'timestamp' in data):
                print(f"[ERROR] 缺字段: {pkl_path}")
                return None, None, None
        except Exception as e:
            print(f"[ERROR] pkl 读取失败: {pkl_path}\n{e}")
            return None, None, None

        try:
            df = pd.read_csv(csv_path)
            offsets = {row['key']: json.loads(row['value']) for _, row in df.iterrows()}
            blink_start_offsets = offsets["start_offsets"]
            blink_end_offsets = offsets["end_offsets"]
        except Exception as e:
            print(f"[ERROR] CSV 读取失败: {csv_path}\n{e}")
            return None, None, None

        temperature_frames = np.array(data['temperature'])  # [N, 12, 16]
        raw_timestamps = data['timestamp']
        parsed_times = [datetime.fromisoformat(ts) for ts in raw_timestamps]
        start_time = parsed_times[0]
        timestamps = np.array([(t - start_time).total_seconds() * 1000 for t in parsed_times])  # ms
        
        # Step 1: 裁剪 CSV 的闭眼区间
        filtered_starts = []
        filtered_ends = []
        for start, end in zip(blink_start_offsets, blink_end_offsets):
            if start >= 1000 and end >= 1000:
                filtered_starts.append(start)
                filtered_ends.append(end)
        blink_start_offsets = filtered_starts
        blink_end_offsets = filtered_ends

        # Step 2: 裁剪帧数据中前1000ms
        valid_indices = np.where(timestamps >= 1000)[0]
        timestamps = timestamps[valid_indices]
        temperature_frames = temperature_frames[valid_indices]

        # 帧级标签
        # labels = np.zeros_like(timestamps, dtype=np.float32)

        # for start, end in zip(blink_start_offsets, blink_end_offsets):
        #     duration = end - start
        #     fade = min(0.5 * duration, 300)  # 自适应渐变区，最多300ms

        #     for i, t in enumerate(timestamps):
        #         if start - fade <= t < start:
        #             labels[i] = max(labels[i], 1 - (start - t) / fade)
        #         elif start <= t <= end:
        #             labels[i] = max(labels[i], 1.0)
        #         elif end < t <= end + fade:
        #             labels[i] = max(labels[i], 1 - (t - end) / fade)
        cont_labels = np.zeros_like(timestamps, dtype=np.float32)
        for start, end in zip(blink_start_offsets, blink_end_offsets):
            dur = end - start
            fade = min(0.5 * dur, 300)
            for i, t in enumerate(timestamps):
                if start - fade <= t < start:
                    # print(cont_labels[i])
                    # 半眨眼过程
                    cont_labels[i] = max(cont_labels[i], 1 - (start - t) / fade)
                elif start <= t <= end:
                    # print(cont_labels[i])
                    # 闭眼
                    cont_labels[i] = 1.0
                elif end < t <= end + fade:
                    # 睁眼
                    cont_labels[i] = max(cont_labels[i], 1 - (t - end) / fade)
        labels_cls = discretize_blink_labels(cont_labels,
                                        low_thr=0.3,
                                        high_thr=0.7)
        # print(labels_cls)
        # 中心裁剪
        h, w = temperature_frames[0].shape
        ch, cw = self.center_size
        sr = h // 2 - ch // 2
        er = sr + ch
        sc = w // 2 - cw // 2
        ec = sc + cw
        cropped_frames = temperature_frames[:, sr:er, sc:ec]  # [N, H', W']
        X = cropped_frames[..., np.newaxis]  # [N, H', W', 1]

        if return_timestamps:
            return X, labels_cls, timestamps
        else:
            return X, labels_cls, None
