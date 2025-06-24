import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import cv2

import torch
from torch.utils.data import Dataset


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
                    base_name = os.path.splitext(filename)[0]  # 去掉 .pkl 后缀
                    parts = base_name.split('_')
                    fuzzy_key = parts[-3] + "_" + parts[-2][:4]


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
            labels.append(torch.tensor(y).float())
            if len(item) == 3:
                timestamps.append(item[2])  # 只在验证集有

        x_seq = torch.stack(frames, dim=0)   # [T, C, H, W]
        y_seq = torch.stack(labels, dim=0)   # [T]

        if timestamps:
            return {"x": x_seq, "y": y_seq, "timestamp": torch.tensor(timestamps)}
        else:
            return {"x": x_seq, "y": y_seq}
        
        
    def assign_frame_labels(timestamps, blink_start_offsets, blink_end_offsets):
        """
        标签方式1：按帧打标签，start ~ end 区间为 1，其他为 0
        """
        labels = np.zeros_like(timestamps, dtype=np.float32)
        for start, end in zip(blink_start_offsets, blink_end_offsets):
            labels[(timestamps >= start) & (timestamps <= end)] = 1.0
        return labels

    def assign_sequence_labels(frame_labels, sequence_length=32, stride=16):
        """
        标签方式2：基于帧标签生成序列标签。
        若序列中既有0又有1，则为1；否则为0。
        """
        sequence_labels = []
        for i in range(0, len(frame_labels) - sequence_length + 1, stride):
            segment = frame_labels[i:i + sequence_length]
            label = 1.0 if np.any(segment != segment[0]) else 0.0
            sequence_labels.append(label)
        return np.array(sequence_labels, dtype=np.float32)
    


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

        # temperature_frames = np.array(data['temperature'])  # [N, 12, 16]
        raw_frames = np.array(data['temperature'])  # [N, H, W]
        # ✅ Step A: 计算全序列 enhanced 的全局 min/max
        enhanced_all = []
        for frame in raw_frames:
            blurred = cv2.GaussianBlur(frame, (3, 3), sigmaX=0.5)
            enhanced = (blurred - np.mean(blurred)) / (np.std(blurred) + 1e-5)
            enhanced_all.append(enhanced)

        enhanced_stack = np.stack(enhanced_all, axis=0)  # [N, H, W]
        global_min = enhanced_stack.min()
        global_max = enhanced_stack.max()
        print(f"🌡️ 全序列归一化后温度范围: min={global_min:.3f}, max={global_max:.3f}")

        # ✅ Step B: 用 global min/max 进行 clip 和 gamma 拉伸
        processed_frames = []
        for enhanced in enhanced_all:
            clipped = np.clip(enhanced, global_min, global_max)
            norm_0_1 = (clipped - global_min) / (global_max - global_min)
            adjusted = np.power(norm_0_1, 0.5)
            norm = (adjusted * 255).astype(np.uint8)
            processed_frames.append(norm)

        temperature_frames = np.stack(processed_frames, axis=0)  # [N, H, W]

        
        raw_timestamps = data['timestamp']
        parsed_times = [datetime.fromisoformat(ts) for ts in raw_timestamps]
        start_time = parsed_times[0]
        timestamps = np.array([(t - start_time).total_seconds() * 1000 for t in parsed_times])  # ms
        
        # ✅ 合并相邻眨眼段（间隔小于等于1000ms）
        merged_starts, merged_ends = [], []
        if blink_start_offsets:
            cur_start = blink_start_offsets[0]
            cur_end = blink_end_offsets[0]
            for i in range(1, len(blink_start_offsets)):
                next_start = blink_start_offsets[i]
                next_end = blink_end_offsets[i]
                if next_start - cur_end <= 1000:
                    cur_end = max(cur_end, next_end)
                else:
                    merged_starts.append(cur_start)
                    merged_ends.append(cur_end)
                    cur_start = next_start
                    cur_end = next_end
            merged_starts.append(cur_start)
            merged_ends.append(cur_end)
            blink_start_offsets = merged_starts
            blink_end_offsets = merged_ends

        
        # 帧级标签
        labels = self.assign_frame_labels(timestamps, blink_start_offsets, blink_end_offsets)
        sequence_labels = self.assign_sequence_labels(labels, sequence_length=self.sequence_length, stride=self.sequence_length // 2)

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
            return X, sequence_labels, timestamps
        else:
            return X, sequence_labels, None
