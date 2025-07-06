import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import cv2
from constants import *
import torch
from torch.utils.data import Dataset

USE_SEQUENCE_LABEL = False  # ✅ 控制是否使用序列标签（False = 使用帧标签）

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
                frames, labels, timestamps = self.process_sample(pkl_path, csv_path, return_timestamps=True)
                if frames is not None:
                    N = frames.shape[0]
                    if WINDOW_MODE:
                        for i in range(0, N - FRAME_STACK_SIZE + 1, 1):
                            window = frames[i:i+FRAME_STACK_SIZE].squeeze(-1)  # [FRAME_STACK_SIZE, H, W]
                            label = float(np.any(labels[i:i+FRAME_STACK_SIZE] == 1.0)) 
                            self.data.append((window, label, None))
                    else:
                        for i in range(N):
                            frame = frames[i].squeeze(-1)  # [H, W]
                            self.data.append((frame[np.newaxis, ...], labels[i], None))  # [1, H, W]
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
                    frames, labels, _ = self.process_sample(pkl_path, csv_path, return_timestamps=False)
                    if frames is not None:
                        N = frames.shape[0]
                        if WINDOW_MODE:
                            # 正确的滑动窗口堆叠
                            for i in range(0, N - FRAME_STACK_SIZE + 1, 1):
                                window = frames[i:i+FRAME_STACK_SIZE].squeeze(-1)  # [FRAME_STACK_SIZE, H, W]
                                label = float(np.any(labels[i:i+FRAME_STACK_SIZE] == 1.0))
                                self.data.append((window, label, None))
                        else:
                            for i in range(N):
                                frame = frames[i].squeeze(-1)  # [H, W]
                                label = labels[i]
                                self.data.append((frame[np.newaxis, ...], label, None))  # [1, H, W]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y, _ = self.data[idx]
        if WINDOW_MODE:
            # x: [FRAME_STACK_SIZE, H, W]，y: float
            x = torch.tensor(x, dtype=torch.float32)  # [T, H, W]
            y = torch.tensor(y, dtype=torch.float32)  # 标量
            return {"x": x, "y": y}
        else:
            # x: [1, H, W]，y: float
            x = torch.tensor(x, dtype=torch.float32)  # [1, H, W]
            y = torch.tensor(y, dtype=torch.float32)  # 标量
            return {"x": x, "y": y}

    def assign_frame_labels(self, timestamps, blink_starts, blink_ends):
        """
        按帧打标签，start ~ end 区间为 1，其他为 0
        """
        labels = np.zeros_like(timestamps, dtype=np.float32)
        for start, end in zip(blink_starts, blink_ends):
            labels[(timestamps >= start) & (timestamps <= end)] = 1.0
        return labels


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

        df = pd.read_csv(csv_path)
        offsets = {row['key']: json.loads(row['value']) for _, row in df.iterrows()}
        blink_start_offsets = offsets["start_offsets"]
        blink_end_offsets = offsets["end_offsets"]
        # ✅ 如果起始数量不一致，且 end 的第一帧在 start 之前，则补 0
        if len(blink_start_offsets) != len(blink_end_offsets):
            print(f"[WARN] 修正中: {os.path.basename(pkl_path)}")
        
            # 👉 如果 end 比 start 多，说明缺少起始，补 0
            if len(blink_end_offsets) > len(blink_start_offsets):
                blink_start_offsets = [0] + blink_start_offsets
                print(f"➕ 插入 start=0")
        
            # 👉 如果 start 比 end 多，说明缺少结束，补最后一帧时间
            elif len(blink_start_offsets) > len(blink_end_offsets):
                last_offset = (datetime.fromisoformat(data['timestamp'][-1]) - datetime.fromisoformat(
                    data['timestamp'][0])).total_seconds() * 1000
                blink_end_offsets.append(int(last_offset))
                print(f"➕ 补充 end={int(last_offset)}")
        
        # 再次检查是否对齐
        if len(blink_start_offsets) != len(blink_end_offsets):
            print(f"[ERROR] 修正后仍不一致: {pkl_path}")
            print(len(blink_start_offsets), len(blink_end_offsets))
            return None, None, None


        # temperature_frames = np.array(data['temperature'])  # [N, 12, 16]
        raw_frames = np.array(data['temperature'])  # [N, H, W]

        # ✅ Step A: 统一预处理并 clip 到固定范围 [-3, 3]
        enhanced_all = []
        for frame in raw_frames:
            # ① 高斯滤波
            blurred = cv2.GaussianBlur(frame, (3, 3), sigmaX=0.5)
            # ② 标准差归一化
            enhanced = (blurred - np.mean(blurred)) / (np.std(blurred) + 1e-5)
            enhanced_all.append(enhanced)

        # ✅ 固定 clip 范围
        global_min = -3.0
        global_max = 2.0
        print(f"🌡️ 使用统一增强范围: min={global_min}, max={global_max}")
        processed = []
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
        for e in enhanced_all:
            c   = np.clip(e, global_min, global_max)
            n01 = (c - global_min) / (global_max - global_min)
            g   = np.power(n01, 0.5) # gamma 0.5
            u8  = (g * 255).astype(np.uint8)
            ce  = clahe.apply(u8).astype(np.float32) / 255.0
            processed.append(ce)
        frames_np = np.stack(processed, axis=0)  # [N, H, W]


        # 5. 时间戳 & 帧级标签
        ts_list = [datetime.fromisoformat(t) for t in data['timestamp']]
        base = ts_list[0]
        timestamps = np.array([(t - base).total_seconds() * 1000 for t in ts_list], dtype=np.float32)
        frame_labels = self.assign_frame_labels(timestamps, blink_start_offsets, blink_end_offsets)
        
        # 中心裁剪
        h, w = frames_np[0].shape
        ch, cw = self.center_size
        sr = h // 2 - ch // 2
        er = sr + ch
        sc = w // 2 - cw // 2
        ec = sc + cw
        cropped_frames = frames_np[:, sr:er, sc:ec]  # [N, H', W']
        X = cropped_frames[..., np.newaxis]  # [N, H', W', 1]


        if return_timestamps:
            return X, frame_labels, timestamps
        else:
            return X, frame_labels, None

