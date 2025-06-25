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
                stride = self.sequence_length // 2
                if USE_SEQUENCE_LABEL:
                    for i in range(0, len(X) - self.sequence_length + 1, stride):
                        x_seq = X[i:i + self.sequence_length]
                        y_seq = y[i // stride]
                        self.data.append((x_seq, y_seq))
                else:
                    for i in range(len(X)):
                        self.data.append((X[i], y[i]))


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

                    X, y, timestamps = self.process_sample(pkl_path, csv_path, return_timestamps=True)

                    if X is not None:
                        stride = self.sequence_length // 2
                        if USE_SEQUENCE_LABEL:
                            for i in range(0, len(X) - self.sequence_length + 1, stride):
                                x_seq = X[i:i + self.sequence_length]
                                y_seq = y[i // stride]
                                ts_seq = timestamps[i:i + self.sequence_length]
                                self.data.append((x_seq, y_seq, ts_seq))
                        else:
                            for i in range(len(X)):
                                self.data.append((X[i], y[i], timestamps[i]))



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
                
                
        if USE_SEQUENCE_LABEL:
            x_seq = torch.stack(frames, dim=0)   # [T, C, H, W]
            y_seq = torch.tensor(labels[0]).float()  # 单个标签
        else:

            x_seq = torch.stack(frames, dim=0)   # [T, C, H, W]
            y_seq = torch.stack(labels, dim=0)   # [T]

        if timestamps:
            return {"x": x_seq, "y": y_seq, "timestamp": torch.tensor(timestamps)}
        else:
            return {"x": x_seq, "y": y_seq}
        
        
    def assign_frame_labels(self, timestamps, blink_start_offsets, blink_end_offsets):
        """
        标签方式1：按帧打标签，start ~ end 区间为 1，其他为 0
        """
        labels = np.zeros_like(timestamps, dtype=np.float32)
        for start, end in zip(blink_start_offsets, blink_end_offsets):
            labels[(timestamps >= start) & (timestamps <= end)] = 1.0
        return labels

    def assign_sequence_labels(self, frame_labels, sequence_length=32, stride=16):
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
            # ✅ 如果起始数量不一致，且 end 的第一帧在 start 之前，则补 0
            if len(blink_start_offsets) != len(blink_end_offsets):
                print(f"[WARN] 修正中: {os.path.basename(pkl_path)}")
                
                # 👉 如果 end 比 start 多，说明缺少起始，补 0
                if len(blink_end_offsets) > len(blink_start_offsets):
                    blink_start_offsets = [0] + blink_start_offsets
                    print(f"➕ 插入 start=0")

                # 👉 如果 start 比 end 多，说明缺少结束，补最后一帧时间
                elif len(blink_start_offsets) > len(blink_end_offsets):
                    last_offset = (datetime.fromisoformat(data['timestamp'][-1]) - datetime.fromisoformat(data['timestamp'][0])).total_seconds() * 1000
                    blink_end_offsets.append(int(last_offset))
                    print(f"➕ 补充 end={int(last_offset)}")

                # 再次检查是否对齐
                if len(blink_start_offsets) != len(blink_end_offsets):
                    print(f"[ERROR] 修正后仍不一致: {pkl_path}")
                    print(len(blink_start_offsets), len(blink_end_offsets))
                    return None, None, None



        except Exception as e:
            print(f"[ERROR] CSV 读取失败: {csv_path}\n{e}")
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

        # ✅ Step B: clip + gamma + CLAHE
        processed_frames = []
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))  # ④ CLAHE 初始化

        for enhanced in enhanced_all:
            # ✅ Clip 到统一范围
            clipped = np.clip(enhanced, global_min, global_max)

            # ✅ 归一化并 gamma 拉伸
            norm_0_1 = (clipped - global_min) / (global_max - global_min)
            adjusted = np.power(norm_0_1, 0.5)

            # ✅ 映射到 [0, 255] 并 CLAHE
            norm = (adjusted * 255).astype(np.uint8)
            contrast_enhanced = clahe.apply(norm)

            processed_frames.append(contrast_enhanced)


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
        if USE_SEQUENCE_LABEL:
            labels = self.assign_sequence_labels(
                self.assign_frame_labels(timestamps, blink_start_offsets, blink_end_offsets),
                sequence_length=self.sequence_length,
                stride=self.sequence_length // 2
            )
        else:
            labels = self.assign_frame_labels(timestamps, blink_start_offsets, blink_end_offsets)


        # 中心裁剪
        h, w = temperature_frames[0].shape
        ch, cw = self.center_size
        sr = h // 2 - ch // 2
        er = sr + ch
        sc = w // 2 - cw // 2
        ec = sc + cw
        cropped_frames = temperature_frames[:, sr:er, sc:ec]  # [N, H', W']
        X = cropped_frames[..., np.newaxis]  # [N, H', W', 1]
        
        # X = temperature_frames[..., np.newaxis]  # [N, H, W, 1]


        if return_timestamps:
            return X, labels, timestamps
        else:
            return X, labels, None





# import matplotlib.pyplot as plt

# def create_composite(images, cols=50, resize_shape=(160, 120)):
#     if len(images) == 0:
#         return None
#     rows = (len(images) + cols - 1) // cols
#     width, height = resize_shape
#     canvas = np.zeros((rows * height, cols * width), dtype=np.uint8)
#     for idx, img in enumerate(images):
#         r, c = divmod(idx, cols)
#         y0, y1 = r * height, (r + 1) * height
#         x0, x1 = c * width, (c + 1) * width
#         canvas[y0:y1, x0:x1] = cv2.resize(img, resize_shape, interpolation=cv2.INTER_NEAREST)
#     return canvas


# def visualize_processed_frames(frames, group_size=160, cols=50, resize_shape=(160, 120)):
#     gap_height = 20
#     rows = []
#     for i in range(0, len(frames), group_size):
#         group = frames[i:i + group_size]
#         group_canvas = create_composite(group, cols=cols, resize_shape=resize_shape)
#         rows.append(group_canvas)
#         gap = np.zeros((gap_height, group_canvas.shape[1]), dtype=np.uint8)
#         rows.append(gap)
#     final_canvas = cv2.vconcat(rows[:-1]) if len(rows) > 1 else rows[0]
#     plt.figure(figsize=(20, 12))
#     plt.imshow(final_canvas, cmap='jet')
#     plt.title("Thermal Frames (每160帧一组，组间隔空行)")
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()


# def main():
#     # ✅ 修改路径为你实际的验证集样本路径
#     val_pkl_path = "/Users/yvonne/Documents/final project/ThermalEye/ira_data/0505/callibration_20250505_161542_107.pkl"
#     val_csv_path = "/Users/yvonne/Documents/final project/ThermalEye/gt_output/0505/blink_offsets_callibration_20250505_161542_482.csv"

#     # ✅ 只加载这一对样本用于可视化
#     dataset = ThermalBlinkDataset(
#         pkl_root=None, csv_root=None, subfolders=[],
#         val_pkl_dir=os.path.dirname(val_pkl_path),
#         val_csv_dir=os.path.dirname(val_csv_path),
#         is_val=True
#     )

#     # ✅ 提取原始增强后的帧（从 process_sample 再跑一次）
#     frames, _, _ = dataset.process_sample(val_pkl_path, val_csv_path, return_timestamps=True)

#     # ✅ frames.shape = [N, H, W, 1]，先 squeeze 并变 list
#     images = [f.squeeze() for f in frames]  # -> List of [H, W]

#     # ✅ 展示
#     visualize_processed_frames(images)

# if __name__ == "__main__":
#     main()
