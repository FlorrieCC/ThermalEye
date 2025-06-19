import json
import os
import pickle
import re
from datetime import datetime

import numpy as np
import pandas as pd
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
            sequence_length: int = 32,
            normalize: bool = False,
            std_enhance: bool = False,
    ):
        self.center_size = center_size
        self.sequence_length = sequence_length
        self.data = []
        self.normalize = normalize
        self.std_enhance = std_enhance

        if is_val:
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
            for subfolder in subfolders:
                pkl_dir = os.path.join(pkl_root, subfolder)
                for filename in os.listdir(pkl_dir):
                    if not filename.endswith(".pkl"):
                        continue
                    pkl_path = os.path.join(pkl_dir, filename)
                    match = re.match(r"(.*_\d{8}_\d{4})", filename)
                    if not match:
                        continue
                    fuzzy_key = match.group(1)
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
                timestamps.append(item[2])
        x_seq = torch.stack(frames, dim=0)  # [T, C, H, W]
        y_seq = torch.stack(labels, dim=0)  # [T]
        if timestamps:
            return {"x": x_seq, "y": y_seq, "timestamp": torch.tensor(timestamps)}
        else:
            return {"x": x_seq, "y": y_seq}

    def process_sample(self, pkl_path, csv_path, return_timestamps=False):
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            if 'temperature' not in data or 'timestamp' not in data:
                print(f"[ERROR] 缺字段: {pkl_path}")
                return None, None, None
        except Exception as e:
            print(f"[ERROR] 读取失败: {pkl_path}\n{e}")
            return None, None, None

        try:
            df = pd.read_csv(csv_path)
            offsets = {row['key']: json.loads(row['value']) for _, row in df.iterrows()}
            blink_start_offsets = offsets["start_offsets"]
            blink_end_offsets = offsets["end_offsets"]
        except Exception as e:
            print(f"[ERROR] CSV读取失败: {csv_path}\n{e}")
            return None, None, None

        temperature_frames = np.array(data['temperature'])  # [N, 12, 16]
        raw_timestamps = data['timestamp']
        parsed_times = [datetime.fromisoformat(ts) for ts in raw_timestamps]
        start_time = parsed_times[0]
        timestamps = np.array([(t - start_time).total_seconds() * 1000 for t in parsed_times])

        # Global normalization (per pkl clip)
        if self.normalize:
            global_min = np.min(temperature_frames)
            global_max = np.max(temperature_frames)
            temperature_frames = (temperature_frames - global_min) / (global_max - global_min + 1e-6)

        if self.std_enhance:
            global_std = np.std(temperature_frames)
            temperature_frames = temperature_frames / (global_std + 1e-6)
            print(
                f"[DEBUG] After norm: min={temperature_frames.min():.4f}, max={temperature_frames.max():.4f}, std={temperature_frames.std():.4f}")

        # 裁剪闭眼段（>=1000ms）
        blink_start_offsets, blink_end_offsets = zip(*[
            (s, e) for s, e in zip(blink_start_offsets, blink_end_offsets) if s >= 1000 and e >= 1000
        ]) if blink_start_offsets else ([], [])

        # 裁剪前1000ms帧
        valid_indices = np.where(timestamps >= 1000)[0]
        timestamps = timestamps[valid_indices]
        temperature_frames = temperature_frames[valid_indices]

        # Soft标签构建
        labels = np.zeros_like(timestamps, dtype=np.float32)
        for start, end in zip(blink_start_offsets, blink_end_offsets):
            duration = end - start
            fade = min(0.5 * duration, 300)
            for i, t in enumerate(timestamps):
                if start - fade <= t < start:
                    labels[i] = max(labels[i], 1 - (start - t) / fade)
                elif start <= t <= end:
                    labels[i] = max(labels[i], 1.0)
                elif end < t <= end + fade:
                    labels[i] = max(labels[i], 1 - (t - end) / fade)

        # 中心裁剪
        h, w = temperature_frames[0].shape
        ch, cw = self.center_size
        sr, sc = h // 2 - ch // 2, w // 2 - cw // 2
        cropped_frames = temperature_frames[:, sr:sr + ch, sc:sc + cw]
        X = cropped_frames[..., np.newaxis]  # [N, H, W, 1]

        if return_timestamps:
            return X, labels, timestamps
        else:
            return X, labels, None

    @staticmethod
    def visualize_in_dataset(dataset):
        import matplotlib.pyplot as plt
        import os
        import pandas as pd
        import numpy as np

        print("[INFO] 正在生成可视化图像...")
        dataset_per_pkl = []
        idx = 0

        while idx < len(dataset.data):
            current_X, current_y = [], []
            while idx < len(dataset.data):
                item = dataset.data[idx]
                if len(item) == 3:
                    x, y, _ = item
                else:
                    x, y = item
                current_X.append(x)
                current_y.append(y)
                idx += 1
            dataset_per_pkl.append((np.stack(current_X), np.array(current_y)))

        os.makedirs("visual_output/compare_fixed", exist_ok=True)
        os.makedirs("visual_output/stds", exist_ok=True)

        std_records = []
        per_pkl_stds = []
        all_frames = []

        for file_idx, (X, labels) in enumerate(dataset_per_pkl):
            stds = np.std(X, axis=(1, 2, 3))
            per_pkl_stds.append(stds)

            std_records.append({
                "video_id": file_idx,
                "mean_std": np.mean(stds),
                "max_std": np.max(stds),
                "min_std": np.min(stds),
                "frame_count": len(stds),
            })

            for i in range(X.shape[0]):
                all_frames.append(X[i, :, :, 0])
                if len(all_frames) >= 12:
                    break
            if len(all_frames) >= 12:
                break

        # 每行显示3帧图像，总共保存4张图，每张图显示3帧
        for img_idx in range(4):
            fig, axes = plt.subplots(1, 3, figsize=(6, 2))
            for i in range(3):
                frame_idx = img_idx * 3 + i
                if frame_idx >= len(all_frames):
                    axes[i].axis("off")
                else:
                    axes[i].imshow(all_frames[frame_idx], cmap='hot')
                    axes[i].set_title(f"Frame {frame_idx}", fontsize=8)
                    axes[i].axis("off")
            plt.tight_layout()
            plt.savefig(f"visual_output/compare_fixed/grid_{img_idx:02d}.png", dpi=300)
            plt.close()

        # 绘制STD分布图
        plt.figure(figsize=(12, 6))
        for i, stds in enumerate(per_pkl_stds):
            plt.plot(stds, label=f"Video {i}")
        plt.xlabel("Frame Index")
        plt.ylabel("STD")
        plt.title("Per-frame STD Distributions")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig("visual_output/stds/std_distributions.png", dpi=300)
        plt.close()

        # 打印和保存STD汇总表格
        summary_df = pd.DataFrame(std_records)
        print("\n[INFO] STD统计信息：")
        print(summary_df)
        summary_df.to_csv("visual_output/stds/std_summary.csv", index=False)
