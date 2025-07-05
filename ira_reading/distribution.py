import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import cv2

def sigmoid_stretch(X, slope=10, center=0.5):
    return 1 / (1 + np.exp(-slope * (X - center)))

def gaussian_filter(frame, ksize=(3, 3), sigma=0):
    return cv2.GaussianBlur(frame, ksize, sigma)


def zscore_standardize(all_frames):
    flat = np.concatenate([f.flatten() for f in all_frames])
    mean = flat.mean()
    std = flat.std()
    return [(f - mean) / std for f in all_frames]


def sliding_window_normalize(frames, window_size=160, step_size=160):
    """
    对每个滑动窗口内做 Min-Max 归一化。
    不足的部分用最后一个窗口的结果填充。
    """
    norm_frames = []
    N = len(frames)
    for start in range(0, N, step_size):
        end = min(start + window_size, N)
        window = frames[start:end]
        flat = np.concatenate([f.flatten() for f in window])
        w_min = flat.min()
        w_max = flat.max()
        window_norm = [(f - w_min) / (w_max - w_min + 1e-8) for f in window]
        norm_frames.extend(window_norm)
    return norm_frames


def process_single(pkl_path, csv_dir, output_dir):
    base_name = os.path.basename(pkl_path)
    parts = base_name.split('_')
    match_token = parts[-3] + "_" + parts[-2][:4]

    # 找到匹配的 CSV
    csv_filename = None
    for file in os.listdir(csv_dir):
        if file.endswith(".csv") and match_token in file:
            csv_filename = file
            break

    if csv_filename is None:
        print(f"[ERROR] CSV not found for {pkl_path}")
        return

    csv_path = os.path.join(csv_dir, csv_filename)
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    frames = data["temperature"]
    timestamps = data["timestamp"]
    base_time = datetime.fromisoformat(timestamps[0])

    df = pd.read_csv(csv_path)
    start_offsets = list(map(int, json.loads(df[df["key"] == "start_offsets"]["value"].values[0])))
    end_offsets = list(map(int, json.loads(df[df["key"] == "end_offsets"]["value"].values[0])))

    ts_offsets = []
    for ts in timestamps:
        current_time = datetime.fromisoformat(ts)
        offset = int((current_time - base_time).total_seconds() * 1000)
        ts_offsets.append(offset)

    blink_idxs = []
    for start, end in zip(start_offsets, end_offsets):
        seg_idxs = [i for i, ts in enumerate(ts_offsets) if start <= ts <= end]
        blink_idxs.extend(seg_idxs)

    all_idxs = set(range(len(frames)))
    eye_open_idxs = list(all_idxs - set(blink_idxs))

    print(f"[INFO] Blink frames: {len(blink_idxs)}, Eye-open frames: {len(eye_open_idxs)}")
    
    # frames_center = []
    # for f in frames:
    #     center = f[1:10, 2:14]
    #     frames_center.append(center)
        
    # frames = frames_center

    # ✅ Step 1: 高斯滤波
    frames_smoothed = [gaussian_filter(f) for f in frames]

    # ✅ Step 2: 全局 Z-score 标准化
    frames_standardized = zscore_standardize(frames_smoothed)

    # ✅ Step 3: 滑动窗口局部归一化
    frames_normalized = sliding_window_normalize(frames_standardized, window_size=160, step_size=160)
    
    # 新增：对每帧做 sigmoid stretch
    frames_stretched = [sigmoid_stretch(f) for f in frames_normalized]

    # ✅ Step 4: flatten
    blink_pixels = np.concatenate([frames_stretched[i].flatten() for i in blink_idxs]) if blink_idxs else np.array([])
    eye_open_pixels = np.concatenate([frames_stretched[i].flatten() for i in eye_open_idxs]) if eye_open_idxs else np.array([])

    # ✅ Step 5: Plot and save
    plt.figure(figsize=(12, 5))
    plt.hist(eye_open_pixels, bins=50, alpha=0.5, label="Eye-Open")
    plt.hist(blink_pixels, bins=50, alpha=0.5, label="Blink")
    plt.xlabel("Normalized Temperature")
    plt.ylabel("Frequency")
    plt.title(f"Distribution (Filtered + Z-score + Local Norm): {base_name}")
    plt.legend()
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{base_name.replace('.pkl', '')}_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[SAVED] {save_path}")


def main(mode="single",
         pkl_path=None,
         pkl_dir=None,
         csv_dir=None,
         output_dir=None):

    assert mode in ["single", "batch"], "Mode must be 'single' or 'batch'."

    if mode == "single":
        assert pkl_path is not None and csv_dir is not None and output_dir is not None
        process_single(pkl_path, csv_dir, output_dir)

    elif mode == "batch":
        assert pkl_dir is not None and csv_dir is not None and output_dir is not None
        all_pkl = [f for f in os.listdir(pkl_dir) if f.endswith(".pkl")]
        print(f"[INFO] Found {len(all_pkl)} PKL files in {pkl_dir}")

        for file in all_pkl:
            pkl_path = os.path.join(pkl_dir, file)
            process_single(pkl_path, csv_dir, output_dir)

    print("[DONE]")


if __name__ == "__main__":
    main(
        mode="batch",  # "single" or "batch"
        pkl_path="/Users/yvonne/Documents/final project/ThermalEye/ira_data/0702/xx_indoor_severe_20250702_194844_203.pkl",
        pkl_dir="/Users/yvonne/Documents/final project/ThermalEye/ira_data/0703_outdoor/",
        csv_dir="/Users/yvonne/Documents/final project/ThermalEye/gt_output/0703_outdoor/",
        output_dir="/Users/yvonne/Documents/final project/ThermalEye/distribution_2/0703_outdoor/"
    )
