import os
import pickle
from datetime import datetime

def inspect_video_timestamps(mp4_path, pkl_dir):
    # 解析文件名
    video_dir = os.path.dirname(mp4_path)
    video_name = os.path.basename(mp4_path)
    base_name = os.path.splitext(video_name)[0]

    # === 1. 读取 .txt ===
    txt_path = os.path.join(video_dir, f"ts_{base_name}.txt")
    if not os.path.exists(txt_path):
        print(f"[ERROR] 找不到对应的 .txt 文件: {txt_path}")
    else:
        with open(txt_path, 'r') as f:
            txt_lines = [line.strip() for line in f if line.strip()]

        print("\n📝 .txt 文件时间戳:")
        for i, line in enumerate(txt_lines[:3]):
            print(f"  [{i}] {line}")

        print("\n📝 .txt 文件时间戳:")
        for i, line in enumerate(txt_lines[-3:]):
            print(f"  [-{5 - i}] {line}")

    # === 2. 匹配并读取 .pkl ===
    parts = base_name.split('_')
    match_token = parts[-3] + "_" + parts[-2][:4]
    matched_pkl = None
    for f in os.listdir(pkl_dir):
        if f.endswith(".pkl") and match_token in f:
            matched_pkl = os.path.join(pkl_dir, f)
            break

    if not matched_pkl:
        print("\n[ERROR] 找不到对应的 .pkl 文件")
    else:
        with open(matched_pkl, 'rb') as f:
            data = pickle.load(f)
        ts_data = data['timestamp']

        print(f"\n📦 .pkl 文件路径: {matched_pkl}")
        print("\n📦 .pkl 中 timestamp:")
        for i, ts in enumerate(ts_data[:3]):
            print(f"  [{i}] {ts}")

        print("\n📦 .pkl 中 timestamp:")
        for i, ts in enumerate(ts_data[-3:]):
            print(f"  [-{5 - i}] {ts}")

if __name__ == "__main__":
    mp4_path = "/Users/yvonne/Documents/final project/ThermalEye/real_data/0611down/xx_normal_20250611_175154_461.mp4"
    pkl_dir = "/Users/yvonne/Documents/final project/ThermalEye/ira_data/0611down"
    inspect_video_timestamps(mp4_path,pkl_dir)
