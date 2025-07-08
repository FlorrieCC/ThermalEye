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
from utils import set_random_seeds

class ThermalBlinkDataset(Dataset):
    def __init__(
            self,
            pkl_root: str,
            csv_root: str,
            subfolders: list,
            split: str = "train",  # "train", "val", "test"
            center_size: tuple = (12, 16),
    ):
        """
        Args:
            pkl_root: heatmap pkl directory path
            csv_root: blink annotation csv directory path
            subfolders: list of subfolders, e.g. ["0503", "0505"]
            val_pkl_dir: validation set pkl directory path (enabled when is_val=True)
            val_csv_dir: validation set csv directory path (enabled when is_val=True)
            is_val: whether this is the validation set
            center_size: center crop size, default (12, 16)
        """
        self.center_size = center_size
        self.split = split
        self.data = []
        # load all data
        all_data = self.load_all_data(pkl_root, csv_root, subfolders)
        segments = [all_data[i:i+SEGMENT_LEN] for i in range(0, len(all_data), SEGMENT_LEN)]
        set_random_seeds()  # use utils.py's random seed function
        np.random.shuffle(segments)

        # divide segments into train, val, test
        train_count = int(len(segments) * TRAIN_RATIO)
        val_count = int(len(segments) * VAL_RATIO)
        test_count = len(segments) - train_count - val_count

        self.train_segments = segments[:train_count]
        self.val_segments = segments[train_count:train_count + val_count]
        self.test_segments = segments[train_count + val_count:]

        self.update_data()

    def load_all_data(self, pkl_root, csv_root, subfolders):
        """
        load all data from pkl and csv files
        """
        all_data = []
        for subfolder in subfolders:
            pkl_dir = os.path.join(pkl_root, subfolder)
            for filename in os.listdir(pkl_dir):
                if not filename.endswith(".pkl"):
                    continue
                pkl_path = os.path.join(pkl_dir, filename)
                base_name = os.path.splitext(filename)[0]
                parts = base_name.split('_')
                fuzzy_key = parts[-3] + "_" + parts[-2][:4]

                csv_subdir = os.path.join(csv_root, subfolder)
                if not os.path.exists(csv_subdir):
                    print(f"[WARN] CSV subdirectory does not exist: {csv_subdir}")
                    continue
                matched_csvs = [
                    f for f in os.listdir(csv_subdir)
                    if f.startswith("blink_offsets_") and fuzzy_key in f and f.endswith(".csv")
                ]
                if not matched_csvs:
                    print(f"[WARN] No matching CSV found: {fuzzy_key} in {csv_subdir}")
                    continue
                csv_path = os.path.join(csv_subdir, matched_csvs[0])
                frames, labels, _ = self.process_sample(pkl_path, csv_path, return_timestamps=False)
                if frames is not None:
                    N = frames.shape[0]
                    if WINDOW_MODE:
                        # window mode: every SEGMENT_LEN frames as a window
                        for i in range(0, N - SEGMENT_LEN + 1, SEGMENT_LEN):
                            window = frames[i:i+SEGMENT_LEN]  # [SEGMENT_LEN, H, W]
                            label = float(np.any(labels[i:i+SEGMENT_LEN] == 1.0))  # one label for the whole window
                            all_data.append((window, label, None))
                    else:
                        # non-window mode: single frame stacking
                        for i in range(N):
                            frame = frames[i]  # [H, W]
                            stacked_frame = np.repeat(frame[np.newaxis, ...], FRAME_STACK_SIZE, axis=0)  # [FRAME_STACK_SIZE, H, W]
                            label = labels[i]
                            all_data.append((stacked_frame, label, None))  # [FRAME_STACK_SIZE, H, W]
        return all_data

    def update_data(self):
        """
        update the data based on the current split
        """
        if self.split == "train":
            self.data = [item for seg in self.train_segments for item in seg]
        elif self.split == "val":
            self.data = [item for seg in self.val_segments for item in seg]
        elif self.split == "test":
            self.data = [item for seg in self.test_segments for item in seg]
        else:
            raise ValueError(f"unknown split: {self.split}")

    def reshuffle_segments(self):
        """
        every epoch reshuffle train and val segments
        """
        set_random_seeds()  # use utils.py's random seed function
        np.random.shuffle(self.train_segments)
        self.update_data()

    def __len__(self):
        """
        return the length of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        return a single item
        """
        x, y, _ = self.data[idx]
        if WINDOW_MODE:
            # x: [FRAME_STACK_SIZE, H, W]，y: float
            x = torch.tensor(x, dtype=torch.float32)  # [T, H, W]
            y = torch.tensor(y, dtype=torch.float32)  # scalar
            return {"x": x, "y": y}
        else:
            # x: [1, H, W]，y: float
            x = torch.tensor(x, dtype=torch.float32)  # [1, H, W]
            y = torch.tensor(y, dtype=torch.float32)  # scalar
            return {"x": x, "y": y}

    def assign_frame_labels(self, timestamps, blink_starts, blink_ends):
        """
        Assign labels to frames, with start ~ end interval as 1, others as 0
        """
        labels = np.zeros_like(timestamps, dtype=np.float32)
        for start, end in zip(blink_starts, blink_ends):
            labels[(timestamps >= start) & (timestamps <= end)] = 1.0
        return labels


    def process_sample(self, pkl_path, csv_path, return_timestamps=False):
        """
        process a single frame from pkl and csv files
        """
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            if not ('temperature' in data and 'timestamp' in data):
                print(f"[ERROR] lacking fields: {pkl_path}")
                return None, None, None
        except Exception as e:
            print(f"[ERROR] failed to read pkl: {pkl_path}\n{e}")
            return None, None, None

        df = pd.read_csv(csv_path)
        offsets = {row['key']: json.loads(row['value']) for _, row in df.iterrows()}
        blink_start_offsets = offsets["start_offsets"]
        blink_end_offsets = offsets["end_offsets"]
        # if the lengths are not equal, we need to fix them
        if len(blink_start_offsets) != len(blink_end_offsets):
            print(f"[WARN] fixing: {os.path.basename(pkl_path)}")
            if len(blink_end_offsets) > len(blink_start_offsets):
                blink_start_offsets = [0] + blink_start_offsets
                print(f"inserted start=0")
            elif len(blink_start_offsets) > len(blink_end_offsets):
                last_offset = (datetime.fromisoformat(data['timestamp'][-1]) - datetime.fromisoformat(
                    data['timestamp'][0])).total_seconds() * 1000
                blink_end_offsets.append(int(last_offset))
                print(f"inserted end={int(last_offset)}")
        
        # check if the lengths are still not equal after fixing
        if len(blink_start_offsets) != len(blink_end_offsets):
            print(f"[ERROR] fixing failed: {pkl_path}")
            print(len(blink_start_offsets), len(blink_end_offsets))
            return None, None, None
        # temperature_frames = np.array(data['temperature'])  # [N, 12, 16]
        raw_frames = np.array(data['temperature'])  # [N, H, W]
        enhanced_all = []
        for frame in raw_frames:
            # ① Guassian filter
            blurred = cv2.GaussianBlur(frame, (3, 3), sigmaX=0.5)
            # ② standardization and normalization
            enhanced = (blurred - np.mean(blurred)) / (np.std(blurred) + 1e-5)
            enhanced_all.append(enhanced)

        # fix the clip range to [-3, 2]
        global_min = -3.0
        global_max = 2.0
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

        # timestamps & labels per frame
        ts_list = [datetime.fromisoformat(t) for t in data['timestamp']]
        base = ts_list[0]
        timestamps = np.array([(t - base).total_seconds() * 1000 for t in ts_list], dtype=np.float32)
        frame_labels = self.assign_frame_labels(timestamps, blink_start_offsets, blink_end_offsets)
        
        # crop frames to center size
        h, w = frames_np[0].shape
        ch, cw = self.center_size
        sr = h // 2 - ch // 2
        er = sr + ch
        sc = w // 2 - cw // 2
        ec = sc + cw
        cropped_frames = frames_np[:, sr:er, sc:ec]  # [N, H', W']
        X = cropped_frames  # [N, H', W']


        if return_timestamps:
            return X, frame_labels, timestamps
        else:
            return X, frame_labels, None

