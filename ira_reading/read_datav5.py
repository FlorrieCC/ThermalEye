"""
    v5
    增加横坐标偏移
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.signal import savgol_filter


def process_and_plot_temperature(pkl_path, sensor_type=0, start_time_ms=None, end_time_ms=None,
                                 blink_start_offsets=None, blink_end_offsets=None,
                                 zero_based_time_axis=False):
    # sensor_type: 0 for 24x32, 1 for 12x16
    if sensor_type == 0:
        fps_list = [6.43, 6.47, 6.50, 6.53, 6.56]
        center_size = (10, 10)
    elif sensor_type == 1:
        fps_list = [15.43, 15.47, 15.50, 15.53, 15.56]
        center_size = (5, 5)
    else:
        raise ValueError("sensor_type must be 0 (24x32) or 1 (12x16)")

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict) and 'temperature' in data and 'timestamp' in data:
        temperature_frames = data['temperature']
        from datetime import datetime
        raw_timestamps = data['timestamp']
        parsed_times = [datetime.fromisoformat(ts) for ts in raw_timestamps]
        start_time = parsed_times[0]
        timestamps = np.array([(t - start_time).total_seconds() * 1000 for t in parsed_times])
    else:
        raise ValueError("pkl文件必须包含'temperature'和'timestamp'字段")

    fps = 15.5
    expected_interval = 1000 / fps
    threshold = 5.0

    for i in range(1, len(timestamps)):
        actual_interval = timestamps[i] - timestamps[i - 1]
        delay = actual_interval - expected_interval
        if abs(delay) > threshold:
            print(f"[!] Frame {i}: Delay = {delay:+.2f} ms (exceeds ±{threshold} ms)")

    total_frames = len(temperature_frames)

    h, w = temperature_frames[0].shape
    ch, cw = center_size
    start_row = h // 2 - ch // 2
    end_row = start_row + ch
    start_col = w // 2 - cw // 2
    end_col = start_col + cw

    avg_temp_per_frame = []
    for frame in temperature_frames:
        center_region = frame[start_row:end_row, start_col:end_col]
        avg_temp = np.mean(center_region[center_region > 0])
        avg_temp_per_frame.append(avg_temp)
    avg_temp_per_frame = np.array(avg_temp_per_frame)

    window_length = 21
    polyorder = 2
    smooth_temp = savgol_filter(avg_temp_per_frame, window_length, polyorder)
    residuals = avg_temp_per_frame - smooth_temp
    noise_std = np.std(residuals)

    time_axis = timestamps

    if start_time_ms is not None:
        start_frame = np.searchsorted(time_axis, start_time_ms)
        end_frame = np.searchsorted(time_axis, end_time_ms) if end_time_ms else total_frames

        clip_time = time_axis[start_frame:end_frame]
        clip_temp = avg_temp_per_frame[start_frame:end_frame]
        clip_smooth = smooth_temp[start_frame:end_frame]

        # 横坐标从 0 开始 + blink 时间偏移
        time_offset = clip_time[0] if zero_based_time_axis else 0
        clip_time = clip_time - time_offset if zero_based_time_axis else clip_time
        blink_start_shifted = [t - time_offset for t in blink_start_offsets] if blink_start_offsets else []
        blink_end_shifted = [t - time_offset for t in blink_end_offsets] if blink_end_offsets else []

        plt.rcParams.update({'pdf.fonttype': 42})
        plt.rcParams.update({'ps.fonttype': 42})
        plt.rcParams.update({'font.size': 16})

        plt.figure(figsize=(8, 4))
        plt.plot(clip_time, clip_temp, color='#472A79', linewidth=1.5, label='Raw Temp')
        # plt.plot(clip_time, clip_smooth, color='blue', linewidth=2, label='Smoothed Temp')
        # plt.fill_between(clip_time, clip_smooth - 2 * noise_std, clip_smooth + 2 * noise_std,
        #                  color='gray', alpha=0.3, label='Noise Range (±2σ)')

        if blink_start_shifted and blink_end_shifted:
            events = sorted(list(zip(blink_start_shifted, ['start'] * len(blink_start_shifted))) +
                            list(zip(blink_end_shifted, ['end'] * len(blink_end_shifted))))
            last_time = 0
            last_type = 'end'
            for time_point, event_type in events:
                if 0 <= time_point <= clip_time[-1]:
                    plt.axvspan(last_time, time_point,
                                color='#FCE623' if last_type == 'start' else '#57C665', alpha=0.13)
                    last_time = time_point
                    last_type = event_type
            plt.axvspan(last_time, clip_time[-1],
                        color='#FCE623' if last_type == 'start' else '#57C665', alpha=0.13)

        custom_patches = [
            Patch(facecolor='#FCE623', edgecolor='none', alpha=0.1, label='Eyes Open'),
            Patch(facecolor='#57C665', edgecolor='none', alpha=0.3, label='Eyes Closed')
        ]

        if blink_start_shifted:
            for t in blink_start_shifted:
                if 0 <= t <= clip_time[-1]:
                    plt.axvline(x=t, color='red', linestyle='--', alpha=0.6, linewidth=1.2, label='Blink Start')
        if blink_end_shifted:
            for t in blink_end_shifted:
                if 0 <= t <= clip_time[-1]:
                    plt.axvline(x=t, color='blue', linestyle='--', alpha=0.6, linewidth=1.2, label='Blink End')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(list(by_label.values()) + custom_patches,
                   list(by_label.keys()) + [p.get_label() for p in custom_patches],
                   loc='upper right')

        plt.xlabel("Time (ms)", fontsize=12)
        plt.ylabel("Temperature (°C)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.ylim(33.5, 36)
        # plt.xlim(0, clip_time[-1])
        # 强制写上右端刻度

        # 1. 将最大横轴向上取整为最近的1000倍数
        max_time = int(np.ceil(clip_time[-1] / 1000.0)) * 1000

        # 2. 强制设定x轴范围与刻度
        plt.xlim(0, max_time)
        plt.xticks(np.arange(0, max_time + 1, 20000))  # 每隔20秒（20000ms）一个刻度，你也可设10000

        plt.tight_layout()
        plt.savefig("blink_output.pdf", dpi=300, bbox_inches='tight')
        plt.show()


blink_start = [6000, 6533, 13600, 20533, 28533, 35200, 41533, 49267, 56800, 68267, 77800, 87333, 94600, 103333, 111267]

blink_end = [6400, 8267, 16800, 23733, 31533, 38333, 46133, 54000, 61800, 72667, 82067, 90333, 99333, 107467, 115733]


def adjust_blink_times(start_list, end_list, start_offset=0, end_offset=50):
    """
    调整 blink_start 和 blink_end 的时间偏移量。

    参数：
        start_list (list[int]): 原始 blink_start 列表
        end_list (list[int]): 原始 blink_end 列表
        start_offset (int): blink_start 的偏移量（默认 -500）
        end_offset (int): blink_end 的偏移量（默认 +500）

    返回：
        tuple: (新的 blink_start 列表, 新的 blink_end 列表)
    """
    adjusted_start = [max(0, s + start_offset) for s in start_list]  # 防止负值
    adjusted_end = [e + end_offset for e in end_list]
    return adjusted_start, adjusted_end


new_start, new_end = adjust_blink_times(blink_start, blink_end)

process_and_plot_temperature(
    pkl_path="/Users/yvonne/Documents/final project/ThermalEye/ira_data/0505/callibration_20250505_161542_107.pkl",
    sensor_type=1,
    start_time_ms=20000,
    end_time_ms=80000,
    blink_start_offsets=new_start,
    blink_end_offsets=new_end,
    zero_based_time_axis=True  # 横轴从0开始
)
