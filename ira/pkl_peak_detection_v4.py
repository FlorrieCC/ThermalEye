import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, find_peaks
import pandas as pd
import json



def detect_open_close_intervals(
        pkl_path, sensor_type=1,
        derivative_threshold=-0.0003,
        continuous_fall_window=5,
        min_distance_ms=3000,
        peak_prominence=0.1,
        valley_prominence=0.25,
        smooth_window=21,
        blink_start=None,
        blink_end=None,
        start_time_ms=None,
        end_time_ms=None
):
    # 1. Load data
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    assert 'temperature' in data and 'timestamp' in data

    frames = data['temperature']
    raw_timestamps = [datetime.fromisoformat(ts) for ts in data['timestamp']]
    start_time = raw_timestamps[0]
    full_timestamps = np.array([(t - start_time).total_seconds() * 1000 for t in raw_timestamps])

    h, w = frames[0].shape
    center_size = (10, 10) if sensor_type == 0 else (5, 5)
    ch, cw = center_size
    sr, er = h // 2 - ch // 2, h // 2 + ch // 2
    sc, ec = w // 2 - cw // 2, w // 2 + cw // 2

    avg_temp_all = np.array([
        np.mean(f[sr:er, sc:ec][f[sr:er, sc:ec] > 0])
        for f in frames
    ])

    # 2. Compute smoothed signal and extrema detection from full signal
    smooth_all = savgol_filter(avg_temp_all, smooth_window, 2)
    deriv = np.gradient(smooth_all, full_timestamps)

    fall_indices = [i for i in range(continuous_fall_window, len(deriv))
                    if np.all(deriv[i - continuous_fall_window:i] < derivative_threshold)]

    all_peaks, _ = find_peaks(smooth_all, prominence=peak_prominence, distance=5)
    all_valleys, _ = find_peaks(-smooth_all, prominence=valley_prominence, distance=5)

    matched_peaks = []
    for fall_idx in fall_indices:
        candidates = [p for p in all_peaks if p < fall_idx]
        if candidates:
            matched_peaks.append(candidates[-1])

    dedup_peaks = []
    last_time = -np.inf
    for idx in matched_peaks:
        t = full_timestamps[idx]
        if t - last_time > min_distance_ms:
            dedup_peaks.append(idx)
            last_time = t

    open_times = [int(full_timestamps[i]) for i in dedup_peaks]
    close_times = []
    for peak_idx in dedup_peaks:
        candidates = [v for v in all_valleys if v > peak_idx]
        if candidates:
            close_times.append(int(full_timestamps[candidates[0]]))

    # 3. Slice plotting region
    if start_time_ms is None:
        start_time_ms = 0
    if end_time_ms is None:
        end_time_ms = full_timestamps[-1]

    mask = (full_timestamps >= start_time_ms) & (full_timestamps <= end_time_ms)
    timestamps = full_timestamps[mask] - start_time_ms
    avg_temp = avg_temp_all[mask]
    smooth_temp = smooth_all[mask]

    # Determine indices of dedup_peaks and all_valleys within masked range
    dedup_peaks_masked = [i for i in dedup_peaks if start_time_ms <= full_timestamps[i] <= end_time_ms]
    all_valleys_masked = [i for i in all_valleys if start_time_ms <= full_timestamps[i] <= end_time_ms]
    peak_plot_x = [full_timestamps[i] - start_time_ms for i in dedup_peaks_masked]
    peak_plot_y = [smooth_all[i] for i in dedup_peaks_masked]
    valley_plot_x = [full_timestamps[i] - start_time_ms for i in all_valleys_masked]
    valley_plot_y = [smooth_all[i] for i in all_valleys_masked]

    # 4. Construct predicted yellow intervals: from each valley to next peak
    predicted_intervals = []
    for i in range(len(close_times)):
        if i + 1 < len(open_times):
            start = close_times[i]
            end = open_times[i + 1]
        else:
            start = close_times[i]
            end = open_times[-1]
        if start < end:
            # Allow partial overlap into target region
            if end >= start_time_ms and start <= end_time_ms:
                s_plot = max(start, start_time_ms) - start_time_ms
                e_plot = min(end, end_time_ms) - start_time_ms
                if s_plot < e_plot:
                    predicted_intervals.append((s_plot, e_plot))

    # 5. Plot
    plt.figure(figsize=(12, 6))
    timestamps_sec = timestamps / 1000
    peak_plot_x_sec = np.array(peak_plot_x) / 1000
    valley_plot_x_sec = np.array(valley_plot_x) / 1000
    predicted_intervals_sec = [(s / 1000, e / 1000) for s, e in predicted_intervals]

    plt.plot(timestamps_sec, avg_temp, label='Raw Temp', color='gray', alpha=0.3, linewidth=1)
    plt.plot(timestamps_sec, smooth_temp, label='Smoothed Temp', color='blue', linewidth=1.5)
    plt.scatter(peak_plot_x_sec, peak_plot_y, color='green', marker='^', label='Eye Open (Peak)')
    plt.scatter(valley_plot_x_sec, valley_plot_y, color='red', marker='v', label='Eye Close (Valley)')

    for i, (start, end) in enumerate(predicted_intervals_sec):
        plt.axvspan(start, end, color='yellow', alpha=0.25, label='Predicted Recovery' if i == 0 else "")

    if blink_start and blink_end:
        gt_intervals = [(s, e) for s, e in zip(blink_start, blink_end)
                        if (s >= start_time_ms and s <= end_time_ms) or (e >= start_time_ms and e <= end_time_ms)]
        for i, (s, e) in enumerate(gt_intervals):
            s_adj = max(s, start_time_ms) - start_time_ms
            e_adj = min(e, end_time_ms) - start_time_ms
            if s_adj < e_adj:
                plt.axvspan(s_adj / 1000, e_adj / 1000, color='pink', alpha=0.3,
                            label='Ground Truth Closed' if i == 0 else "")

    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (°C)")
    plt.legend(fontsize=14, loc='upper right')
    plt.xticks(np.arange(0, timestamps_sec.max() + 1, 5))
    plt.ylim([min(avg_temp.min(), smooth_temp.min()) - 0.5, max(avg_temp.max(), smooth_temp.max()) + 0.5])
    y_min = min(avg_temp.min(), smooth_temp.min()) - 0.5
    y_max = max(avg_temp.max(), smooth_temp.max()) + 0.5
    y_ticks = np.round(np.arange(y_min, y_max + 0.2, 0.2), 1)
    plt.yticks(y_ticks)

    # 边框线宽度设为1
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1)

    plt.grid(False)
    plt.tight_layout()
    plt.savefig("/Users/yvonne/Documents/final project/ThermalEye/ira/thermal_peak_output.pdf", dpi=200, bbox_inches='tight')
    plt.show()

    # 6. Compute overlap only in selected time range
    if blink_start and blink_end:
        total_gt = 0
        total_overlap = 0
        for s_gt, e_gt in zip(blink_start, blink_end):
            if e_gt < start_time_ms or s_gt > end_time_ms:
                continue
            s_gt_clip = max(s_gt, start_time_ms) - start_time_ms
            e_gt_clip = min(e_gt, end_time_ms) - start_time_ms
            total_gt += e_gt_clip - s_gt_clip
            for s_pred, e_pred in predicted_intervals:
                inter_start = max(s_pred, s_gt_clip)
                inter_end = min(e_pred, e_gt_clip)
                if inter_start < inter_end:
                    total_overlap += inter_end - inter_start
                    break
        overlap_ratio = total_overlap / total_gt if total_gt > 0 else 0
        print(f"✅ [区间内] 黄色区域与闭眼 Ground Truth 重合比例: {overlap_ratio:.2%}")

    print("Detected Open Times (ms):", open_times)
    print("Detected Close Times (ms):", close_times)
    return open_times, close_times


# ======= 调用 =======
# blink_start = [6000, 6533, 13600, 20533, 28533, 35200, 41533, 49267, 56800, 68267, 77800, 87333, 94600, 103333, 111267]
# blink_end = [6400, 8267, 16800, 23733, 31533, 38333, 46133, 54000, 61800, 72667, 82067, 90333, 99333, 107467, 115733]

# 读取 CSV 并解析 JSON 格式的偏移列表
offset_csv_path = '/Users/yvonne/Documents/final project/ThermalEye/blink_output/blink_offsets_mild_20250517_042435_732.csv'
df = pd.read_csv(offset_csv_path)
offsets = {row['key']: json.loads(row['value']) for _, row in df.iterrows()}

blink_start = offsets['start_offsets']
blink_end = offsets['end_offsets']

open_list, close_list = detect_open_close_intervals(
    pkl_path="/Users/yvonne/Documents/final project/ThermalEye/ira_data/0517/mild_20250517_042434_989.pkl",
    sensor_type=1,
    derivative_threshold=-0.0003,
    continuous_fall_window=5,
    min_distance_ms=3000,
    peak_prominence=0.1,
    valley_prominence=0.25,
    smooth_window=21,
    start_time_ms=26000,
    end_time_ms=86000,
    blink_start=blink_start,
    blink_end=blink_end
)
