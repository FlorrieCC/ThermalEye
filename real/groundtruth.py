import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from matplotlib.patches import Patch


def process_and_plot_temperature(pkl_path, sensor_type=0, start_time_ms=None, end_time_ms=None,
                                 blink_start_offsets=None, blink_end_offsets=None):
    # sensor_type: 0 for 24x32, 1 for 12x16
    if sensor_type == 0:
        fps_list = [6.43, 6.47, 6.50, 6.53, 6.56]
        center_size = (10, 10)
    elif sensor_type == 1:
        fps_list = [15.43, 15.47, 15.50, 15.53, 15.56]
        center_size = (5, 5)
    else:
        raise ValueError("sensor_type must be 0 (24x32) or 1 (12x16)")

    # 读取数据
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # 兼容旧格式和新格式（带temperature和timestamp）
    if isinstance(data, dict) and 'temperature' in data and 'timestamp' in data:
        temperature_frames = data['temperature']
        from datetime import datetime

        # 将字符串时间戳转换为毫秒时间戳（从第一个时间点起）
        raw_timestamps = data['timestamp']
        parsed_times = [datetime.fromisoformat(ts) for ts in raw_timestamps]
        start_time = parsed_times[0]
        timestamps = np.array([(t - start_time).total_seconds() * 1000 for t in parsed_times])  # 毫秒单位
    else:
        raise ValueError("pkl文件必须包含'temperature'和'timestamp'字段")

    # 帧率和理论间隔
    fps = 15.5
    expected_interval = 1000 / fps  # 理论帧间隔（ms）
    threshold = 5.0  # 阈值设定（单位：ms）

    # 延迟分析（仅输出超过阈值的帧）
    for i in range(1, len(timestamps)):
        actual_interval = timestamps[i] - timestamps[i - 1]
        delay = actual_interval - expected_interval
        if abs(delay) > threshold:
            print(f"[!] Frame {i}: Delay = {delay:+.2f} ms (exceeds ±{threshold} ms)")

    total_frames = len(temperature_frames)

    # 获取帧中心区域
    h, w = temperature_frames[0].shape
    ch, cw = center_size
    start_row = h // 2 - ch // 2
    end_row = start_row + ch
    start_col = w // 2 - cw // 2
    end_col = start_col + cw

    # 提取中心区域平均温度
    avg_temp_per_frame = []
    for frame in temperature_frames:
        center_region = frame[start_row:end_row, start_col:end_col]
        avg_temp = np.mean(center_region[center_region > 0])
        avg_temp_per_frame.append(avg_temp)
    avg_temp_per_frame = np.array(avg_temp_per_frame)

    # 平滑拟合（Savitzky-Golay滤波）
    window_length = 21  # 窗口长度（必须奇数）
    polyorder = 2  # 多项式阶数
    smooth_temp = savgol_filter(avg_temp_per_frame, window_length, polyorder)

    # 计算噪声范围（标准差）
    residuals = avg_temp_per_frame - smooth_temp
    noise_std = np.std(residuals)

    # 构造时间轴（用timestamp）
    time_axis = timestamps

    if start_time_ms is not None:
        start_frame = np.searchsorted(time_axis, start_time_ms)
        end_frame = np.searchsorted(time_axis, end_time_ms) if end_time_ms else total_frames

        clip_time = time_axis[start_frame:end_frame]
        clip_temp = avg_temp_per_frame[start_frame:end_frame]
        clip_smooth = smooth_temp[start_frame:end_frame]

        plt.rcParams.update({'pdf.fonttype': 42})
        plt.rcParams.update({'ps.fonttype': 42})
        plt.rcParams.update({'font.size': 16})

        plt.figure(figsize=(12, 5))
        # plt.plot(clip_time, clip_temp, color='darkorange', linewidth=1.5, label='Raw Temp')
        plt.plot(clip_time, clip_smooth, color='blue', linewidth=2, label='Smoothed Temp')
        plt.fill_between(clip_time, clip_smooth - 2 * noise_std, clip_smooth + 2 * noise_std,
                         color='gray', alpha=0.3, label='Noise Range (±2σ)')

        # 叠加背景色
        if blink_start_offsets and blink_end_offsets:
            events = sorted(list(zip(blink_start_offsets, ['start'] * len(blink_start_offsets))) +
                            list(zip(blink_end_offsets, ['end'] * len(blink_end_offsets))))
            last_time = start_time_ms
            last_type = 'end'
            for time_point, event_type in events:
                if time_point < start_time_ms or time_point > end_time_ms:
                    continue
                plt.axvspan(last_time, time_point,
                            color='lightgreen' if last_type == 'start' else 'lightpink', alpha=0.3)
                last_time = time_point
                last_type = event_type
            plt.axvspan(last_time, clip_time[-1],
                        color='lightgreen' if last_type == 'start' else 'lightpink', alpha=0.3)

        # 背景说明图例
        custom_patches = [
            Patch(facecolor='lightpink', edgecolor='none', alpha=0.3, label='Eyes Open'),
            Patch(facecolor='lightgreen', edgecolor='none', alpha=0.3, label='Eyes Closed')
        ]

        # 叠加竖线
        if blink_start_offsets:
            for t in blink_start_offsets:
                if start_time_ms <= t <= end_time_ms:
                    plt.axvline(x=t, color='red', linestyle='--', alpha=0.6, linewidth=1.2, label='Blink Start')
        if blink_end_offsets:
            for t in blink_end_offsets:
                if start_time_ms <= t <= end_time_ms:
                    plt.axvline(x=t, color='green', linestyle='--', alpha=0.6, linewidth=1.2, label='Blink End')

        # 图例去重 + 自定义图例
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(list(by_label.values()) + custom_patches,
                   list(by_label.keys()) + [p.get_label() for p in custom_patches],
                   loc='upper right')

        plt.title(f"Smoothed Temp with Noise Range ({start_time_ms}ms - {clip_time[-1]:.0f}ms)", fontsize=13)
        plt.xlabel("Time (ms)", fontsize=12)
        plt.ylabel("Average Temperature (°C)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig("blink_output" + ".pdf", dpi=300, bbox_inches='tight')
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
    pkl_path="D:/Projects/PyCharmProject/AIoT/EyeDetection/0505/callibration_20250505_161542_107.pkl",
    sensor_type=1,
    start_time_ms=20000,
    end_time_ms=120000,
    blink_start_offsets=new_start,
    blink_end_offsets=new_end
)
