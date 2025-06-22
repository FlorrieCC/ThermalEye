import os
import cv2
import pickle
import shutil
from datetime import datetime, timedelta

def backup_and_replace(file_path, temp_path):
    folder = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    backup_dir = os.path.join(folder, "backup")
    os.makedirs(backup_dir, exist_ok=True)

    backup_path = os.path.join(backup_dir, filename)
    shutil.copy2(file_path, backup_path)
    os.remove(file_path)
    shutil.move(temp_path, file_path)

    print(f"[BACKUP] 已备份到: {backup_path}")
    print(f"[SAVED] 已替换原文件: {file_path}")


def align_timestamps(mp4_path, pkl_dir):
    base_name = os.path.splitext(os.path.basename(mp4_path))[0]
    video_dir = os.path.dirname(mp4_path)
    txt_path = os.path.join(video_dir, f"ts_{base_name}.txt")

    with open(txt_path, 'r') as f:
        txt_lines = [line.strip() for line in f if line.strip()]
    txt_times = [datetime.fromisoformat(ts) for ts in txt_lines]

    parts = base_name.split('_')
    match_token = parts[-3] + "_" + parts[-2][:4]
    matched_pkl = None
    for f in os.listdir(pkl_dir):
        if f.endswith(".pkl") and match_token in f:
            matched_pkl = os.path.join(pkl_dir, f)
            break
    if not matched_pkl:
        print(f"[ERROR] ❌ 未找到匹配的 .pkl: {base_name}")
        return None, None, None, None

    with open(matched_pkl, 'rb') as f:
        pkl_data = pickle.load(f)

    pkl_times_raw = pkl_data['timestamp']
    pkl_temps = pkl_data['temperature']
    pkl_times = [datetime.fromisoformat(t) for t in pkl_times_raw]

    delta_start = (pkl_times[0] - txt_times[0]).total_seconds() * 1000
    delta_end = (pkl_times[-1] - txt_times[-1]).total_seconds() * 1000

    if abs(delta_start) < 100 and abs(delta_end) < 100:
        print(f"[SKIP] ✅ 时间戳已对齐：{base_name}")
        print(f"       首帧差: {delta_start:.1f} ms，末帧差: {delta_end:.1f} ms")
        return txt_lines, pkl_data, mp4_path, matched_pkl

    print(f"[ALIGN] ⏱️ 对齐时间戳: {base_name}")
    print(f"       首帧差: {delta_start:.1f} ms，末帧差: {delta_end:.1f} ms")

    # 对齐
    txt_start = txt_times[0]
    txt_end = txt_times[-1]
    pkl_start = pkl_times[0]
    pkl_end = pkl_times[-1]

    if pkl_start <= txt_start:
        # pkl 更早
        new_pkl_indices = [i for i, t in enumerate(pkl_times) if t >= txt_start]
        new_txt_indices = [i for i, t in enumerate(txt_times) if t <= pkl_end]
    else:
        # ⛔ pkl 比 txt 晚
        new_pkl_indices = list(range(len(pkl_times)))  # 保留全部 pkl
        new_txt_indices = [i for i, t in enumerate(txt_times) if t >= pkl_start and t <= pkl_end]

    new_pkl_data = {
        'temperature': [pkl_temps[i] for i in new_pkl_indices],
        'timestamp': [pkl_times_raw[i] for i in new_pkl_indices]
    }
    new_txt_lines = [txt_lines[i] for i in new_txt_indices]

    return new_txt_lines, new_pkl_data, mp4_path, matched_pkl

def truncate_data(txt_lines, pkl_data, mp4_path, pkl_path, truncate_time_ms):
    # 1. 截断 txt
    txt_times = [datetime.fromisoformat(ts) for ts in txt_lines]
    base_time = txt_times[0]
    timestamps = [int((t - base_time).total_seconds() * 1000) for t in txt_times]

    cutoff = timestamps[0] + truncate_time_ms
    keep_indices_txt = [i for i, t in enumerate(timestamps) if t >= cutoff]
    truncated_txt = [txt_lines[i] for i in keep_indices_txt]

    # 2. 截断 pkl
    ts_objs = [datetime.fromisoformat(t) for t in pkl_data['timestamp']]
    base_ts = ts_objs[0]
    cutoff_ts = base_ts + timedelta(milliseconds=truncate_time_ms)
    keep_indices_pkl = [i for i, t in enumerate(ts_objs) if t >= cutoff_ts]

    truncated_pkl = {
        'temperature': [pkl_data['temperature'][i] for i in keep_indices_pkl],
        'timestamp': [pkl_data['timestamp'][i] for i in keep_indices_pkl]
    }

    # 3. 截断 mp4（生成临时路径）
    cap = cv2.VideoCapture(mp4_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    start_frame = keep_indices_txt[0] if keep_indices_txt else total_frames
    base_name_mp4 = os.path.splitext(os.path.basename(mp4_path))[0]
    temp_mp4_path = os.path.join(os.path.dirname(mp4_path), f"__temp__{base_name_mp4}.mp4")
    out = cv2.VideoWriter(temp_mp4_path, fourcc, fps, (width, height))

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i >= start_frame:
            out.write(frame)
    cap.release()
    out.release()
    
    # === 写入临时 .txt 文件 ===
    base_name_txt = os.path.splitext(os.path.basename(mp4_path))[0]
    temp_txt_path = os.path.join(os.path.dirname(mp4_path), f"__temp__{base_name_txt}.txt")
    with open(temp_txt_path, 'w') as f:
        for line in truncated_txt:
            f.write(line + "\n")

    # === 写入临时 .pkl 文件 ===
    base_name_pkl = os.path.splitext(os.path.basename(pkl_path))[0]
    temp_pkl_path = os.path.join(os.path.dirname(pkl_path), f"__temp__{base_name_pkl}.pkl")
    with open(temp_pkl_path, 'wb') as f:
        pickle.dump(truncated_pkl, f)

    # ✅ 返回临时文件路径
    return temp_txt_path, temp_pkl_path, temp_mp4_path 

def process_single(mp4_path, pkl_dir, truncate_time_ms):
    aligned_txt, aligned_pkl, aligned_mp4_path, pkl_path = align_timestamps(mp4_path, pkl_dir)
    if aligned_txt is None or aligned_pkl is None or aligned_mp4_path is None or pkl_path is None:
        print(f"[ERROR] 数据为空, 未能成功对齐时间戳")
        return

    # 如果对齐后想仅查看时间差，把下面注释掉
    temp_txt_path, temp_pkl_path, temp_mp4_path = truncate_data(
        aligned_txt, aligned_pkl, aligned_mp4_path, pkl_path, truncate_time_ms
    )

    txt_path = os.path.join(os.path.dirname(mp4_path), f"ts_{os.path.splitext(os.path.basename(mp4_path))[0]}.txt")
    backup_and_replace(txt_path, temp_txt_path)
    backup_and_replace(mp4_path, temp_mp4_path)
    backup_and_replace(pkl_path, temp_pkl_path)

def process_folder(mp4_folder, pkl_folder, truncate_time_ms):
    for fname in os.listdir(mp4_folder):
        if not fname.endswith(".mp4"):
            continue
        if fname.startswith("__temp__"):
            continue  # ⛔ 跳过临时文件
        mp4_path = os.path.join(mp4_folder, fname)
        print(f"\n[📁] 正在处理文件: {fname}")
        process_single(mp4_path, pkl_folder, truncate_time_ms)
        
        
if __name__ == "__main__":
    mode = 'single'  # 改成 'batch' 可切换为批量模式
    truncate_time_ms = 5000

    if mode == 'single':
        mp4_path = "/Users/yvonne/Documents/final project/ThermalEye/real_data/0618/shy_bottom_cold_mild_20250619_040434_126.mp4"
        pkl_dir = "/Users/yvonne/Documents/final project/ThermalEye/ira_data/0618"
        process_single(mp4_path, pkl_dir, truncate_time_ms)

    elif mode == 'batch':
        mp4_folder = "/Users/yvonne/Documents/final project/ThermalEye/real_data/0618"
        pkl_folder = "/Users/yvonne/Documents/final project/ThermalEye/ira_data/0618"
        process_folder(mp4_folder, pkl_folder, truncate_time_ms)

    else:
        print("[ERROR] 无效模式，请设置 mode = 'single' 或 'folder'")
