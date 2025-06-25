import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import json
from datetime import datetime

# ✅ Step 1: 读取pkl数据
pkl_path = "/Users/yvonne/Documents/final project/ThermalEye/ira_data/0611/xx_left_mild_20250611_165549_726.pkl"
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)
frames = data["temperature"]
timestamps = data["timestamp"]
base_time = datetime.fromisoformat(timestamps[0])

# ✅ Step 2: 读取blink区间
csv_path = "/Users/yvonne/Documents/final project/ThermalEye/gt_output/0611/blink_offsets_xx_left_mild_20250611_165550_361.csv"
df = pd.read_csv(csv_path)
start_offsets = list(map(int, json.loads(df[df["key"] == "start_offsets"]["value"].values[0])))
end_offsets = list(map(int, json.loads(df[df["key"] == "end_offsets"]["value"].values[0])))

# ✅ Step 3: 判断是否为闭眼帧
def is_blink_frame(ts):
    for start, end in zip(start_offsets, end_offsets):
        if start <= ts <= end:
            return True
    return False

# ✅ Step A: 预计算全局增强后的最小值和最大值（用于统一clip）
enhanced_all = []

for frame in frames:
    blurred = cv2.GaussianBlur(frame, (3, 3), sigmaX=0.5)
    enhanced = (blurred - np.mean(blurred)) / (np.std(blurred) + 1e-5)
    enhanced_all.append(enhanced)

enhanced_stack = np.stack(enhanced_all, axis=0)  # [N, H, W]
global_min = enhanced_stack.min()
global_max = enhanced_stack.max()

print(f"🌡️ 全序列归一化后温度范围: min={global_min:.3f}, max={global_max:.3f}")


# ✅ Step 4: 预处理并分类帧
# blink_frames, open_frames = [], []
processed_frames = []
resize_shape = (160, 120)

for frame, ts in zip(frames, timestamps):
    # current_time = datetime.fromisoformat(ts)
    # ts_offset = int((current_time - base_time).total_seconds() * 1000)  # 转为毫秒
    # ① 高斯滤波去噪
    blurred = cv2.GaussianBlur(frame, (3, 3), sigmaX=0.5)

    # ② 标准差归一化
    mean = np.mean(blurred)
    std = np.std(blurred)
    enhanced = (blurred - mean) / (std + 1e-5)
    
    # ✅ 热区裁剪：限制温度范围在 [-1, 2]
    # clipped = np.clip(enhanced, -1.0, 2.0)
    clipped = np.clip(enhanced, global_min, global_max)  #全局最值

    # ③ 映射到 [0, 255] 并转为 uint8（CLAHE 要求）
    # norm = ((enhanced - enhanced.min()) / (enhanced.max() - enhanced.min()) * 255).astype(np.uint8)
    norm_0_1 = (clipped - clipped.min()) / (clipped.max() - clipped.min())
    gamma = 0.5
    adjusted = np.power(norm_0_1, gamma)

    # ✅ 转为 0~255 的 uint8 图像
    norm = (adjusted * 255).astype(np.uint8)

    # ④ CLAHE 对比度增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
    contrast_enhanced = clahe.apply(norm)

    # ⑤ resize
    resized = cv2.resize(contrast_enhanced, resize_shape, interpolation=cv2.INTER_NEAREST)


    # if is_blink_frame(ts_offset):
    #     blink_frames.append(resized)
    # else:
    #     open_frames.append(resized)
    processed_frames.append(resized)
    

# ✅ Step 5: 坐标拼接工具
def create_composite(images, cols=50, resize_shape=(160, 120)):
    if len(images) == 0:
        return None
    rows = (len(images) + cols - 1) // cols
    width, height = resize_shape
    canvas = np.zeros((rows * height, cols * width), dtype=np.uint8)

    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        y0, y1 = r * height, (r + 1) * height
        x0, x1 = c * width, (c + 1) * width
        canvas[y0:y1, x0:x1] = img
    return canvas

# # ✅ Step 6: 合并拼图
# blink_canvas = create_composite(blink_frames)
# open_canvas = create_composite(open_frames)

# if blink_canvas is None or open_canvas is None:
#     print("❗ 数据不足，某一类帧为空")
# else:
#     gap_height = 50
#     gap = np.zeros((gap_height, blink_canvas.shape[1]), dtype=np.uint8)
#     total_canvas = cv2.vconcat([blink_canvas, gap, open_canvas])

#     # ✅ Step 7: 显示总图
#     plt.figure(figsize=(20, 12))
#     plt.imshow(total_canvas, cmap='viridis')
#     plt.title("Top: Blinking Frames (闭眼), Bottom: Open Eye Frames (睁眼)")
#     plt.axis('off')
#     plt.tight_layout()
#     plt.colorbar()
#     plt.show()
group_size = 160       # 每组帧数
cols = 50              # 每组最多拼成几列（宽度）
gap_height = 20        # 空白行高度
width, height = resize_shape

rows = []

for i in range(0, len(processed_frames), group_size):
    group = processed_frames[i:i + group_size]
    group_canvas = create_composite(group, cols=cols, resize_shape=resize_shape)  # 按列数拼
    rows.append(group_canvas)
    # 添加空白行
    gap = np.zeros((gap_height, group_canvas.shape[1]), dtype=np.uint8)
    rows.append(gap)

# 最后一组可能不用 gap
final_canvas = cv2.vconcat(rows[:-1]) if len(rows) > 1 else rows[0]

# 显示图像（无colorbar）
plt.figure(figsize=(20, 12))
plt.imshow(final_canvas, cmap='jet')  # 热点颜色更加直观
# plt.imshow(final_canvas, cmap='viridis')
plt.title("All Frames in Order (每160帧为一组，组间隔空行)")
plt.axis('off')
plt.tight_layout()
plt.show()

