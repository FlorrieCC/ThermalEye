import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ✅ Step 1: 读取数据
pkl_path = "/Users/yvonne/Documents/final project/ThermalEye/ira_data/0517/normal_20250517_042030_643.pkl"
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)
frames = data["temperature"]

# ✅ Step 2: 图像增强并resize
enhanced_images = []
resize_shape = (160, 120)  # 或 (32, 24)
for frame in frames:
    mean = np.mean(frame)
    std = np.std(frame)
    enhanced = (frame - mean) / (std + 1e-5)
    norm = ((enhanced - enhanced.min()) / (enhanced.max() - enhanced.min()) * 255).astype(np.uint8)
    resized = cv2.resize(norm, resize_shape, interpolation=cv2.INTER_NEAREST)
    enhanced_images.append(resized)

# ✅ Step 3: 拼图成一张大图
cols = 50
rows = (len(enhanced_images) + cols - 1) // cols

canvas = []
for r in range(rows):
    row_imgs = enhanced_images[r * cols:(r + 1) * cols]
    if len(row_imgs) < cols:
        pad = np.zeros_like(row_imgs[0])
        row_imgs += [pad] * (cols - len(row_imgs))  # 补齐空位
    canvas.append(cv2.hconcat(row_imgs))
final_image = cv2.vconcat(canvas)

# ✅ Step 4: 显示为单张图
plt.figure(figsize=(10, 8))
plt.imshow(final_image, cmap='viridis')
plt.title("All Frames Enhanced by STD Normalization")
plt.axis('off')
plt.colorbar()
plt.show()
