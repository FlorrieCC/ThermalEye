# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
#
# def parse_frame_indices(frame_input, max_index):
#     """解析帧输入，支持 '22:70' 和 '22,23,25' 形式"""
#     if isinstance(frame_input, str):
#         if ':' in frame_input:
#             start, end = map(int, frame_input.split(':'))
#             return list(range(start, min(end + 1, max_index)))
#         else:
#             return [int(i) for i in frame_input.split(',') if i.isdigit()]
#     elif isinstance(frame_input, list):
#         return [i for i in frame_input if isinstance(i, int) and 0 <= i < max_index]
#     else:
#         return []
#
# def plot_subframes(images, indices, cmap='viridis', cols=10, figsize=(20, 8)):
#     """将多个帧以子图方式展示"""
#     rows = (len(indices) + cols - 1) // cols
#     fig, axs = plt.subplots(rows, cols, figsize=figsize)
#     axs = axs.flatten()
#
#     for ax in axs[len(indices):]:
#         ax.axis('off')  # 隐藏多余子图
#
#     for i, idx in enumerate(indices):
#         ax = axs[i]
#         ax.imshow(images[idx], cmap=cmap)
#         ax.set_title(f'Frame {idx}')
#         ax.axis('off')
#
#     plt.tight_layout()
#     plt.show()
#
# def visualize_temperature_frames_with_std(pkl_path, cols=50, resize_shape=(160, 120),
#                                           cmap='viridis', highlight_frames=None):
#     """主函数：读取PKL、增强图像、拼接全部帧、并可视化指定帧为子图"""
#     # 加载数据
#     with open(pkl_path, 'rb') as f:
#         data = pickle.load(f)
#
#     # 解析温度帧
#     if isinstance(data, dict) and "temperature" in data:
#         frames = data["temperature"]
#     elif isinstance(data, list) and isinstance(data[0], np.ndarray):
#         frames = data
#     else:
#         raise ValueError("pkl文件格式不正确，应为'dict'或'ndarray'列表")
#
#     # 标准差归一化 + 尺寸统一
#     enhanced_images = []
#     for frame in frames:
#         mean = np.mean(frame)
#         std = np.std(frame)
#         enhanced = (frame - mean) / (std + 1e-5)
#         norm_enhanced = ((enhanced - enhanced.min()) / (enhanced.max() - enhanced.min()) * 255).astype(np.uint8)
#         resized = cv2.resize(norm_enhanced, resize_shape, interpolation=cv2.INTER_NEAREST)
#         enhanced_images.append(resized)
#
#     # 拼接所有帧图像为大图
#     rows = (len(enhanced_images) + cols - 1) // cols
#     height, width = resize_shape[1], resize_shape[0]
#     composite = np.zeros((rows * height, cols * width), dtype=np.uint8)
#
#     for idx, img in enumerate(enhanced_images):
#         r, c = divmod(idx, cols)
#         composite[r * height:(r + 1) * height, c * width:(c + 1) * width] = img
#
#     # 展示拼接图像
#     plt.figure(figsize=(20, 10))
#     plt.imshow(composite, cmap=cmap)
#     plt.axis('off')
#     plt.title('All Frames Enhanced by STD Normalization')
#     plt.colorbar()
#     plt.show()
#
#     # 展示选中的帧（子图形式）
#     if highlight_frames:
#         indices = parse_frame_indices(highlight_frames, len(enhanced_images))
#         if indices:
#             plot_subframes(enhanced_images, indices, cmap=cmap, cols=10, figsize=(20, 8))
#         else:
#             print("未解析出有效帧编号。")
#
# # ✅ 用法示例
# visualize_temperature_frames_with_std(
#     "D:/Projects/PyCharmProject/AIoT/EyeDetection/0505/callibration_20250505_161542_107.pkl",
#     highlight_frames="555,584"  # 或者 "22,23,25"
# )
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2

def parse_frame_indices(frame_input, max_index):
    """解析帧输入，支持 '22:70' 和 '22,23,25' 形式"""
    if isinstance(frame_input, str):
        if ':' in frame_input:
            start, end = map(int, frame_input.split(':'))
            return list(range(start, min(end + 1, max_index)))
        else:
            return [int(i) for i in frame_input.split(',') if i.isdigit()]
    elif isinstance(frame_input, list):
        return [i for i in frame_input if isinstance(i, int) and 0 <= i < max_index]
    else:
        return []

def plot_selected_frames(images, indices, cmap='viridis', figsize=(10, 8), output_path='selected_frames.pdf'):
    """仅展示并保存选中的两帧图像"""
    fig, axs = plt.subplots(len(indices), 1, figsize=figsize)

    if len(indices) == 1:
        axs = [axs]  # 保证 axs 可迭代

    for ax, idx in zip(axs, indices):
        ax.imshow(images[idx], cmap=cmap)
        ax.set_title(f'Frame {idx}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.close()
    print(f"已保存为PDF: {output_path}")

def visualize_two_frames_and_save(pkl_path, highlight_frames, resize_shape=(160, 120), cmap='viridis'):
    """主函数：读取PKL、增强图像、仅处理并保存两帧"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "temperature" in data:
        frames = data["temperature"]
    elif isinstance(data, list) and isinstance(data[0], np.ndarray):
        frames = data
    else:
        raise ValueError("pkl文件格式不正确，应为'dict'或'ndarray'列表")

    enhanced_images = []
    for frame in frames:
        mean = np.mean(frame)
        std = np.std(frame)
        enhanced = (frame - mean) / (std + 1e-5)
        norm_enhanced = ((enhanced - enhanced.min()) / (enhanced.max() - enhanced.min()) * 255).astype(np.uint8)
        resized = cv2.resize(norm_enhanced, resize_shape, interpolation=cv2.INTER_NEAREST)
        enhanced_images.append(resized)

    indices = parse_frame_indices(highlight_frames, len(enhanced_images))
    if not indices:
        print("未解析出有效帧编号。")
        return

    plot_selected_frames(enhanced_images, indices, cmap=cmap, figsize=(6, 6 * len(indices)))

# ✅ 用法
visualize_two_frames_and_save(
    "D:/Projects/PyCharmProject/AIoT/EyeDetection/0505/callibration_20250505_161542_107.pkl",
    highlight_frames="555,584"
)
