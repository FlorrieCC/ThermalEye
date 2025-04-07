import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import sys
from datetime import datetime

def parse_args():
    """
    解析命令行参数，判断是否保存数据，默认不保存
    """
    save_flag = True
    if len(sys.argv) > 1:
        save_flag = sys.argv[1].lower() not in ['false', '0', 'no']
    return save_flag

def check_device():
    """
    检查是否有 RealSense 设备连接，如果没有则退出程序
    """
    ctx = rs.context()
    if len(ctx.query_devices()) == 0:
        print("没有检测到 RealSense 设备，程序退出。")
        sys.exit(1)

def create_save_dir(save_flag, save_dir):
    """
    如果需要保存数据且保存目录不存在，则创建保存目录
    """
    if save_flag and not os.path.exists(save_dir):
        os.makedirs(save_dir)

def initialize_pipeline(frame_width, frame_height, fps):
    """
    初始化 RealSense pipeline 和 config，返回 pipeline 对象
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, frame_width, frame_height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, frame_width, frame_height, rs.format.bgr8, fps)
    pipeline.start(config)
    return pipeline

def new_filename(save_dir, file_index):
    """
    生成带时间戳和编号的 MP4 文件名，保存在指定目录下
    """
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(save_dir, f"output_{now_str}_{file_index}.mp4")

def main():
    # 参数设置
    SAVE_INTERVAL = 2 * 60  # 每隔 2 分钟保存一个文件
    FRAME_WIDTH, FRAME_HEIGHT = 640, 480
    FPS = 15
    SAVE_DIR = "real_data"  # 保存目录
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 编码格式

    # 解析命令行参数
    save_flag = parse_args()
    
    # 检查 RealSense 设备是否连接
    check_device()
    
    # 创建保存目录（如果需要保存数据）
    create_save_dir(save_flag, SAVE_DIR)
    
    # 初始化 pipeline
    pipeline = initialize_pipeline(FRAME_WIDTH, FRAME_HEIGHT, FPS)
    
    # 初始化保存变量
    start_time = time.time()
    file_index = 1
    out = None
    if save_flag:
        current_filename = new_filename(SAVE_DIR, file_index)
        out = cv2.VideoWriter(current_filename, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
        print("开始保存新文件：", current_filename)
    
    try:
        while True:
            # 获取 RealSense 帧
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # 转为 Numpy 数组
            color_image = np.asanyarray(color_frame.get_data())

            # 保存视频帧（如果需要保存数据）
            if save_flag:
                out.write(color_image)

            # 显示视频帧
            cv2.imshow("RealSense", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 检查是否超时，需要切换保存文件
            if save_flag:
                elapsed = time.time() - start_time
                if elapsed >= SAVE_INTERVAL:
                    out.release()
                    file_index += 1
                    current_filename = new_filename(SAVE_DIR, file_index)
                    out = cv2.VideoWriter(current_filename, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
                    print("开始保存新文件：", current_filename)
                    start_time = time.time()
    except KeyboardInterrupt:
        print("\n用户中断, 关闭realsense")
    except Exception as e:
        print("Error:", e)
    finally:
        if save_flag and out is not None:
            out.release()
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
