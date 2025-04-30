import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import sys
from datetime import datetime

def parse_args():
    """
    解析命令行参数：
    sys.argv[1] -- save_flag (控制是否保存数据，传入 False/0/no 则关闭保存，默认为 True)
    sys.argv[2] -- run_time (程序运行时间，单位秒，不传入或为0则无限制运行)
    sys.argv[3] -- save_dir (保存数据文件夹，不传入则默认为 "real_data")
    sys.argv[4] -- save_name (保存文件名前缀，不传入则默认为 "output")
    """
    # 默认值
    save_flag = True
    run_time = 0   # 无限运行
    save_dir = "real_data"
    save_name = "output"
    
    if len(sys.argv) > 1:
        save_flag = sys.argv[1].lower() not in ['false', '0', 'no']
    if len(sys.argv) > 2:
        try:
            run_time = float(sys.argv[2])
        except ValueError:
            print("运行时间参数不合法，使用默认无限制运行")
            run_time = 0
    if len(sys.argv) > 3:
        save_dir = sys.argv[3]
    if len(sys.argv) > 4:
        save_name = sys.argv[4]
    
    return save_flag, run_time, save_dir, save_name

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

def new_filename(save_dir, save_name, file_index):
    """
    生成带时间戳和编号的 MP4 文件名，保存在指定目录下，
    文件名前缀使用命令行参数 save_name。
    """
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(save_dir, f"{save_name}_{now_str}_{file_index}.mp4")

def record_realsense(save_flag, run_time, save_dir, save_name):
    """
    根据传入的参数开启 RealSense 录像，
    - save_flag: 是否保存数据
    - run_time: 运行时间（秒），若为 0 则无限运行
    - save_dir: 保存目录
    - save_name: 文件名前缀
    """
    # 参数设置
    SAVE_INTERVAL = 60 * 60  # 每隔 60 分钟保存一个文件
    FRAME_WIDTH, FRAME_HEIGHT = 640, 480
    FPS = 15
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 编码格式

    # 检查设备并创建保存目录
    check_device()
    create_save_dir(save_flag, save_dir)
    
    # 初始化 pipeline
    pipeline = initialize_pipeline(FRAME_WIDTH, FRAME_HEIGHT, FPS)
    
    # 初始化保存变量
    start_time = time.time()      # 当前文件保存起始时间
    total_start_time = start_time # 程序整体运行起始时间
    record_start_time = start_time + 5  # 延迟5秒后开始保存数据

    file_index = 1
    out = None
    if save_flag:
        current_filename = new_filename(save_dir, save_name, file_index)
        out = cv2.VideoWriter(current_filename, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
        print("开始保存新文件：", current_filename)
    
    try:
        while True:
            # 检查整体运行时间是否达到设定值（run_time > 0 时）
            if run_time > 0 and (time.time() - total_start_time) >= run_time:
                print(f"达到设定的运行时间 {run_time} 秒，程序自动停止。")
                break
            
            # 获取 RealSense 帧
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # 转换为 Numpy 数组
            color_image = np.asanyarray(color_frame.get_data())

            # 保存视频帧（如果需要保存数据）
            if save_flag and time.time() >= record_start_time:
                out.write(color_image)

            # 显示视频帧
            cv2.imshow("RealSense", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 检查是否超出当前文件保存间隔，需要新建文件
            if save_flag:
                elapsed = time.time() - start_time
                if elapsed >= SAVE_INTERVAL:
                    out.release()
                    file_index += 1
                    current_filename = new_filename(save_dir, save_name, file_index)
                    out = cv2.VideoWriter(current_filename, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
                    print("开始保存新文件：", current_filename)
                    start_time = time.time()
    except KeyboardInterrupt:
        print("\n用户中断, 关闭 RealSense")
    except Exception as e:
        print("Error:", e)
    finally:
        if save_flag and out is not None:
            out.release()
        pipeline.stop()
        cv2.destroyAllWindows()

def main():
    # 解析命令行参数
    save_flag, run_time, save_dir, save_name = parse_args()
    # 调用录像函数，将参数传递进去
    record_realsense(False, 0, save_dir, save_name)

if __name__ == "__main__":
    main()
