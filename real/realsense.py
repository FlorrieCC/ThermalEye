import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import sys
from datetime import datetime

def parse_args():
    save_flag = True
    run_time = 0
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
    ctx = rs.context()
    if len(ctx.query_devices()) == 0:
        print("没有检测到 RealSense 设备，程序退出。")
        sys.exit(1)

def create_save_dir(save_flag, save_dir):
    if save_flag and not os.path.exists(save_dir):
        os.makedirs(save_dir)

def initialize_pipeline(frame_width, frame_height, fps):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, frame_width, frame_height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, frame_width, frame_height, rs.format.bgr8, fps)
    pipeline.start(config)
    return pipeline

def new_filename(save_dir, save_name, file_index):
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(save_dir, f"{save_name}_{now_str}_{file_index}.mp4")

def record_realsense(save_flag, run_time, save_dir, save_name):
    SAVE_INTERVAL = 2 * 60
    FRAME_WIDTH, FRAME_HEIGHT = 640, 480
    FPS = 15
    RECORD_DELAY = 5  # ⏱ 延迟 5 秒后开始保存

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    check_device()
    create_save_dir(save_flag, save_dir)
    pipeline = initialize_pipeline(FRAME_WIDTH, FRAME_HEIGHT, FPS)

    file_index = 1
    out = None

    start_time = time.time()           # 程序开始时间
    record_start_time = start_time + RECORD_DELAY  # 保存开始时间点
    total_start_time = start_time      # 用于整体计时
    if save_flag:
     print(f"[INFO] 程序启动，等待 {RECORD_DELAY} 秒后开始保存视频数据...")

    try:
        while True:
            if run_time > 0 and (time.time() - total_start_time) >= run_time:
                print(f"[INFO] 已达到设定的运行时间 {run_time} 秒，程序自动停止。")
                break

            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            current_time = time.time()

            # 显示视频帧
            display_img = color_image.copy()
            if current_time < record_start_time:
                cv2.putText(display_img, "等待開始錄製...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow("RealSense", display_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] 用户主动退出程序。")
                break

            # 开始保存数据（延迟后）
            if save_flag and current_time >= record_start_time:
                # 如果是刚开始保存，或到达保存间隔，开启新文件
                if out is None:
                    current_filename = new_filename(save_dir, save_name, file_index)
                    out = cv2.VideoWriter(current_filename, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
                    print("开始保存新文件：", current_filename)
                    file_start_time = current_time

                # 写入当前帧
                out.write(color_image)

                # 检查是否超过保存间隔时间，切新文件
                if (current_time - file_start_time) >= SAVE_INTERVAL:
                    out.release()
                    file_index += 1
                    current_filename = new_filename(save_dir, save_name, file_index)
                    out = cv2.VideoWriter(current_filename, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
                    print("开始保存新文件：", current_filename)
                    file_start_time = current_time

    except KeyboardInterrupt:
        print("\n[INFO] 用户按下 Ctrl+C，中断录像。")
    except Exception as e:
        print("[ERROR] 程序出错：", e)
    finally:
        if out:
            out.release()
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] 设备已关闭，程序结束。")

def main():
    save_flag, run_time, save_dir, save_name = parse_args()
    record_realsense(save_flag, run_time, save_dir, save_name)

if __name__ == "__main__":
    main()
