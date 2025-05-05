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

def new_filename(save_dir, save_name):
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Keep milliseconds only
    video_path = os.path.join(save_dir, f"{save_name}_{now_str}.mp4")
    txt_path = os.path.join(save_dir, f"ts_{save_name}_{now_str}.txt")
    return video_path, txt_path

def record_realsense(save_flag, run_time, save_dir, save_name):
    FRAME_WIDTH, FRAME_HEIGHT = 640, 480
    FPS = 15
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    check_device()
    create_save_dir(save_flag, save_dir)
    pipeline = initialize_pipeline(FRAME_WIDTH, FRAME_HEIGHT, FPS)

    total_start_time = time.time()
    out = None

    if save_flag:
        video_path, ts_path = new_filename(save_dir, save_name)
        out = cv2.VideoWriter(video_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
        print("开始保存新文件：", video_path)
        ts_file = open(ts_path, "w")


    try:
        while True:
            if run_time > 0 and (time.time() - total_start_time) >= run_time:
                print(f"达到设定的运行时间 {run_time} 秒，程序自动停止。")
                break

            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            if save_flag:
                current_ts = datetime.now()
                out.write(color_image)
                ts_file.write(f"{current_ts.isoformat()}\n")
                    
            cv2.imshow("RealSense", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n用户中断, 关闭 RealSense")
    except Exception as e:
        print("Error:", e)
    finally:
        if save_flag and out is not None:
            ts_file.close()

            out.release()
        pipeline.stop()
        cv2.destroyAllWindows()

def main():
    save_flag, run_time, save_dir, save_name = parse_args()
    record_realsense(save_flag, run_time, save_dir, save_name)
    # record_realsense(True, 0, "real_data/0503/", "test")

if __name__ == "__main__":
    main()
