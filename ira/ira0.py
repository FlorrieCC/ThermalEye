import serial
import numpy as np
import cv2
import os
import pickle
import time
from datetime import datetime
import ast
import sys

def parse_args():
    """
    解析命令行参数：
    sys.argv[1] -- save_flag (控制是否保存数据，传入 False/0/no 则关闭保存，默认为 True)
    sys.argv[2] -- run_time (程序运行时间，单位秒，不传入或为0则无限制运行)
    sys.argv[3] -- save_dir (保存数据文件夹，不传入则默认为 "ira_data/")
    sys.argv[4] -- save_name (保存文件名前缀，不传入则默认为 "data")
    """
    # 默认值
    save_flag = True
    run_time = 0   # 默认无限制运行
    save_dir = "ira_data/"
    save_name = "data"
    
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

# 双线性插值函数
def SubpageInterpolating(subpage):
    shape = subpage.shape
    mat = subpage.copy()
    for i in range(shape[0]):
        for j in range(shape[1]):
            if mat[i, j] > 0.0:
                continue
            num = 0
            try:
                top = mat[i-1, j]
                num += 1
            except Exception:
                top = 0.0
            try:
                down = mat[i+1, j]
                num += 1
            except Exception:
                down = 0.0
            try:
                left = mat[i, j-1]
                num += 1
            except Exception:
                left = 0.0
            try:
                right = mat[i, j+1]
                num += 1
            except Exception:
                right = 0.0
            mat[i, j] = (top + down + left + right) / num
    return mat

# 在图像上叠加温度值（可调密度）
def overlay_temperature_values(image, temperature, scale_factor=20, step=4):
    for i in range(0, temperature.shape[0], step):
        for j in range(0, temperature.shape[1], step):
            value = temperature[i, j]
            text = f"{value:.1f}"
            x, y = j * scale_factor, i * scale_factor
            cv2.putText(image, text, (x+2, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return image

# 串口监听主程序，增加了 run_time 参数控制程序运行时间（秒）
def monitor_serial(port='', baud_rate=921600, save_flag=True, run_time=0, save_dir='', save_name=''):
    try:
        # 创建保存数据的文件夹与文件名
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(save_dir, f"video_{timestamp}.mp4")
        data_path = os.path.join(save_dir, f"{save_name}_{timestamp}.pkl")

        # 初始化串口
        ser = serial.Serial(port, baud_rate, timeout=1)
        print(f"成功打开串口 {port}，波特率 {baud_rate}")

        # 视频相关参数
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10
        frame_size = (32 * 20, 24 * 20)
        # 若需要保存视频，可启用 video_writer
        # video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

        cv2.namedWindow('IR Temperature', cv2.WINDOW_AUTOSIZE)
        all_temperature_data = []

        # 帧率统计与程序运行时间记录（共用同一个 start_time）
        frame_count = 0
        start_time = time.time()
        display_fps = 0

        while True:
            # 监听串口数据
            if ser.in_waiting > 0:
                raw_data = ser.readline()
                try:
                    text_data = raw_data.decode('utf-8').strip()
                except UnicodeDecodeError:
                    print("解码失败")
                    continue

                try:
                    data_dict = ast.literal_eval(text_data)
                    temperature_data = np.array(data_dict["data"])
                    if temperature_data.size == 768:
                        Detected_temperature = temperature_data.reshape((24, 32))
                        all_temperature_data.append(Detected_temperature.copy())

                        # 数据插值
                        ira_interpolated = SubpageInterpolating(Detected_temperature)

                        # 计算帧率
                        frame_count += 1
                        if frame_count % 50 == 0:
                            elapsed = time.time() - start_time
                            display_fps = frame_count / elapsed
                            print(f"[INFO] 当前帧率: {display_fps:.2f} FPS")

                        # 对图像进行归一化、扩展、着色处理
                        ira_norm = ((ira_interpolated - np.min(ira_interpolated)) / (39 - np.min(ira_interpolated))) * 255
                        ira_expand = np.repeat(ira_norm, 20, axis=0)
                        ira_expand = np.repeat(ira_expand, 20, axis=1)
                        ira_img_colored = cv2.applyColorMap(ira_expand.astype(np.uint8), cv2.COLORMAP_JET)

                        # 在图像上添加温度数值及帧率显示文字
                        ira_img_colored = overlay_temperature_values(ira_img_colored, ira_interpolated)
                        cv2.putText(ira_img_colored, f"FPS: {display_fps:.1f}", (10, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        cv2.imshow('IR Temperature', ira_img_colored)
                        # 如需保存视频帧，可打开下面的语句
                        # video_writer.write(ira_img_colored)
                    else:
                        print(f"数据大小不匹配: {temperature_data.size}, 应为 768")
                except Exception as e:
                    print(f"数据处理出错: {e}")

            # 检测键盘输入
            key = cv2.waitKey(1)
            if key == 27 or key == 113:  # 按下 ESC 或 q 键退出
                break

            # 判断是否达到预设的运行时间（当 run_time 为正数时）
            if run_time > 0 and (time.time() - start_time) > run_time:
                print(f"达到设定的运行时间 {run_time} 秒，程序自动停止。")
                break

    except serial.SerialException as e:
        print(f"串口错误: {e}")
    except KeyboardInterrupt:
        print("\n用户中断，关闭串口")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("串口已关闭")
        # if 'video_writer' in locals():
        #     video_writer.release()
        #     print(f"视频保存至: {video_path}")
        if all_temperature_data and save_flag:
            with open(data_path, 'wb') as f:
                pickle.dump(all_temperature_data, f)
            print(f"温度数据保存至: {data_path}")
        cv2.destroyAllWindows()

def main():
    # 解析命令行参数
    save_flag, run_time, save_dir, save_name = parse_args()
    # 调用串口监听函数，将解析的参数传递进去
    monitor_serial(port='/dev/ttyUSB0', baud_rate=921600, save_flag=save_flag, 
                   run_time=run_time, save_dir=save_dir, save_name=save_name)

if __name__ == "__main__":
    main()
