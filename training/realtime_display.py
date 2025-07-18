from turtle import width
import cv2
import numpy as np
import serial
import ast
# import pyrealsense2 as rs
from collections import deque
from realtime_infer import RealTimeBlinkInfer

# ==== 配置区域 ====
ENABLE_THERMAL = True         # 是否启用热图
ENABLE_RGB = False            # 是否启用 RealSense RGB
THERMAL_PORT = "/dev/tty.SLAB_USBtoUART"  # 替换为你的串口号
THERMAL_BAUD = 921600

DISPLAY_WIDTH = 640           # 每路图像宽度
DISPLAY_HEIGHT = 480          # 每路图像高度
GRAPH_HEIGHT = 480            # 第二行图像高度（曲线图）

FPS = 15                 # 实际帧率
GRAPH_DURATION = 30      # 要显示的秒数（横轴时间）
MAX_POINTS = FPS * GRAPH_DURATION  # 最多保留的预测点数量

# ==== 假数据模拟曲线 ====
USE_FAKE_DATA = False
binary_values = []  # 改为 list，不限长度

# ==== 热图处理 ====
def read_thermal_frame(ser):
    try:
        raw = ser.readline()
        text_data = raw.decode('utf-8').strip()
        data_dict = ast.literal_eval(text_data)
        temperature_data = np.array(data_dict["data"])
        if temperature_data.size != 192:
            return None
        return temperature_data.reshape((12, 16))
    except Exception as e:
        print(f"[Thermal Error] {e}")
        return None

def process_thermal_image(frame):
    norm = ((frame - np.min(frame)) / (39 - np.min(frame))) * 255
    interp = np.repeat(norm, 40, axis=0)
    interp = np.repeat(interp, 40, axis=1)
    colored = cv2.applyColorMap(interp.astype(np.uint8), cv2.COLORMAP_JET)
    return colored

# ==== RealSense 初始化 ====
def init_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, DISPLAY_WIDTH, DISPLAY_HEIGHT, rs.format.bgr8, 15)
    pipeline.start(config)
    return pipeline

def read_rgb_frame(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    return np.asanyarray(color_frame.get_data())

# ==== 曲线图绘制 ====
def draw_prediction_graph(values, width, height):
    graph = np.zeros((height, width, 3), dtype=np.uint8)
    if len(values) < 2:
        return graph

    # 只保留最近 width 个值（自动左移）
    values = values[-width:]

    # 不足 width 时左侧补零
    if len(values) < width:
        padded = np.zeros(width)
        padded[-len(values):] = values
    else:
        padded = np.array(values)

    # 映射到高度坐标
    ys = height - (np.clip(padded, 0, 1) * height).astype(int)

    for i in range(1, width):
        color = (0, 0, 255) if padded[i] >= 0.5 else (0, 255, 0)
        cv2.line(graph, (i - 1, ys[i - 1]), (i, ys[i]), color, 2)
        
    return graph


# ==== 主函数 ====


def main():
    frame_count = 0  # 新增：帧计数器
    thermal_ser = None
    rs_pipeline = None

    if ENABLE_THERMAL:
        try:
            thermal_ser = serial.Serial(THERMAL_PORT, THERMAL_BAUD, timeout=1)
            print("[✅] Thermal serial opened")
        except Exception as e:
            print(f"[❌] Failed to open thermal serial: {e}")
            return

    if ENABLE_RGB:
        try:
            rs_pipeline = init_realsense()
            print("[✅] RealSense pipeline started")
        except Exception as e:
            print(f"[❌] Failed to init RealSense: {e}")
            return
        
    infer_engine = RealTimeBlinkInfer(
        model_path="checkpoints/sample_model.pth",
        model_name="resnet18",       
        frame_stack_size=6,
        device="cpu"
    )

    cv2.namedWindow("Blink Detection System", cv2.WINDOW_AUTOSIZE)

    while True:
        top_row_images = []

        # === Thermal Frame ===
        if ENABLE_THERMAL:
            thermal_raw = read_thermal_frame(thermal_ser)
            if thermal_raw is None:
                continue  # 跳过坏帧

            frame_count += 1
            if frame_count <= 10:
                print(f"[⏳] Skipping warm-up frame {frame_count}")
                continue  # ❗跳过前10帧

            thermal_img = process_thermal_image(thermal_raw)
            thermal_img = cv2.resize(thermal_img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        else:
            thermal_img = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
        top_row_images.append(thermal_img)

        # === RGB Frame ===
        if ENABLE_RGB:
            rgb = read_rgb_frame(rs_pipeline)
            if rgb is not None:
                rgb_resized = cv2.resize(rgb, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            else:
                rgb_resized = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
        else:
            rgb_resized = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
        top_row_images.append(rgb_resized)

        # 拼接上排图像
        top_row = np.hstack(top_row_images)

        # === 曲线图 ===
        _, _, binary = infer_engine.predict(thermal_raw)
        binary_values.append(binary)  # 用平滑后的画图
        
        
        # 加上当前状态文字
        status_text = "Close" if binary == 1 else "Open"
        text_color = (0, 0, 255) if binary == 1 else (0, 255, 0)
        cv2.putText(top_row, f"State: {status_text}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)


        # === 最终拼图 ===
        graph = draw_prediction_graph(binary_values, width=top_row.shape[1], height=GRAPH_HEIGHT)
        full_view = np.vstack([top_row, graph])
        cv2.imshow("Blink Detection System", full_view)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

    # ==== 清理 ====
    if thermal_ser is not None:
        thermal_ser.close()
    if rs_pipeline is not None:
        rs_pipeline.stop()
    cv2.destroyAllWindows()
    print("[💡] Exit cleanly")

if __name__ == "__main__":
    main()
