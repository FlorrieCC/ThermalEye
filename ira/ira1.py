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
    save_flag = True
    run_time = 0
    save_dir = "ira_data/"
    save_name = "data"

    if len(sys.argv) > 1:
        save_flag = sys.argv[1].lower() not in ['false', '0', 'no']
    if len(sys.argv) > 2:
        try:
            run_time = float(sys.argv[2])
        except ValueError:
            print("Invalid run time argument. Defaulting to unlimited run time.")
            run_time = 0
    if len(sys.argv) > 3:
        save_dir = sys.argv[3]
    if len(sys.argv) > 4:
        save_name = sys.argv[4]

    return save_flag, run_time, save_dir, save_name

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

# Overlay temperature values on the image (density adjustable)
def overlay_temperature_values(image, temperature, scale_factor=20, step=4):
    for i in range(0, temperature.shape[0], step):
        for j in range(0, temperature.shape[1], step):
            value = temperature[i, j]
            text = f"{value:.1f}"
            x, y = j * scale_factor, i * scale_factor
            cv2.putText(image, text, (x+2, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return image

def monitor_serial(port='COM3', baud_rate=921600, save_flag=True, run_time=0, save_dir='', save_name=''):
    try:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # millisecond precision
        video_path = os.path.join(save_dir, f"video_{timestamp}.mp4")
        data_path = os.path.join(save_dir, f"{save_name}_{timestamp}.pkl")

        ser = serial.Serial(port, baud_rate, timeout=1)
        print(f"Successfully opened serial port {port} at baud rate {baud_rate}")

        # Video saving parameters (can be enabled)
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fps = 10
        # frame_size = (16 * 20, 12 * 20)
        # video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

        cv2.namedWindow('IR Temperature', cv2.WINDOW_AUTOSIZE)
        all_temperature_data = []
        all_timestamp_data = []

        frame_count = 0
        start_time = time.time()
        display_fps = 0

        while True:
            if ser.in_waiting > 0:
                raw_data = ser.readline()
                try:
                    text_data = raw_data.decode('utf-8').strip()
                except UnicodeDecodeError:
                    print("Decoding failed")
                    continue

                try:
                    data_dict = ast.literal_eval(text_data)
                    temperature_data = np.array(data_dict["data"])
                    if temperature_data.size == 192:
                        Detected_temperature = temperature_data.reshape((12, 16))

                        current_time = time.time()
                        all_temperature_data.append(Detected_temperature.copy())
                        all_timestamp_data.append(datetime.now().isoformat(timespec='milliseconds'))

                        ira_interpolated = SubpageInterpolating(Detected_temperature)
                        frame_count += 1
                        if frame_count % 50 == 0:
                            elapsed = current_time - start_time
                            display_fps = frame_count / elapsed
                            print(f"[INFO] Current FPS: {display_fps:.2f}")

                        ira_norm = ((ira_interpolated - np.min(ira_interpolated)) / (39 - np.min(ira_interpolated))) * 255
                        ira_expand = np.repeat(ira_norm, 20, axis=0)
                        ira_expand = np.repeat(ira_expand, 20, axis=1)
                        ira_img_colored = cv2.applyColorMap(ira_expand.astype(np.uint8), cv2.COLORMAP_JET)

                        ira_img_colored = overlay_temperature_values(ira_img_colored, ira_interpolated)
                        cv2.putText(ira_img_colored, f"FPS: {display_fps:.1f}", (10, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        cv2.imshow('IR Temperature', ira_img_colored)
                        # To save frames, uncomment video_writer.write(...)
                    else:
                        print(f"[WARNING] Data size mismatch: {temperature_data.size}, expected 768")
                except Exception as e:
                    print(f"[ERROR] Failed to process data: {e}")

            key = cv2.waitKey(1)
            if key == 27 or key == 113:  # ESC or 'q'
                break

            if run_time > 0 and (time.time() - start_time) > run_time:
                print(f"Reached specified run time of {run_time} seconds, stopping program.")
                break

    except serial.SerialException as e:
        print(f"Serial port error: {e}")
    except KeyboardInterrupt:
        print("\nUser interrupted, closing serial port")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed")
        # if 'video_writer' in locals():
        #     video_writer.release()
        #     print(f"Video saved to: {video_path}")
        if all_temperature_data and save_flag:
            with open(data_path, 'wb') as f:
                pickle.dump({
                    "temperature": all_temperature_data,
                    "timestamp": all_timestamp_data
                }, f)
            print(f"Temperature data saved to: {data_path}")
        cv2.destroyAllWindows()

def main():
    save_flag, run_time, save_dir, save_name = parse_args()
    monitor_serial(port='/dev/ttyUSB0', baud_rate=921600, save_flag=save_flag, 
                   run_time=run_time, save_dir=save_dir, save_name=save_name)

if __name__ == "__main__":
    main()
