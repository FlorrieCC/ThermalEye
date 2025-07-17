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
    Parse command-line arguments:
    sys.argv[1] -- save_flag (controls whether to save data; passing False/0/no disables saving, default is True)
    sys.argv[2] -- run_time (execution duration in seconds; if not provided or 0, runs indefinitely)
    sys.argv[3] -- save_dir (folder to save data; defaults to "ira_data/")
    sys.argv[4] -- save_name (prefix for saved file name; defaults to "data")
    """
    # Default values
    save_flag = True
    run_time = 0   # Run indefinitely by default
    save_dir = "ira_data/"
    save_name = "data"
    
    if len(sys.argv) > 1:
        save_flag = sys.argv[1].lower() not in ['false', '0', 'no']
    if len(sys.argv) > 2:
        try:
            run_time = float(sys.argv[2])
        except ValueError:
            print("Invalid run_time argument, using default (infinite)")
            run_time = 0
    if len(sys.argv) > 3:
        save_dir = sys.argv[3]
    if len(sys.argv) > 4:
        save_name = sys.argv[4]
    
    return save_flag, run_time, save_dir, save_name

# Bilinear interpolation function
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

# Serial port monitoring main function, supports run_time to limit runtime (in seconds)
def monitor_serial(port='', baud_rate=921600, save_flag=True, run_time=0, save_dir='', save_name=''):
    try:
        # Create save folder and file names
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(save_dir, f"video_{timestamp}.mp4")
        data_path = os.path.join(save_dir, f"{save_name}_{timestamp}.pkl")

        # Initialize serial port
        ser = serial.Serial(port, baud_rate, timeout=1)
        print(f"Serial port {port} opened successfully at {baud_rate} baud.")

        # Video configuration
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10
        frame_size = (32 * 20, 24 * 20)
        # To enable video saving, uncomment the line below
        # video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

        cv2.namedWindow('IR Temperature', cv2.WINDOW_AUTOSIZE)
        all_temperature_data = []

        # FPS tracking and runtime control (shared start_time)
        frame_count = 0
        start_time = time.time()
        display_fps = 0

        while True:
            # Read from serial port
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
                    if temperature_data.size == 768:
                        Detected_temperature = temperature_data.reshape((24, 32))
                        all_temperature_data.append(Detected_temperature.copy())

                        # Apply interpolation
                        ira_interpolated = SubpageInterpolating(Detected_temperature)

                        # Update FPS
                        frame_count += 1
                        if frame_count % 50 == 0:
                            elapsed = time.time() - start_time
                            display_fps = frame_count / elapsed
                            print(f"[INFO] Current FPS: {display_fps:.2f}")

                        # Normalize, upscale, and colorize the image
                        ira_norm = ((ira_interpolated - np.min(ira_interpolated)) / (39 - np.min(ira_interpolated))) * 255
                        ira_expand = np.repeat(ira_norm, 20, axis=0)
                        ira_expand = np.repeat(ira_expand, 20, axis=1)
                        ira_img_colored = cv2.applyColorMap(ira_expand.astype(np.uint8), cv2.COLORMAP_JET)

                        # Add temperature text and FPS display
                        ira_img_colored = overlay_temperature_values(ira_img_colored, ira_interpolated)
                        cv2.putText(ira_img_colored, f"FPS: {display_fps:.1f}", (10, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        cv2.imshow('IR Temperature', ira_img_colored)
                        # If you want to save video frames, uncomment the line below
                        # video_writer.write(ira_img_colored)
                    else:
                        print(f"Data size mismatch: {temperature_data.size}, expected 768")
                except Exception as e:
                    print(f"Error processing data: {e}")

            # Check keyboard input
            key = cv2.waitKey(1)
            if key == 27 or key == 113:  # ESC or 'q' to quit
                break

            # Check if runtime limit is reached
            if run_time > 0 and (time.time() - start_time) > run_time:
                print(f"Runtime limit {run_time} seconds reached. Stopping.")
                break

    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted by user, closing serial port")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed")
        # if 'video_writer' in locals():
        #     video_writer.release()
        #     print(f"Video saved to: {video_path}")
        if all_temperature_data and save_flag:
            with open(data_path, 'wb') as f:
                pickle.dump(all_temperature_data, f)
            print(f"Temperature data saved to: {data_path}")
        cv2.destroyAllWindows()

def main():
    # Parse command-line arguments
    save_flag, run_time, save_dir, save_name = parse_args()
    # Call serial monitor function with parsed arguments
    monitor_serial(port='/dev/ttyUSB0', baud_rate=921600, save_flag=save_flag, 
                   run_time=run_time, save_dir=save_dir, save_name=save_name)

if __name__ == "__main__":
    main()
