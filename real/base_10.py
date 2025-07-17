'''
    v10
    Update: Batch video processing, supports real-time display and fast processing mode
'''

import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import matplotlib.ticker as ticker

'''
Read CSV
import pandas as pd
import json

df = pd.read_csv('blink_offsets_callibration_20250505_161542_482.csv')
offsets = {row['key']: json.loads(row['value']) for _, row in df.iterrows()}

print(offsets['start_offsets'])  # -> list
print(offsets['end_offsets'])    # -> list
'''

def process_video(video_path, real_time_mode=False, threshold=40, output_dir='gt_output'):
    '''
        v9
        Update: Add real-time mode, support fast video processing
    '''

    import cv2
    import cvzone
    from cvzone.FaceMeshModule import FaceMeshDetector
    from cvzone.PlotModule import LivePlot
    import numpy as np
    from datetime import datetime, timedelta
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    import json

    cap = cv2.VideoCapture(video_path)
    detector = FaceMeshDetector(maxFaces=1)
    plotY = LivePlot(640, 360, [25, 55], invert=True)

    idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
    ratioList = []
    blinkCounter = 0
    color = (255, 0, 255)

    # Calibration
    calibrationRatios = []
    calibrationFrames = 100
    calibrated = True
    adaptiveThreshold = threshold

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time_ms = 1000 / fps

    blink_start_frames = []
    blink_end_frames = []
    blink_start_offsets = []
    blink_end_offsets = []

    max_distance = None
    min_distance = None
    video_start_timestamp = None
    beijing_offset = timedelta(hours=8)

    def get_beijing_time():
        return datetime.utcnow() + beijing_offset

    eye_closed = False
    min_blink_duration_frames = int(fps * 0.1)
    closed_frames = 0

    recorded_ratios = []
    recorded_timestamps = []

    print(f"[INFO] Video FPS: {fps} FPS")
    print(f"[INFO] Total frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
    print(f"[INFO] Duration: {cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps:.2f} seconds")
    
    video_start_timestamp = get_beijing_time()

    while True:
        success, img = cap.read()
        if not success:
            break

        img, faces = detector.findFaceMesh(img, draw=False)
        if faces:
            face = faces[0]
            for id in idList:
                cv2.circle(img, face[id], 5, color, cv2.FILLED)

            leftUp, leftDown = face[159], face[23]
            leftLeft, leftRight = face[130], face[243]
            lenghtVer, _ = detector.findDistance(leftUp, leftDown)
            lenghtHor, _ = detector.findDistance(leftLeft, leftRight)
            cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
            cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)

            ratio = int((lenghtVer / lenghtHor) * 100)
            ratioList.append(ratio)
            if len(ratioList) > 3:
                ratioList.pop(0)
            ratioAvg = sum(ratioList) / len(ratioList)

            max_distance = max(max_distance or 0, lenghtVer)
            min_distance = min(min_distance or lenghtVer, lenghtVer)

            if not calibrated:
                calibrationRatios.append(ratioAvg)
                if len(calibrationRatios) >= calibrationFrames:
                    meanRatio = np.mean(calibrationRatios)
                    stdRatio = np.std(calibrationRatios)
                    adaptiveThreshold = meanRatio - stdRatio - 1
                    calibrated = True
                    print(f'[CALIBRATION DONE] Adaptive threshold: {adaptiveThreshold:.2f}')
                    cap.release()
                    cap = cv2.VideoCapture(video_path)
                    video_start_timestamp = get_beijing_time()
                    continue
            else:
                # State machine for blink detection
                if ratioAvg < adaptiveThreshold:
                    closed_frames += 1
                    if not eye_closed and closed_frames >= min_blink_duration_frames:
                        eye_closed = True
                        blinkCounter += 1
                        color = (0, 200, 0)

                        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        blink_start_frames.append(frame_index)

                        if video_start_timestamp:
                            offset = frame_index * frame_time_ms
                            blink_start_offsets.append(round(offset))
                else:
                    if eye_closed:
                        eye_closed = False
                        color = (255, 0, 255)

                        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        blink_end_frames.append(frame_index)

                        if video_start_timestamp:
                            offset = frame_index * frame_time_ms
                            blink_end_offsets.append(round(offset))

                        if max_distance > 0:
                            closure_degree = 1 - (min_distance / max_distance)
                            print(f"Frame: {frame_index}, Closure Degree: {closure_degree:.2f}")
                        max_distance = None
                        min_distance = None
                    closed_frames = 0

                # ✅ Record data
                # if video_start_timestamp and cap.get(cv2.CAP_PROP_POS_FRAMES) > calibrationFrames:
                if video_start_timestamp:
                    frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    offset = frame_index * frame_time_ms
                    recorded_ratios.append(ratioAvg)
                    recorded_timestamps.append(offset)

            imgPlot = plotY.update(ratioAvg, color)
            # ➕ Add horizontal line to indicate threshold
            if calibrated:
                cv2.line(imgPlot, (0, int(360 - adaptiveThreshold * 3.6)), (640, int(360 - adaptiveThreshold * 3.6)), (0, 0, 255), 2)
            cv2.putText(imgPlot, f'Blinks: {blinkCounter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            img = cv2.resize(img, (640, 360))
            imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
        else:
            img = cv2.resize(img, (640, 360))
            imgStack = cvzone.stackImages([img, img], 2, 1)

        if real_time_mode:
            cv2.imshow("Blink Detection", imgStack)
            if cv2.waitKey(max(1, int(frame_time_ms))) & 0xFF == ord('q'):
                break

    cap.release()
    if real_time_mode:
        cv2.destroyAllWindows()

    print("\nBlink Start Offsets (ms):", blink_start_offsets)
    print("Blink End Offsets (ms):", blink_end_offsets)
    print("Blink Start Frames:", blink_start_frames)
    print("Blink End Frames:", blink_end_frames)

    # ✅ Save data and plot
    output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    video_filename = os.path.splitext(os.path.basename(video_path))[0]

    # Save ratio data
    # csv_path = os.path.join(output_dir, 'blink_ratios.csv')
    # df = pd.DataFrame({
    #     'timestamp_ms': recorded_timestamps,
    #     'ratioAvg': recorded_ratios
    # })
    # df.to_csv(csv_path, index=False)
    # print(f"[SAVED] Ratio data saved to: {csv_path}")

    # Save frame indices
    # frame_csv_path = os.path.join(output_dir, 'blink_frames.csv')
    # df_frames = pd.DataFrame({
    #     'blink_start_frame': pd.Series(blink_start_frames),
    #     'blink_end_frame': pd.Series(blink_end_frames)
    # })
    # df_frames.to_csv(frame_csv_path, index=False)
    # print(f"[SAVED] Blink frame indices saved to: {frame_csv_path}")

    # Plot blink detection curve
    plt.figure(figsize=(12, 5))
    plt.plot(recorded_timestamps, recorded_ratios, label='Ratio Avg', color='blue')
    for start, end in zip(blink_start_offsets, blink_end_offsets):
        plt.axvspan(start, end, color='gray', alpha=0.3)
        
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5000))
    plt.xlabel('Time (ms)')
    plt.ylabel('Eye Aspect Ratio')
    plt.title('Blink Detection Curve with Highlighted Blink Periods')
    plt.grid(False)
    plt.legend()
    img_path = os.path.join(output_dir, f'blink_plot_{video_filename}.png')
    plt.savefig(img_path)
    # plt.show()
    print(f"[SAVED] Plot saved to: {img_path}")

    # ✅ Save offset values as CSV (key-value format)
    offset_csv_path = os.path.join(output_dir, f'blink_offsets_{video_filename}.csv')

    df_offset = pd.DataFrame([
        {"key": "start_offsets", "value": json.dumps(blink_start_offsets)},
        {"key": "end_offsets", "value": json.dumps(blink_end_offsets)}
    ])
    df_offset.to_csv(offset_csv_path, index=False)
    print(f"[SAVED] Blink offsets saved to: {offset_csv_path}")


if __name__ == "__main__":
    # ✅ Set processing mode: "single" or "batch"
    MODE = "single"

    # ✅ If MODE = "single", set video path
    single_video_path = "/Users/yvonne/Documents/final project/ThermalEye/real_data/0618/shy_left_hot_severe_20250619_031634_848.mp4"

    # ✅ If MODE = "batch", set folder path
    batch_folder_path = "/Users/yvonne/Documents/final project/ThermalEye/real_data/0618"

    # ✅ Enable real-time display (True = show window, False = fast process)
    enable_realtime = False
    
    # ✅ Threshold
    threshold = 37  # Adjustable threshold
    
    # ✅ Output directory
    output_dir = "gt_output/0618"

    if MODE == "single":
        print(f"\n🟢 Processing single video: {single_video_path}")
        process_video(single_video_path, real_time_mode=enable_realtime, threshold=threshold, output_dir=output_dir)

    elif MODE == "batch":
        print(f"\n🟢 Processing folder in batch mode: {batch_folder_path}")
        for filename in os.listdir(batch_folder_path):
            if filename.endswith(".mp4"):
                video_path = os.path.join(batch_folder_path, filename)
                print(f"\n👉 Processing: {video_path}")
                process_video(video_path, real_time_mode=enable_realtime, threshold=threshold, output_dir=output_dir)

    else:
        print("❌ Invalid MODE setting, use 'single' or 'batch'")
