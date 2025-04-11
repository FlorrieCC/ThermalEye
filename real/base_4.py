import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import numpy as np
import time
from datetime import datetime, timedelta

cap = cv2.VideoCapture('Video.mp4')
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [25, 55], invert=True)

idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
ratioList = []
blinkCounter = 0
counter = 0
color = (255, 0, 255)

# 校准阶段
calibrationRatios = []
calibrationFrames = 100  # 校准帧数
calibrated = False
adaptiveThreshold = 0

# 获取视频的帧率（FPS）
fps = cap.get(cv2.CAP_PROP_FPS)

# 时间计数器
start_time = time.time()  # 记录程序开始时间
last_blink_time = start_time  # 记录上一次眨眼的时间

# 单次眨眼时间记录
blink_start_time = None  # 记录单次眨眼的开始时间
blink_end_time = None  # 记录单次眨眼的结束时间

# 北京时间的时区偏移（UTC+8）
beijing_offset = timedelta(hours=8)

# 距离记录
max_distance = None  # 两次 end_time 之间的最大距离
min_distance = None  # 两次 end_time 之间的最小距离

def get_beijing_time():
    """获取当前北京时间（精确到毫秒）"""
    utc_now = datetime.utcnow()  # 获取当前UTC时间
    beijing_time = utc_now + beijing_offset  # 转换为北京时间
    return beijing_time

while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        blinkCounter = 0  # 重置眨眼计数器
        start_time = time.time()  # 重置时间计数器

    success, img = cap.read()
    if not success:
        break

    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img, face[id], 5, color, cv2.FILLED)

        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]
        lenghtVer, _ = detector.findDistance(leftUp, leftDown)
        lenghtHor, _ = detector.findDistance(leftLeft, leftRight)

        cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
        cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)

        ratio = int((lenghtVer / lenghtHor) * 100)
        ratioList.append(ratio)
        if len(ratioList) > 3:
            ratioList.pop(0)
        ratioAvg = sum(ratioList) / len(ratioList)

        # 更新最大距离和最小距离
        if max_distance is None or lenghtVer > max_distance:
            max_distance = lenghtVer
        if min_distance is None or lenghtVer < min_distance:
            min_distance = lenghtVer

        # 校准逻辑
        if not calibrated:
            calibrationRatios.append(ratioAvg)
            if len(calibrationRatios) >= calibrationFrames:
                meanRatio = np.mean(calibrationRatios)
                stdRatio = np.std(calibrationRatios)
                adaptiveThreshold = meanRatio - stdRatio  # 自适应阈值
                calibrated = True
                print(f'Calibration complete. Adaptive threshold: {adaptiveThreshold}')
        else:
            if ratioAvg < adaptiveThreshold and counter == 0:
                blinkCounter += 1
                color = (0, 200, 0)
                counter = 1
                last_blink_time = time.time()  # 更新上一次眨眼的时间
                blink_start_time = get_beijing_time()  # 记录单次眨眼的开始时间（北京时间）
            if counter != 0:
                counter += 1
                if counter > 10:
                    counter = 0
                    color = (255, 0, 255)
                    blink_end_time = get_beijing_time()  # 记录单次眨眼的结束时间（北京时间）
                    # 计算单次眨眼时间
                    if blink_start_time and blink_end_time:
                        blink_duration = (blink_end_time - blink_start_time).total_seconds()
                        print(f"start_time: {blink_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}, "
                              f"end_time: {blink_end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}, "
                              f"duration: {blink_duration:.3f}")
                        # 计算闭合程度
                        if max_distance is not None and min_distance is not None and max_distance > 0:
                            closure_degree = 1 - (min_distance / max_distance)
                            print(f"Closure Degree: {closure_degree:.2f}")
                        # 重置最大距离和最小距离
                        max_distance = None
                        min_distance = None

        # 计算眨眼频率（每分钟眨眼的次数）
        current_time = time.time()  # 当前时间
        elapsed_time = current_time - start_time  # 从开始到现在的总时间（秒）
        if elapsed_time > 0:
            blink_frequency = (blinkCounter / elapsed_time) * 60  # 每分钟眨眼次数
        else:
            blink_frequency = 0

        # 显示眨眼次数和频率
        cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50, 100),
                           colorR=color)
        cvzone.putTextRect(img, f'Blink Frequency: {blink_frequency:.2f} /min', (50, 150),
                           colorR=color)

        imgPlot = plotY.update(ratioAvg, color)
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
    else:
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, img], 2, 1)

    cv2.imshow("Image", imgStack)
    if cv2.waitKey(25) & 0xFF == ord('q'):  # 按 'q' 退出
        break

cap.release()
cv2.destroyAllWindows()