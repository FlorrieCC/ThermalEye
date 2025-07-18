# realtime_infer.py

import torch
import numpy as np
import cv2
from collections import deque
from scipy.signal import medfilt
from models.get_model import get_model

class RealTimeBlinkInfer:
    def __init__(self, model_path, model_name, frame_stack_size=6, device="cpu", postprocess=True):
        self.device = torch.device(device)
        self.frame_stack_size = frame_stack_size
        self.use_postprocess = postprocess
        self.kernel_size = 7
        self.min_valid_len = 1
        self.threshold = 0.42
        self.prob_history = []
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))

        # ✅ 加载模型结构+权重
        self.model = get_model(model_name)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, frame):
        blurred = cv2.GaussianBlur(frame, (3, 3), sigmaX=0.5)
        norm = (blurred - np.mean(blurred)) / (np.std(blurred) + 1e-5)
        global_min, global_max = -3.0, 2.0
        clipped = np.clip(norm, global_min, global_max)
        scaled = (clipped - global_min) / (global_max - global_min)
        gamma_corrected = np.power(scaled, 0.5)
        u8 = (gamma_corrected * 255).astype(np.uint8)
        clahe_out = self.clahe.apply(u8).astype(np.float32) / 255.0
        return clahe_out

    def predict(self, frame_2d):

        proc = self.preprocess(frame_2d)  # [H, W]
        stacked = np.repeat(proc[np.newaxis, :, :], self.frame_stack_size, axis=0)  # [C, H, W]
        input_tensor = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, C, H, W]

        with torch.no_grad():
            logits = self.model(input_tensor)
            prob = torch.sigmoid(logits).item()  # 标量

        self.prob_history.append(prob)
        self.prob_history = self.prob_history[-1000:]  # 限制长度

        if not self.use_postprocess or len(self.prob_history) < self.kernel_size:
            return prob, prob, 0  # 还不能平滑，返回自身

        smoothed_probs, binary_preds = self.postprocess_predictions(self.prob_history)
        return prob, smoothed_probs[-1], binary_preds[-1]


    def postprocess_predictions(self, probs):
        probs = np.array(probs)
        probs_smoothed = medfilt(probs, kernel_size=self.kernel_size)
        preds_bin = (probs_smoothed >= self.threshold).astype(int)

        # 去除短闭眼段
        processed = preds_bin.copy()
        in_segment = False
        start = 0
        for i, val in enumerate(preds_bin):
            if val == 1 and not in_segment:
                start = i
                in_segment = True
            elif val == 0 and in_segment:
                if i - start < self.min_valid_len:
                    processed[start:i] = 0
                in_segment = False
        if in_segment and len(preds_bin) - start < self.min_valid_len:
            processed[start:] = 0

        return probs_smoothed, processed
