import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dataset import ThermalBlinkDataset
from constants import *
from models.get_model import get_model
import numpy as np
import matplotlib.pyplot as plt
import os

@torch.no_grad()
def extract_blink_segments(sequence, threshold=0.7, min_len=3):
    """提取预测值大于 threshold 的连续闭眼段。"""
    segments = []
    start = None
    for i, val in enumerate(sequence):
        if val >= threshold:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= min_len:
                segments.append((start, i - 1))
                start = None
    if start is not None and len(sequence) - start >= min_len:
        segments.append((start, len(sequence) - 1))
    return segments

def compute_segment_metrics(pred_segments, gt_segments, iou_threshold=0.5):
    matched_pred = set()
    matched_gt = set()
    ious = []
    start_offsets = []
    end_offsets = []

    for i, gt in enumerate(gt_segments):
        for j, pred in enumerate(pred_segments):
            inter_start = max(gt[0], pred[0])
            inter_end = min(gt[1], pred[1])
            inter = max(0, inter_end - inter_start + 1)
            union = max(gt[1], pred[1]) - min(gt[0], pred[0]) + 1
            iou = inter / union
            if iou > iou_threshold:
                matched_pred.add(j)
                matched_gt.add(i)
                ious.append(iou)
                start_offsets.append(pred[0] - gt[0])
                end_offsets.append(pred[1] - gt[1])
                break

    precision = len(matched_pred) / len(pred_segments) if pred_segments else 0
    recall = len(matched_gt) / len(gt_segments) if gt_segments else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": np.mean(ious) if ious else 0,
        "mean_start_offset": np.mean(start_offsets) if start_offsets else 0,
        "mean_end_offset": np.mean(end_offsets) if end_offsets else 0,
    }

def evaluate_model(checkpoint_path):
    # 1. 加载模型
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = get_model(MODEL_NAME)
    model.load_state_dict(checkpoint)
    model.eval()

    # 2. 验证集
    val_dataset = ThermalBlinkDataset(
        pkl_root=PKL_ROOT,
        csv_root=CSV_ROOT,
        subfolders=SUBFOLDERS,
        val_pkl_dir=VAL_PKL_DIR,
        val_csv_dir=VAL_CSV_DIR,
        is_val=True,
        center_size=CENTER_SIZE,
        normalize=True,
        std_enhance=True
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    all_preds = []
    all_labels = []

    for batch in val_loader:
        x_seq, y_seq = batch["x"], batch["y"]  # [1, T, C, H, W], [1, T]
        x_seq = x_seq.squeeze(0)               # [T, C, H, W]
        y_seq = y_seq.squeeze(0).numpy()       # [T]

        with torch.no_grad():
            logits = model(x_seq.unsqueeze(0)).squeeze(0)  # [T]
            probs = torch.sigmoid(logits).numpy()          # ← sigmoid变成概率

        all_preds.extend(probs)
        all_labels.extend(y_seq)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 3. 回归评估
    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    print("\n🔍 回归评估指标：")
    print(f"✅ MAE: {mae:.4f}")
    print(f"✅ MSE: {mse:.4f}")

    # 4. 段级评估
    # 打印分布统计信息
    print(f"[DEBUG] Pred stats: min={all_preds.min():.4f}, max={all_preds.max():.4f}, mean={all_preds.mean():.4f}")
    print(
        f"[DEBUG] #>0.7: {(all_preds > 0.7).sum()} | <0.3: {(all_preds < 0.3).sum()} | in (0.3~0.7): {((all_preds > 0.3) & (all_preds < 0.7)).sum()}")

    # 段级评估（用闭眼段预测）
    pred_segments = extract_blink_segments(all_preds, threshold=0.7)
    gt_segments = extract_blink_segments(all_labels, threshold=0.7)

    print("\n📦 段级眨眼评估：")
    print("  - 预测眨眼段数量 :", len(pred_segments))
    print("  - GT眨眼段数量   :", len(gt_segments))

    metrics = compute_segment_metrics(pred_segments, gt_segments)
    print(f"  - Precision      : {metrics['precision']:.4f}")
    print(f"  - Recall         : {metrics['recall']:.4f}")
    print(f"  - F1 Score       : {metrics['f1']:.4f}")
    print(f"  - Mean IoU       : {metrics['mean_iou']:.4f}")
    print(f"  - Start Offset   : {metrics['mean_start_offset']:.2f} frames")
    print(f"  - End Offset     : {metrics['mean_end_offset']:.2f} frames")

    # 5. 可视化
    plt.figure(figsize=(12, 4))
    plt.plot(all_labels, label="Groundtruth", color="black")
    plt.plot(all_preds, label="Predicted", color="blue")
    plt.fill_between(range(len(all_preds)), 0, 1,
                     where=all_preds > 0.7,
                     color='red', alpha=0.15, label='Predicted Closed')
    plt.fill_between(range(len(all_preds)), 0, 1,
                     where=all_preds < 0.3,
                     color='green', alpha=0.15, label='Predicted Open')
    plt.title("Blink Probability Prediction")
    plt.xlabel("Frame Index")
    plt.ylabel("Predicted Blink Probability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("evaluate_output", exist_ok=True)
    plt.savefig("evaluate_output/blink_prediction_curve.png")
    print("\n📈 可视化图已保存至 evaluate_output/blink_prediction_curve.png")


if __name__ == '__main__':
    evaluate_model(f"{CHECKPOINT_PATH}/tcn_final.pth")
