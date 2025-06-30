import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from dataset import ThermalBlinkDataset
from constants import *
from models.get_model import get_model
import numpy as np
import matplotlib.pyplot as plt
import os


@torch.no_grad()
def extract_blink_segments(sequence, threshold=0.5, min_len=3):
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

def extract_window_segments(labels, threshold=0.5, min_len=1):
    # labels: 窗口级概率或标签
    segments = []
    start = None
    for i, val in enumerate(labels):
        if val >= threshold:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= min_len:
                segments.append((start, i - 1))
                start = None
    if start is not None and len(labels) - start >= min_len:
        segments.append((start, len(labels) - 1))
    return segments


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
    )
<<<<<<< Updated upstream
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
=======
    val_loader = DataLoader(
    val_dataset,
    batch_size=1,              # 每次读一条完整序列
    shuffle=False,
    num_workers=0             # eval 通常单进程就够
)
>>>>>>> Stashed changes

    all_preds = []
    all_labels = []

    for batch in val_loader:
<<<<<<< Updated upstream
        x_seq, y_seq = batch["x"], batch["y"]  # [1, T, C, H, W], [1, T]
        x_seq = x_seq.squeeze(0)               # [T, C, H, W]
        y_seq = y_seq.squeeze(0).numpy()       # [T]

        with torch.no_grad():
            logits = model(x_seq.unsqueeze(0)).squeeze(0)
            probs = torch.sigmoid(logits).numpy()  # ← 在这里加 sigmoid

=======
        x = batch["x"].to(device)  # [B, C, H, W]
        y = batch["y"].to(device)  # [B]
        with torch.no_grad():
            logits = model(x)  # [B, 1] 或 [B]
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()  # [B]
        y = y.cpu().numpy()
>>>>>>> Stashed changes
        all_preds.extend(probs)
        all_labels.extend(y)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 3. 回归评估
    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    print("\n🔍 回归评估指标：")
    print(f"✅ MAE: {mae:.4f}")
    print(f"✅ MSE: {mse:.4f}")

    # 4. 二分类评估（用 0.5 阈值）
    bin_preds = (all_preds >= 0.5).astype(int)
    bin_labels = (all_labels >= 0.5).astype(int)

    acc = accuracy_score(bin_labels, bin_preds)
    f1 = f1_score(bin_labels, bin_preds)

    print("\n📊 二分类评估：")
    print(f"✅ Accuracy : {acc:.4f}")
    print(f"✅ F1 Score : {f1:.4f}")

    # 5. 段级评估
    print(f"[DEBUG] Pred stats: min={all_preds.min():.4f}, max={all_preds.max():.4f}, mean={all_preds.mean():.4f}")
    print(f"[DEBUG] >0.5: {(all_preds > 0.5).sum()} | <0.5: {(all_preds < 0.5).sum()}")
    
    if WINDOW_MODE:
        print("\n📦 窗口级眨眼段评估：")
        pred_segments = extract_window_segments(all_preds, threshold=0.5)
        gt_segments = extract_window_segments(all_labels, threshold=0.5)
    else:
        print("\n📦 帧级眨眼段评估：")
        pred_segments = extract_blink_segments(all_preds, threshold=0.5)
        gt_segments = extract_blink_segments(all_labels, threshold=0.5)
    
    print("  - 预测眨眼段数量 :", len(pred_segments))
    print("  - GT眨眼段数量   :", len(gt_segments))
    
    metrics = compute_segment_metrics(pred_segments, gt_segments)
    print(f"  - Precision      : {metrics['precision']:.4f}")
    print(f"  - Recall         : {metrics['recall']:.4f}")
    print(f"  - F1 Score       : {metrics['f1']:.4f}")
    print(f"  - Mean IoU       : {metrics['mean_iou']:.4f}")
    print(f"  - Start Offset   : {metrics['mean_start_offset']:.2f} frames")
    print(f"  - End Offset     : {metrics['mean_end_offset']:.2f} frames")

    # 6. 可视化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # 上图：Groundtruth
    ax1.plot(all_labels, label="Groundtruth", color="black")
    ax1.fill_between(range(len(all_labels)), 0, 1,
                     where=all_labels > 0.5,
                     color='red', alpha=0.2, label='Closed')
    ax1.fill_between(range(len(all_labels)), 0, 1,
                     where=all_labels <= 0.5,
                     color='green', alpha=0.1, label='Open')
    ax1.set_ylabel("Groundtruth")
    ax1.set_title("Groundtruth (window)" if WINDOW_MODE else "Groundtruth (frame)")
    ax1.legend()
    ax1.grid(True)
    
    # 下图：Predicted
    ax2.plot(all_preds, label="Predicted", color="blue", alpha=0.7)
    ax2.fill_between(range(len(all_preds)), 0, 1,
                     where=all_preds > 0.5,
                     color='red', alpha=0.2, label='Predicted Closed')
    ax2.fill_between(range(len(all_preds)), 0, 1,
                     where=all_preds <= 0.5,
                     color='green', alpha=0.1, label='Predicted Open')
    ax2.set_xlabel("Window Index" if WINDOW_MODE else "Frame Index")
    ax2.set_ylabel("Predicted")
    ax2.set_title("Predicted (window)" if WINDOW_MODE else "Predicted (frame)")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    os.makedirs("evaluate_output", exist_ok=True)
    plt.savefig("evaluate_output/blink_prediction_curve.png")
    print("\n📈 可视化图已保存至 evaluate_output/blink_prediction_curve.png")


if __name__ == '__main__':
    evaluate_model(f"{CHECKPOINT_PATH}/tcn_final.pth")