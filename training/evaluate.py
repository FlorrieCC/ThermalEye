import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import ThermalBlinkDataset
from constants import *
from models.get_model import get_model
import numpy as np
import matplotlib.pyplot as plt
import os
from model import OptimizedBlinkClassifier 

@torch.no_grad()

def extract_segments(sequence, threshold=0.1, upper_threshold=0.9, min_len=3):
    """
    提取处于过渡区间（例如0.1~0.9）内的连续段，作为眨眼过程段。
    """
    segments = []
    start = None
    for i, val in enumerate(sequence):
        if threshold < val < upper_threshold:
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
    """
    比较预测段和 GT 段，计算 Precision, Recall, F1, 起止偏差, IoU。
    """
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

    # 2. 加载验证集
    val_dataset = ThermalBlinkDataset(
        pkl_root=PKL_ROOT,
        csv_root=CSV_ROOT,
        subfolders=SUBFOLDERS,
        val_pkl_dir=VAL_PKL_DIR,
        val_csv_dir=VAL_CSV_DIR,
        is_val=True,
        center_size=CENTER_SIZE,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    all_preds = []
    all_labels = []
    

    for batch in val_loader:
        x_seq, y_seq = batch["x"], batch["y"]  # x: [1, T, C, H, W], y: [1, T]
        x_seq = x_seq.squeeze(0)               # [T, C, H, W]
        y_seq = y_seq.squeeze(0).numpy().astype(int)       # [T]

        with torch.no_grad():
            logits = model(x_seq.unsqueeze(0)).squeeze(0).numpy()  # [T, num_classes]
            preds_cls = np.argmax(logits, axis=1)
            print("🔍 Predicted classes:", preds_cls)  # 打印整数类别预测
            print("🔍 Groundtruth classes:", y_seq)


        all_preds.extend(preds_cls.tolist())
        all_labels.extend(y_seq.tolist())
        

    # 3. 评估指标
    print("🔍 分类准确率 & 段级评估结果：")   

    # 转为 numpy array
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # 分类准确率
    acc = (all_preds == all_labels).mean()
    print(f"✅ Accuracy: {acc:.4f}")


    # 分类指标
    # 提取段
    pred_segments = extract_segments(all_preds)
    gt_segments = extract_segments(all_labels)
    
    print("Number of predicted blink segments:", len(pred_segments))
    print("Number of ground truth blink segments:", len(gt_segments))



    metrics = compute_segment_metrics(pred_segments, gt_segments)

    print("📊 段级匹配评估结果：")
    print(f"- Precision: {metrics['precision']:.4f}")
    print(f"- Recall:    {metrics['recall']:.4f}")
    print(f"- F1 Score:  {metrics['f1']:.4f}")
    print(f"- Mean IoU:  {metrics['mean_iou']:.4f}")
    print(f"- Start Offset: {metrics['mean_start_offset']:.2f} frames")
    print(f"- End Offset:   {metrics['mean_end_offset']:.2f} frames")



    # 4. 可视化预测 vs GT（前一段）
    plt.figure(figsize=(12, 4))
    plt.plot(all_labels, label="Groundtruth", color="black")
    plt.plot(all_preds, label="Predicted", color="blue")
    plt.fill_between(range(len(all_preds)), 0, 1,
                     where=np.array(all_preds) > 0.5,
                     color='gray', alpha=0.2, label='Predicted Blink')
    plt.title("Blink Probability Prediction")
    plt.xlabel("Frame Index")
    plt.ylabel("Blink Probability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("evaluate_output", exist_ok=True)
    plt.savefig("evaluate_output/blink_prediction_curve.png")
    print("📈 预测曲线已保存至 evaluate_output/blink_prediction_curve.png")

if __name__ == '__main__':
    evaluate_model(f"{CHECKPOINT_PATH}/tcn_final.pth")
