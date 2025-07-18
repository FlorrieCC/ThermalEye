import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, precision_recall_curve, auc
from dataset import ThermalBlinkDataset
from constants import *
from models.get_model import get_model
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import medfilt


@torch.no_grad()
def extract_blink_segments(sequence, min_len=3):
    """
    Extract blink segments from a binarized sequence.
    Args:
        sequence: Binarized sequence (0/1)
        min_len: Minimum valid segment length
    Returns:
        segments: List of closed-eye segment start and end indices
    """
    segments = []
    start = None
    for i, val in enumerate(sequence):
        if val == 1:  # Closed-eye state
            if start is None:
                start = i
        else:
            if start is not None and i - start >= min_len:
                segments.append((start, i - 1))
                start = None
    if start is not None and len(sequence) - start >= min_len:
        segments.append((start, len(sequence) - 1))
    return segments

def compute_segment_metrics(pred_segments, gt_segments, iou_threshold=0.7):
    """
    Compute segment-level Accuracy, Precision, Recall, and F1 Score.
    Args:
        pred_segments: Predicted blink segments [(start1, end1), (start2, end2), ...]
        gt_segments: Groundtruth blink segments [(start1, end1), (start2, end2), ...]
        iou_threshold: IoU threshold for segment matching
    Returns:
        metrics: Dictionary with Accuracy, Precision, Recall, F1 Score
    """
    matched_pred = set()
    matched_gt = set()

    for i, gt in enumerate(gt_segments):
        for j, pred in enumerate(pred_segments):
            iou = calculate_iou(pred, gt)
            if iou >= iou_threshold:
                matched_pred.add(j)
                matched_gt.add(i)
                break

    precision = len(matched_pred) / len(pred_segments) if pred_segments else 0
    recall = len(matched_gt) / len(gt_segments) if gt_segments else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    accuracy = len(matched_gt) / max(len(gt_segments), len(pred_segments)) if gt_segments or pred_segments else 0

    ious_matched = []

    for i, gt in enumerate(gt_segments):
        for j, pred in enumerate(pred_segments):
            iou = calculate_iou(pred, gt)
            if iou >= iou_threshold:
                matched_pred.add(j)
                matched_gt.add(i)
                ious_matched.append(iou)   # Save matched IoU
                break

    # Return updated values
    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "mean_iou": np.mean(ious_matched) if ious_matched else 0.0
    }
    
    
def calculate_iou(pred_segment, gt_segment):
    """
    Calculate IoU (Intersection over Union) between two segments.
    Args:
        pred_segment: Predicted segment as (start, end)
        gt_segment: Groundtruth segment as (start, end)
    Returns:
        iou: IoU value
    """
    if not isinstance(pred_segment, tuple) or not isinstance(gt_segment, tuple):
        raise ValueError(f"Invalid segment format: pred_segment={pred_segment}, gt_segment={gt_segment}")

    start1, end1 = pred_segment
    start2, end2 = gt_segment

    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    inter = max(0, inter_end - inter_start + 1)

    if inter == 0:
        return 0

    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union = union_end - union_start + 1

    iou = inter / union if union > 0 else 0
    return iou

def postprocess_predictions(probs, threshold=0.5, kernel_size=7, min_valid_len=5):
    """
    Postprocess prediction sequence:
    - Median filter smoothing
    - Remove segments that are too short (false blinks)
    """
    assert kernel_size % 2 == 1, "kernel_size must be odd"

    # 1. Median filtering for smoothing
    probs_smoothed = medfilt(probs, kernel_size=kernel_size)

    # 2. Binarization
    preds_bin = (probs_smoothed >= threshold).astype(int)

    # 3. Remove short segments (closed-eye segments < min_valid_len)
    processed = preds_bin.copy()
    in_segment = False
    start = 0

    for i, val in enumerate(preds_bin):
        if val == 1 and not in_segment:
            start = i
            in_segment = True
        elif val == 0 and in_segment:
            if i - start < min_valid_len:
                processed[start:i] = 0  # Remove short segments
            in_segment = False
    if in_segment and len(preds_bin) - start < min_valid_len:
        processed[start:] = 0  # Remove short tail segment

    return probs_smoothed, processed


def evaluate_model(checkpoint_path):
    # 1. Load model
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = get_model(MODEL_NAME)
    model.load_state_dict(checkpoint)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # Auto-detect device
    model.eval()

    # 2. Validation dataset
    if SINGLE_TEST:
        test_dataset = ThermalBlinkDataset(
            pkl_root=PKL_ROOT,
            csv_root=CSV_ROOT,
            subfolders=TEST_SUBFOLDERS,
            split="test",  # Specify test split
            center_size=CENTER_SIZE,
        )
    else:
        test_dataset = ThermalBlinkDataset(
            pkl_root=PKL_ROOT,
            csv_root=CSV_ROOT,
            subfolders=SUBFOLDERS,
            split="test",  # Specify test split
            center_size=CENTER_SIZE,
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,              # Load one sequence at a time
        shuffle=False,
        num_workers=0              # Single-threaded evaluation
    )

    all_preds = []
    all_labels = []
    for batch in test_loader:
        x = batch["x"].to(DEVICE)  # Ensure data is on the correct device
        y = batch["y"].to(DEVICE)
        with torch.no_grad():
            logits = model(x)  # [B, 1] or [B]
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            if probs.ndim == 0:
                probs = np.expand_dims(probs, axis=0)  # Convert to shape=(1,)
        y = y.cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(y)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_preds_smoothed, bin_preds = postprocess_predictions(
        all_preds, threshold=0.42, kernel_size=7, min_valid_len=1
    )

    # Ground truth binary labels
    bin_labels = (all_labels >= 0.45).astype(int)

    # 3. Regression evaluation
    mae = mean_absolute_error(all_labels, all_preds_smoothed)
    mse = mean_squared_error(all_labels, all_preds_smoothed)
    print("\n🔍 Regression metrics:")
    print(f"✅ MAE: {mae:.4f}")
    print(f"✅ MSE: {mse:.4f}")

    # 4. Binary classification evaluation (threshold = 0.5)

    acc = accuracy_score(bin_labels, bin_preds)
    f1 = f1_score(bin_labels, bin_preds)
    
    cm = confusion_matrix(bin_labels, bin_preds)
    recall = recall_score(bin_labels, bin_preds)
    precision = precision_score(bin_labels, bin_preds)
    pr_precision, pr_recall, _ = precision_recall_curve(bin_labels, bin_preds)
    auc_pr = auc(pr_recall, pr_precision)
    
    # Save the precision-recall curve values
    # np.save("evaluate_output/pre_shy.npy", pr_precision)
    # np.save("evaluate_output/recall_shy.npy", pr_recall)
    

    print("\n📊 Binary classification metrics:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"AUC-PR: {auc_pr:.4f}")

    # 5. Segment-level evaluation
    
    if WINDOW_MODE:
        print("\n📦 Window-level blink segment evaluation:")
        pred_segments = extract_blink_segments(bin_preds)
        gt_segments = extract_blink_segments(bin_labels)
    else:
        print("\n📦 Frame-level blink segment evaluation:")
        pred_segments = extract_blink_segments(bin_preds)
        gt_segments = extract_blink_segments(bin_labels)
    
    print("  - Predicted blink segments:", len(pred_segments))
    print("  - Groundtruth blink segments:", len(gt_segments))
    
    pred_blink_count = len(pred_segments)
    gt_blink_count = len(gt_segments)
    blink_count_error = abs(pred_blink_count - gt_blink_count)
    print(f"  - Blink count error: {blink_count_error}")
    
    metrics = compute_segment_metrics(pred_segments, gt_segments, iou_threshold=0.5)
    print(f"  - Accuracy : {metrics['accuracy']:.4f}")
    print(f"  - Recall   : {metrics['recall']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - F1 Score : {metrics['f1']:.4f}")
    print("  - Mean IoU (matched):", metrics["mean_iou"])

    # Visualization

    # ✅ font settings for matplotlib
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 16
    })
    max_frames = 5000  # Maximum number of frames to visualize
    all_labels = all_labels[:max_frames]  # Slice ground truth labels
    bin_preds = bin_preds[:max_frames]    # Slice predicted labels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Groundtruth visualization
    ax1.plot(all_labels, label="Groundtruth", color="black", linewidth=0.5)
    ax1.fill_between(range(len(all_labels)), 0, 1,
                    where=all_labels > 0.5,
                    color='red', alpha=0.2, label='Closed')
    ax1.fill_between(range(len(all_labels)), 0, 1,
                    where=all_labels <= 0.5,
                    color='green', alpha=0.1, label='Open')
    ax1.set_ylabel("Groundtruth")
    ax1.set_title("Groundtruth")
    ax1.legend(loc="upper right")
    ax1.set_xlim(0, max_frames - 1)
    ax1.set_ylim(0, 1)
    ax1.grid(False)

    # Set spine linewidth
    for spine in ax1.spines.values():
        spine.set_linewidth(1)

    # Predicted visualization
    ax2.plot(bin_preds, label="Predicted", color="blue", alpha=0.7, linewidth=0.5)
    ax2.fill_between(range(len(bin_preds)), 0, 1,
                    where=bin_preds > 0.5,
                    color='red', alpha=0.2, label='Predicted Closed')
    ax2.fill_between(range(len(bin_preds)), 0, 1,
                    where=bin_preds <= 0.5,
                    color='green', alpha=0.1, label='Predicted Open')
    ax2.set_xlabel("Window Index" if WINDOW_MODE else "Frame Index")
    ax2.set_ylabel("Predicted")
    ax2.set_title("Predicted")
    ax2.legend(loc="upper right")
    ax2.set_xlim(0, max_frames - 1)
    ax2.set_ylim(0, 1)
    ax2.grid(False)

    # Set spine linewidth
    for spine in ax2.spines.values():
        spine.set_linewidth(1)

    # Save the figure
    plt.tight_layout(pad=0.1)
    os.makedirs("evaluate_output", exist_ok=True)
    plt.savefig("evaluate_output/blink_prediction_curve.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    print("\n📈 Visualization saved to evaluate_output/blink_prediction_curve.pdf")

if __name__ == '__main__':
    evaluate_model(f"{CHECKPOINT_PATH}/sample_model.pth")