import numpy as np

def discretize_blink_labels(cont_labels, low_thr=0.3, high_thr=0.7):
    """
    cont_labels: numpy array or torch.Tensor of shape [N], 连续[0,1]标签
    返回 int 类标: 0=闭眼 (>high_thr), 1=中间态, 2=睁眼 (<low_thr)
    """

    # 如果是 tensor，先转 numpy
    x = cont_labels.cpu().numpy() if hasattr(cont_labels, "cpu") else cont_labels

    idx = np.zeros_like(x, dtype=np.int64)
    idx[x > high_thr] = 0      # 0 类: 真闭眼
    idx[x < low_thr] = 2       # 2 类: 真睁眼
    # 中间态 (low_thr ≤ x ≤ high_thr) 保持为 1
    idx[(x >= low_thr) & (x <= high_thr)] = 1

    return idx  # shape [N], dtype=int64

