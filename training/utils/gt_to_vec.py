import numpy as np

def convert_to_onehot_label(probs: np.ndarray) -> np.ndarray:
    """
    将原始概率序列转换为 one-hot 标签：
        - prob < 0.3 => [0, 0, 1] (闭眼)
        - 0.3 <= prob <= 0.7 => [0, 1, 0] (中间态)
        - prob > 0.7 => [1, 0, 0] (睁眼)
    """
    onehot_labels = []
    for p in probs:
        if p < 0.3:
            onehot_labels.append([0, 0, 1])
        elif p > 0.7:
            onehot_labels.append([1, 0, 0])
        else:
            onehot_labels.append([0, 1, 0])
    return np.array(onehot_labels, dtype=np.float32)

