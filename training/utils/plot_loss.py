import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. 指定 CSV log 的根目录
log_dir = "/Users/freddie/Documents/course-sem2/project/code/ThermalEye/training/logs/blink"  # 跟你在 CSVLogger 里设置的 LOG_DIR/blink 对应

# 找到最新的 version
versions = [d for d in os.listdir(log_dir) if d.startswith("version_")]
latest = sorted(versions, key=lambda s: int(s.split("_")[1]))[-1]
csv_path = os.path.join(log_dir, latest, "metrics.csv")

# 2. 读取 metrics.csv
df = pd.read_csv(csv_path)

# 3. 打印列名，确认 train_loss 和 val_loss 的列名
print("Available columns:", df.columns.tolist())

# 4. 选一个横轴，可以用 epoch 也可以用行号
x = df["epoch"] if "epoch" in df.columns else df.index

# 5. 提取 loss 数据（有些 Lightning 版本会叫 train_loss_step 或 train_loss_epoch）
train_col = "train_loss"     # 或者 "train_loss_step"
val_col   = "val_loss"

train_loss = df[train_col].dropna()
val_loss   = df[val_col].dropna()

# 6. 画图
plt.figure()
plt.plot(x[:len(train_loss)], train_loss, label="Train Loss")
plt.plot(x[:len(val_loss)],   val_loss,   label="Val Loss")
plt.xlabel("Epoch" if "epoch" in df.columns else "Step")
plt.ylabel("Loss")
plt.title("Training vs. Validation Loss")
plt.legend()
plt.tight_layout()
plt.show()