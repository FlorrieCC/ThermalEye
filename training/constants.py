SEED = 42
NUM_CLASSES = 4
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-4
CHECKPOINT_PATH = "checkpoints/"
LOG_DIR = "logs/"
DEVICE = "auto"  # 强制使用 CPU
MODEL_NAME = "tcn" # 可选: "tcn", "3dcnn"
# === 基础路径 ===
PKL_ROOT = "D:/Projects/PyCharmProject/AIoT/ThermalEye-main/ira_data"
CSV_ROOT = "D:/Projects/PyCharmProject/AIoT/ThermalEye-main/gt_output"
SUBFOLDERS = ["0503", "0505", "0517", "0611"]

# === 验证集路径 ===
VAL_PKL_DIR = "D:/Projects/PyCharmProject/AIoT/ThermalEye-main/ira_data/0505"
VAL_CSV_DIR = "D:/Projects/PyCharmProject/AIoT/ThermalEye-main/gt_output/0505"

# === 中心裁剪尺寸 ===
CENTER_SIZE = (12, 16)

