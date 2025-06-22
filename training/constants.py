SEED = 42
NUM_CLASSES = 4
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-4
CHECKPOINT_PATH = "checkpoints/"
LOG_DIR = "logs/"
DEVICE = "cpu"  # 强制使用 CPU
MODEL_NAME = "tcn" # 可选: "tcn", "3dcnn"
# === 基础路径 ===
PKL_ROOT = "//Users/freddie/Documents/course-sem2/project/code/ThermalEye/ira_data"
CSV_ROOT = "/Users/freddie/Documents/course-sem2/project/code/ThermalEye/gt_output"
SUBFOLDERS = ["0503", "0505", "0517","0611"]

# === 验证集路径 ===
VAL_PKL_DIR = "/Users/freddie/Documents/course-sem2/project/code/ThermalEye/ira_data/0505"
VAL_CSV_DIR = "/Users/freddie/Documents/course-sem2/project/code/ThermalEye/gt_output/0505"

# === 中心裁剪尺寸 ===
CENTER_SIZE = (12, 16)

