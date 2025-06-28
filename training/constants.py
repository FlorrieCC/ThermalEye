SEED = 42
NUM_CLASSES = 4
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4
CHECKPOINT_PATH = "checkpoints/"
LOG_DIR = "logs/"
DEVICE = "cpu"  # 强制使用 CPU
MODEL_NAME = "transformer" 
# === 基础路径 ===
PKL_ROOT = "/Users/yvonne/Documents/final project/ThermalEye/ira_data"
CSV_ROOT = "/Users/yvonne/Documents/final project/ThermalEye/gt_output"
SUBFOLDERS = ["0503", "0517", "0606", "0611", "0618"]

# === 验证集路径 ===
VAL_PKL_DIR = "/Users/yvonne/Documents/final project/ThermalEye/ira_data/0505"
VAL_CSV_DIR = "/Users/yvonne/Documents/final project/ThermalEye/gt_output/0505"

# === 中心裁剪尺寸 ===
CENTER_SIZE = (12, 16)

SEQ_LEN = 160  # 👈 统一管理序列长度

