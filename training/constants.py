<<<<<<< Updated upstream
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
=======
import torch
import os

# === Training parameters ===
SEED = 42
TRAIN_BATCH_SIZE = 512
VAL_BATCH_SIZE = 512
EPOCHS = 8
LEARNING_RATE = 1e-5
POS_WEIGHT = 5  # Positive sample weight

# === Device and logging ===
SELECTED_GPU_ID = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(SELECTED_GPU_ID)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/"
LOG_DIR = "logs/"
MODEL_NAME = "resnet" 

# === Data paths ===
PKL_ROOT = "/home/yvonne/ThermalEye/ira_data"
CSV_ROOT = "/home/yvonne/ThermalEye/gt_output"
SUBFOLDERS = ["0503", "0517", "0618", "0505", "0611"]

# === Validation set paths ===
VAL_PKL_DIR = "/home/yvonne/ThermalEye/ira_data/0606"
VAL_CSV_DIR = "/home/yvonne/ThermalEye/gt_output/0606"
>>>>>>> Stashed changes

# === Data processing parameters ===
CENTER_SIZE = (12, 16)      # Center crop size
FRAME_STACK_SIZE = 1        # Frame stack size
WINDOW_MODE = True          # Whether to use window mode