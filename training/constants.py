import torch
import os

# === Training parameters ===
SEED = 42
TRAIN_BATCH_SIZE = 512
VAL_BATCH_SIZE = 512
EPOCHS = 5
LEARNING_RATE = 1e-5
POS_WEIGHT = 3  # Positive sample weight

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
SUBFOLDERS = ["0503", "0517", "0618", "0505", "0611", "0630"]

# === Validation set paths ===
VAL_PKL_DIR = "/home/yvonne/ThermalEye/ira_data/0606"
VAL_CSV_DIR = "/home/yvonne/ThermalEye/gt_output/0606"

# === Data processing parameters ===
CENTER_SIZE = (12, 16)      # Center crop size
FRAME_STACK_SIZE = 1        # Frame stack size
WINDOW_MODE = False          # Whether to use window mode