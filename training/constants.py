import torch
import os

# === Training parameters ===
SEED = 42
TRAIN_BATCH_SIZE = 512
VAL_BATCH_SIZE = 512
EPOCHS = 10
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
SUBFOLDERS = ["slow", "mild", "severe", "normal"]  # Subfolders for different blink severity levels
# === Validation set paths ===
# VAL_PKL_DIR = "/home/yvonne/ThermalEye/ira_data/0606"
# VAL_CSV_DIR = "/home/yvonne/ThermalEye/gt_output/0606"

# === Data processing parameters ===
CENTER_SIZE = (12, 16)      # Center crop size
FRAME_STACK_SIZE = 1        # Frame stack size
WINDOW_MODE = False          # Whether to use window mode

# === Dataset and training constants ===
SEGMENT_LEN = 160  # length of each segment in frames
TRAIN_RATIO = 0.7  # ratio of training set
VAL_RATIO = 0.2    # ratio of validation set
TEST_RATIO = 0.1   # ratio of test set