import torch
import os

# === Training parameters ===
SEED = 42
TRAIN_BATCH_SIZE = 512
VAL_BATCH_SIZE = 512
EPOCHS = 15
LEARNING_RATE = 1e-5
POS_WEIGHT = 4  # Positive sample weight

# === Device and logging ===
SELECTED_GPU_ID = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(SELECTED_GPU_ID)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/"
LOG_DIR = "logs/"
MODEL_NAME = "resnet18" 

# === Data paths ===
PKL_ROOT = "ira_data"  
CSV_ROOT = "gt_output"
# SUBFOLDERS = ["cold", "hot"] 
SUBFOLDERS = ["normal", "severe", "mild", "slow"]  # Subfolders for training and validation
TEST_SUBFOLDERS = ["severe"]  # Subfolders for test set
SINGLE_TEST = False  # Whether to use a single test subfolder

# === Data processing parameters ===
CENTER_SIZE = (12, 16)      # Center crop size
FRAME_STACK_SIZE = 6        # Frame stack size
WINDOW_MODE = False          # Whether to use window mode

# === Dataset and training constants ===
SEGMENT_LEN = 160  # length of each segment in frames
TRAIN_RATIO = 0.7  # ratio of training set
VAL_RATIO = 0.1    # ratio of validation set
TEST_RATIO = 0.2   # ratio of test set