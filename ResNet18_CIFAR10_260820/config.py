import torch

# ============================================================
# Dataset
# ============================================================
NUM_CLASSES = 10

BATCH_SIZE = 128
NUM_WORKERS = 4

# ============================================================
# Training
# ============================================================
EPOCHS = 100

LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# ============================================================
# Checkpoint
# ============================================================
CHECKPOINT_DIR = "./checkpoints"

LATEST_CHECKPOINT = f"{CHECKPOINT_DIR}/latest.pth"
BEST_CHECKPOINT = f"{CHECKPOINT_DIR}/best.pth"

SAVE_INTERVAL = 5

# ============================================================
# Device
# ============================================================
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")