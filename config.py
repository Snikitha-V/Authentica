import torch
import os

# Paths
DATA_DIR_GENUINE = "data/processed/genuine"
DATA_DIR_FORGED = "data/processed/forged"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Training parameters
IMG_SIZE = (155, 220)
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
EMBEDDING_DIM = 256  # increased for better separation
PATIENCE = 5  # Early stopping patience
CONTRASTIVE_MARGIN = 2.0  # increased margin
NUM_WORKERS = 0  # Windows-safe default; increase if stable
USE_AMP = True  # enable mixed precision on CUDA
CLIP_GRAD_NORM = 1.0  # gradient clipping max-norm

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Enable cuDNN autotuner for faster convolutions when input sizes are constant
if DEVICE.type == "cuda":
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
