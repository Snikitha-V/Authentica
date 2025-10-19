import torch
from torch.utils.data import DataLoader
from dataset import SignaturePairsDataset, get_transforms
from model import SiameseCNN
from config import *
import utils
import os
import numpy as np
import random as pyrandom

# Reproducibility
pyrandom.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Deterministic eval transforms (no augmentation)
transform = get_transforms(IMG_SIZE, train=False)
test_dataset = SignaturePairsDataset(DATA_DIR_GENUINE, DATA_DIR_FORGED, transform=transform)
pin = DEVICE.type == "cuda"
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)

model = SiameseCNN(EMBEDDING_DIM)
state = torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pth"), map_location="cpu")
model_state = model.state_dict()
filtered_state = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
missing = [k for k in model_state.keys() if k not in filtered_state]
unexpected = [k for k in state.keys() if k not in filtered_state]
if missing or unexpected:
    print("Warning: partial checkpoint load. Missing keys:", missing[:5], ("..." if len(missing) > 5 else ""))
    print("Warning: skipped keys due to shape/name mismatch:", unexpected[:5], ("..." if len(unexpected) > 5 else ""))
model.load_state_dict(filtered_state, strict=False)
model = model.to(DEVICE)
model.eval()

labels_list, distances_list = [], []

with torch.no_grad():
    for img1, img2, labels in test_loader:
        img1 = img1.to(DEVICE, non_blocking=pin)
        img2 = img2.to(DEVICE, non_blocking=pin)
        out1, out2 = model(img1, img2)
        distances = torch.nn.functional.pairwise_distance(out1, out2)
        labels_list.extend(labels.numpy().flatten())
        distances_list.extend(distances.cpu().numpy())

# Optimal threshold search (balance FAR and FRR)
distances_arr = np.array(distances_list)
labels_arr = np.array(labels_list).astype(int)
if distances_arr.size == 0:
    print("No distances computed. Check dataset or model.")
else:
    thresholds = np.linspace(0.0, float(distances_arr.max()), 100)
    best_thresh = 0.0
    best_diff = float('inf')
    for t in thresholds:
        preds = (distances_arr < t).astype(int)
        # Compute FAR/FRR directly
        tn = ((labels_arr == 0) & (preds == 0)).sum()
        fp = ((labels_arr == 0) & (preds == 1)).sum()
        fn = ((labels_arr == 1) & (preds == 0)).sum()
        tp = ((labels_arr == 1) & (preds == 1)).sum()
        FAR = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        FRR = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        diff = abs(FAR - FRR)
        if diff < best_diff:
            best_diff = diff
            best_thresh = float(t)

    print(f"Optimal threshold (FAR ≈ FRR): {best_thresh:.3f}")

    # Compute metrics at optimal threshold
    metrics = utils.compute_metrics(labels_list, distances_arr, threshold=best_thresh)
    print("Evaluation metrics at optimal threshold:")
    for k, v in metrics.items():
        print(f"{k}: {v}")
