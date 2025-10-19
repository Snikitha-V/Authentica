import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from dataset import SignaturePairsDataset, get_transforms
from model import SiameseCNN
from loss import ContrastiveLoss
from config import *
from dataset import get_signer_ids
import utils
import argparse
import random as pyrandom
import numpy as np

parser = argparse.ArgumentParser(description="Train Siamese signature model")
parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
parser.add_argument("--lr", type=float, default=LEARNING_RATE)
parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
if __name__ == "__main__":
    args = parser.parse_args()
else:
    args = None

# Allow overrides if running as a script
EPOCHS = args.epochs if args else NUM_EPOCHS
LR = args.lr if args else LEARNING_RATE
BATCH = args.batch_size if args else BATCH_SIZE

# Extra seeds for reproducibility
pyrandom.seed(SEED)
np.random.seed(SEED)

train_tf = get_transforms(IMG_SIZE, train=True)
eval_tf = get_transforms(IMG_SIZE, train=False)
full_dataset = SignaturePairsDataset(DATA_DIR_GENUINE, DATA_DIR_FORGED, transform=None)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

# Wrap subsets to apply appropriate transforms lazily
signer_ids = get_signer_ids(DATA_DIR_GENUINE)
import random as _random
_random.Random(SEED).shuffle(signer_ids)
split_idx = int(0.8 * len(signer_ids))
train_ids, val_ids = signer_ids[:split_idx], signer_ids[split_idx:]

train_dataset = SignaturePairsDataset(DATA_DIR_GENUINE, DATA_DIR_FORGED, transform=train_tf, allowed_ids=train_ids)
val_dataset = SignaturePairsDataset(DATA_DIR_GENUINE, DATA_DIR_FORGED, transform=eval_tf, allowed_ids=val_ids)

pin = DEVICE.type == "cuda"
# Note: we're regenerating pairs on the fly each epoch; keep loaders simple
train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, pin_memory=pin, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, pin_memory=pin, num_workers=NUM_WORKERS)

model = SiameseCNN(EMBEDDING_DIM).to(DEVICE)
criterion = ContrastiveLoss()
optimizer = Adam(model.parameters(), lr=LR)
scheduler = CosineAnnealingLR(optimizer, T_max=max(EPOCHS, 1))

from torch.amp import autocast, GradScaler
scaler = GradScaler('cuda', enabled=(USE_AMP and DEVICE.type == "cuda"))

best_val_auc = 0.0  # higher is better
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for img1, img2, labels in train_loader:
        img1 = img1.to(DEVICE, non_blocking=pin)
        img2 = img2.to(DEVICE, non_blocking=pin)
        labels = labels.to(DEVICE, non_blocking=pin)
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=(USE_AMP and DEVICE.type == "cuda")):
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, labels)
        scaler.scale(loss).backward()
        if CLIP_GRAD_NORM is not None and CLIP_GRAD_NORM > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    avg_loss = total_loss / max(1, len(train_loader))

    # Validation with metrics
    model.eval()
    val_loss = 0.0
    labels_list, distances_list = [], []
    with torch.no_grad():
        for img1, img2, labels in val_loader:
            img1 = img1.to(DEVICE, non_blocking=pin)
            img2 = img2.to(DEVICE, non_blocking=pin)
            labels = labels.to(DEVICE, non_blocking=pin)
            with autocast('cuda', enabled=(USE_AMP and DEVICE.type == "cuda")):
                out1, out2 = model(img1, img2)
                loss = criterion(out1, out2, labels)
            val_loss += loss.item()
            distances = torch.nn.functional.pairwise_distance(out1, out2)
            labels_list.extend(labels.detach().cpu().numpy().flatten())
            distances_list.extend(distances.detach().cpu().numpy())
    avg_val_loss = val_loss / max(1, len(val_loader))

    # Search optimal threshold on validation and compute AUC
    import numpy as np
    distances_arr = np.array(distances_list)
    labels_arr = np.array(labels_list).astype(int)
    if distances_arr.size > 0:
        thresholds = np.linspace(0.0, float(distances_arr.max()), 100)
        best_thresh = 0.5
        best_diff = float('inf')
        for t in thresholds:
            preds = (distances_arr < t).astype(int)
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
        val_metrics = utils.compute_metrics(labels_list, distances_arr, threshold=best_thresh)
        val_auc = val_metrics.get("AUC", 0.0)
    else:
        best_thresh = 0.5
        val_auc = 0.0
        val_metrics = {"FAR": 0.0, "FRR": 0.0}
    FAR, FRR = val_metrics.get("FAR", 0.0), val_metrics.get("FRR", 0.0)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f} | Thr*: {best_thresh:.3f} (FAR={FAR:.3f}, FRR={FRR:.3f})")

    # Save best by AUC
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered")
            break

    scheduler.step()
