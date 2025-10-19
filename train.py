import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from dataset import SignaturePairsDataset, get_transforms
from model import SiameseCNN
from loss import ContrastiveLoss
from config import *
import utils

transform = get_transforms(IMG_SIZE)
dataset = SignaturePairsDataset(DATA_DIR_GENUINE, DATA_DIR_FORGED, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

pin = DEVICE.type == "cuda"
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin, num_workers=NUM_WORKERS)

model = SiameseCNN(EMBEDDING_DIM).to(DEVICE)
criterion = ContrastiveLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingLR(optimizer, T_max=max(NUM_EPOCHS, 1))

from torch.amp import autocast, GradScaler
scaler = GradScaler('cuda', enabled=(USE_AMP and DEVICE.type == "cuda"))

best_val_metric = float("inf")  # lower is better for BER
patience_counter = 0

for epoch in range(NUM_EPOCHS):
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

    # Balanced Error Rate (BER) at 0.5 threshold for selection
    metrics = utils.compute_metrics(labels_list, distances_list, threshold=0.5)
    FAR, FRR = metrics.get("FAR", 0.0), metrics.get("FRR", 0.0)
    BER = (FAR + FRR) / 2.0
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | BER@0.5: {BER:.4f} (FAR={FAR:.3f}, FRR={FRR:.3f})")

    # Save best by BER
    if BER < best_val_metric:
        best_val_metric = BER
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered")
            break

    scheduler.step()
