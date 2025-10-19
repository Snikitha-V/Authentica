import os
import re
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class SignaturePairsDataset(Dataset):
    def __init__(self, genuine_dir, forged_dir, transform=None, allowed_ids=None):
        self.genuine_dir = genuine_dir
        self.forged_dir = forged_dir
        self.allowed_ids = set(allowed_ids) if allowed_ids is not None else None
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

        def _extract_signer_id(filename: str) -> str:
            """Extract signer id from filename by taking the first numeric token.
            Examples:
              - original_16_21.png -> '16'
              - forgeries_10_8.png -> '10'
              - 015_05.png -> '015' (keeps as string)
            Fallback: use the prefix before '_' if no digits are found.
            """
            base = os.path.splitext(os.path.basename(filename))[0]
            m = re.search(r"(\d+)", base)
            if m:
                return m.group(1)
            parts = base.split("_")
            return parts[0] if parts else base

        def _collect(dirpath):
            items = []
            for f in os.listdir(dirpath):
                if os.path.splitext(f)[1].lower() in valid_exts:
                    sid = _extract_signer_id(f)
                    if self.allowed_ids is None or sid in self.allowed_ids:
                        items.append(os.path.join(dirpath, f))
            return items
        self.genuine_imgs = _collect(genuine_dir)
        self.forged_imgs = _collect(forged_dir)
        self.transform = transform
        # Group genuine by signer id (prefix before underscore)
        self.id_to_genuine = {}
        for p in self.genuine_imgs:
            sid = _extract_signer_id(p)
            self.id_to_genuine.setdefault(sid, []).append(p)
        # Group forged by signer id (prefix before underscore)
        self.id_to_forged = {}
        for p in self.forged_imgs:
            sid = _extract_signer_id(p)
            self.id_to_forged.setdefault(sid, []).append(p)

    def __len__(self):
        return max(len(self.genuine_imgs), len(self.forged_imgs))

    def __getitem__(self, idx):
        # 50% chance same-person genuine pair; ensure true positives from same signer id
        if random.random() > 0.5 and len(self.id_to_genuine) > 0:
            sid = random.choice(list(self.id_to_genuine.keys()))
            imgs = self.id_to_genuine[sid]
            if len(imgs) >= 2:
                img1_path, img2_path = random.sample(imgs, 2)
            else:
                img1_path = imgs[0]
                img2_path = random.choice(self.genuine_imgs)
            label = 1
        else:
            # Negative pair: genuine vs forged (prefer forged of the same signer if available)
            if len(self.id_to_genuine) > 0:
                sid = random.choice(list(self.id_to_genuine.keys()))
                img1_path = random.choice(self.id_to_genuine[sid])
                if sid in self.id_to_forged:
                    img2_path = random.choice(self.id_to_forged[sid])
                else:
                    img2_path = random.choice(self.forged_imgs)
            else:
                img1_path = random.choice(self.genuine_imgs)
                img2_path = random.choice(self.forged_imgs)
            label = 0

        img1 = Image.open(img1_path).convert("L")
        img2 = Image.open(img2_path).convert("L")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor([label], dtype=torch.float32)

def get_signer_ids(genuine_dir):
    ids = set()
    for f in os.listdir(genuine_dir):
        if os.path.splitext(f)[1].lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            base = os.path.splitext(os.path.basename(f))[0]
            m = re.search(r"(\d+)", base)
            if m:
                ids.add(m.group(1))
            else:
                parts = base.split("_")
                ids.add(parts[0] if parts else base)
    return sorted(ids)

def get_transforms(img_size, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        # Evaluation/validation: deterministic, no augmentation
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
