import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class SignaturePairsDataset(Dataset):
    def __init__(self, genuine_dir, forged_dir, transform=None):
        self.genuine_dir = genuine_dir
        self.forged_dir = forged_dir
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        self.genuine_imgs = [
            os.path.join(genuine_dir, f)
            for f in os.listdir(genuine_dir)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
        self.forged_imgs = [
            os.path.join(forged_dir, f)
            for f in os.listdir(forged_dir)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
        self.transform = transform
        # Group genuine by signer id (prefix before underscore)
        self.id_to_genuine = {}
        for p in self.genuine_imgs:
            sid = os.path.basename(p).split("_")[0]
            self.id_to_genuine.setdefault(sid, []).append(p)
        # Group forged by signer id (prefix before underscore)
        self.id_to_forged = {}
        for p in self.forged_imgs:
            sid = os.path.basename(p).split("_")[0]
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

def get_transforms(img_size):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
