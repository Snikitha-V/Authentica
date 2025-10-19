import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class SiameseCNN(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseCNN, self).__init__()
        # Start with ImageNet-pretrained weights when available, then adapt to 1 channel
        try:
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            backbone = resnet18(weights=None)
        # Average pretrained conv1 weights across RGB channels to get a good grayscale initialization
        with torch.no_grad():
            old_w = backbone.conv1.weight  # [64,3,7,7]
            new_w = old_w.mean(dim=1, keepdim=True)  # [64,1,7,7]
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            backbone.conv1.weight.copy_(new_w)
        self.backbone = backbone
        feat_dim = backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feat_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
        )

    def forward_once(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)
