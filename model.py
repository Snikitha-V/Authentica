import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class SiameseCNN(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseCNN, self).__init__()
        backbone = resnet18(weights=None)  # random init; can switch to weights='IMAGENET1K_V1' if 3ch
        # Adapt first conv to 1 channel (average pretrained weights or re-init)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = backbone
        feat_dim = backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feat_dim, embedding_dim),
            nn.ReLU(inplace=True)
        )

    def forward_once(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)
