import torch
import torch.nn as nn
from config import CONTRASTIVE_MARGIN


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=CONTRASTIVE_MARGIN):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                          (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss
