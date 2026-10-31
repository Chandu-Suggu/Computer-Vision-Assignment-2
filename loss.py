"""
Loss functions for Track B (from scratch using torch.nn / torch.nn.functional).
Combines pixel-wise Binary Cross Entropy with Soft Dice Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BCESoftDiceLoss(nn.Module):
    """
    Combined BCE and Soft Dice Loss.
    L = BCE(logits, target) + dice_weight * DiceLoss(sigmoid(logits), target)
    """
    def __init__(self, dice_weight=1.0, eps=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, logits, targets):
        # 1. Pixel-wise Binary Cross Entropy
        bce = F.binary_cross_entropy_with_logits(logits, targets)

        # 2. Soft Dice Loss
        probs = torch.sigmoid(logits)
        # Flatten spatial dimensions per channel and batch
        p_flat = probs.view(probs.size(0), probs.size(1), -1)
        t_flat = targets.view(targets.size(0), targets.size(1), -1)

        intersection = (p_flat * t_flat).sum(dim=-1)
        cardinality = p_flat.sum(dim=-1) + t_flat.sum(dim=-1)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        dice_loss = (1.0 - dice).mean()

        return bce + self.dice_weight * dice_loss
