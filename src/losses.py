import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss


class DiceBCELoss(nn.Module):
    """Dice + BCE for region-based (multi-label sigmoid) seg."""

    def __init__(self, lambda_dice=1.0, lambda_bce=1.0):
        super().__init__()
        self.dice = DiceLoss(
            sigmoid=True,
            squared_pred=True,
            smooth_nr=0.0,
            smooth_dr=1e-5,
            batch=True,
        )
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce

    def forward(self, pred, target):
        return (
            self.lambda_dice * self.dice(pred, target)
            + self.lambda_bce * self.bce(pred, target)
        )


class DeepSupervisionLoss(nn.Module):
    """
    输入 pred 可以是：
      - 单个 tensor: 直接计算
      - list/tuple of tensors（多尺度，第 0 个是全分辨率）：加权求和
    Label 自动 downsample 到对应 pred 大小。
    """

    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, preds, target):
        if not isinstance(preds, (list, tuple)):
            return self.base_loss(preds, target)

        # 权重 1, 1/2, 1/4, 1/8 ... 然后归一化
        weights = [1.0 / (2 ** i) for i in range(len(preds))]
        s = sum(weights)
        weights = [w / s for w in weights]

        loss = 0
        for w, p in zip(weights, preds):
            if p.shape[2:] != target.shape[2:]:
                t = F.interpolate(target, size=p.shape[2:], mode="nearest")
            else:
                t = target
            loss = loss + w * self.base_loss(p, t)
        return loss