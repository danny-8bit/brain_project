import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """Dice + BCE for region-based (multi-label sigmoid) seg."""

    def __init__(self, lambda_dice=1.0, lambda_bce=1.0, channel_weights=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce
        self.register_buffer(
            "channel_weights",
            torch.as_tensor(channel_weights or [1.0, 1.0, 1.0], dtype=torch.float32),
        )

    def forward(self, pred, target):
        weights = self.channel_weights.to(device=pred.device, dtype=pred.dtype)
        weights = weights / weights.mean().clamp_min(1e-6)

        probs = torch.sigmoid(pred)
        reduce_dims = (0,) + tuple(range(2, pred.ndim))
        intersection = (probs * target).sum(dim=reduce_dims)
        denominator = (probs.square() + target.square()).sum(dim=reduce_dims)
        dice_loss = 1.0 - (2.0 * intersection) / denominator.clamp_min(1e-5)
        dice_loss = (dice_loss * weights).mean()

        bce_loss = self.bce(pred, target).mean(dim=reduce_dims)
        bce_loss = (bce_loss * weights).mean()

        return self.lambda_dice * dice_loss + self.lambda_bce * bce_loss


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