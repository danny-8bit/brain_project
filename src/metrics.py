"""BraTS-compliant per-case metrics."""

import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def dice_brats(pred, gt):
    """BraTS 官方 Dice 规则（per region per case）"""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    p_sum, g_sum = pred.sum(), gt.sum()
    if g_sum == 0 and p_sum == 0:
        return 1.0
    if g_sum == 0 or p_sum == 0:
        return 0.0
    inter = np.logical_and(pred, gt).sum()
    return float(2.0 * inter / (p_sum + g_sum))


class BratsMetrics:
    """BraTS 三区域 (TC, WT, ET) per-case Dice 累积器。"""

    REGION_NAMES = ["tc", "wt", "et"]

    def __init__(self):
        self.reset()

    def reset(self):
        self.dice = {r: [] for r in self.REGION_NAMES}

    def __call__(self, y_pred, y):
        for p, g in zip(y_pred, y):
            p = _to_numpy(p)
            g = _to_numpy(g)
            assert p.shape == g.shape and p.shape[0] == 3, \
                f"expected (3,D,H,W), got pred={p.shape} gt={g.shape}"
            for ci, r in enumerate(self.REGION_NAMES):
                self.dice[r].append(dice_brats(p[ci], g[ci]))

    def aggregate(self):
        out = {}
        for r in self.REGION_NAMES:
            out[f"dice_{r}"] = float(np.mean(self.dice[r])) if self.dice[r] else 0.0
        out["dice_mean"] = float(np.mean([out[f"dice_{r}"] for r in self.REGION_NAMES]))
        # 为了和 trainer.py 的日志格式兼容，hd95 留 0（训练时不算 hd95）
        for r in self.REGION_NAMES:
            out[f"hd95_{r}"] = 0.0
        out["hd95_mean"] = 0.0
        return out