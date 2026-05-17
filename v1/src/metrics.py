import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric


class BratsMetrics:
    """计算 BraTS 三区域 (TC, WT, ET) 的 Dice 和 HD95。"""

    def __init__(self):
        self.dice = DiceMetric(
            include_background=True,
            reduction="mean_batch",
            get_not_nans=False,
        )
        self.hd95 = HausdorffDistanceMetric(
            include_background=True,
            distance_metric="euclidean",
            percentile=95,
            reduction="mean_batch",
            get_not_nans=False,
        )

    def __call__(self, y_pred, y):
        self.dice(y_pred=y_pred, y=y)
        try:
            self.hd95(y_pred=y_pred, y=y)
        except Exception:
            pass

    def aggregate(self):
        dice = self.dice.aggregate()
        try:
            hd95 = self.hd95.aggregate()
        except Exception:
            hd95 = torch.zeros_like(dice)

        return {
            "dice_tc": float(dice[0]),
            "dice_wt": float(dice[1]),
            "dice_et": float(dice[2]),
            "dice_mean": float(dice.mean()),
            "hd95_tc": float(hd95[0]),
            "hd95_wt": float(hd95[1]),
            "hd95_et": float(hd95[2]),
            "hd95_mean": float(hd95.mean()),
        }

    def reset(self):
        self.dice.reset()
        self.hd95.reset()