import logging
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_dict(base, override):
    out = deepcopy(base)
    for k, v in override.items():
        out[k] = v
    return out


def get_logger(log_file=None, name="brats"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class EMA:
    """Model EMA: exponential moving average of model weights."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_model = deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)
        # 同步 buffers (e.g., BN stats)
        for ema_b, b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.data.copy_(b.data)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state):
        self.ema_model.load_state_dict(state)
