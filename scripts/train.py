"""单个模型 + 单个 fold 训练"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.trainer import Trainer
from src.utils import load_yaml, merge_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="./configs/base.yaml")
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = merge_dict(load_yaml(args.base_config), load_yaml(args.model_config))
    cfg["fold"] = args.fold
    cfg["gpu"] = args.gpu
    if args.resume:
        cfg["resume"] = args.resume

    trainer = Trainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()