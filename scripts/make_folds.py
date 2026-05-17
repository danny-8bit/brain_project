"""生成 5-fold 划分文件 data/folds.json"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import KFold

from src.data.dataset import scan_brats_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data/BraTS2021_TrainingData")
    parser.add_argument("--out", default="./data/folds.json")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    cases = scan_brats_cases(args.data_root)
    print(f"Total cases: {len(cases)}")

    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    folds = {str(i): [] for i in range(args.num_folds)}

    for fold_id, (_, val_idx) in enumerate(kf.split(cases)):
        for i in val_idx:
            folds[str(fold_id)].append(cases[i])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(folds, f, indent=2)

    for k, v in folds.items():
        print(f"Fold {k}: {len(v)} cases")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()