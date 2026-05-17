import os
import glob
import json
import numpy as np

DATA_DIR = "./data/BraTS2021"
SEED = 42
NUM_FOLDS = 5

HOLDOUT_OUT = "./splits_holdout_seed42.json"
KFOLD_OUT = "./splits_5fold_seed42.json"


def main():
    patient_folders = sorted(glob.glob(os.path.join(DATA_DIR, "BraTS2021_*")))
    case_ids = [os.path.basename(p) for p in patient_folders]

    if len(case_ids) == 0:
        raise RuntimeError(f"没有找到病例，请检查 DATA_DIR: {DATA_DIR}")

    rng = np.random.RandomState(SEED)
    rng.shuffle(case_ids)

    # 1. 保持你当前代码的 80/20 holdout 逻辑
    split_idx = int(len(case_ids) * 0.8)
    holdout_split = {
        "seed": SEED,
        "mode": "holdout_80_20",
        "train": case_ids[:split_idx],
        "val": case_ids[split_idx:]
    }

    with open(HOLDOUT_OUT, "w", encoding="utf-8") as f:
        json.dump(holdout_split, f, indent=4, ensure_ascii=False)

    print(f"✅ 已生成 holdout split: {HOLDOUT_OUT}")
    print(f"训练集: {len(holdout_split['train'])}, 验证集: {len(holdout_split['val'])}")

    # 2. 生成 5-fold split
    folds = np.array_split(case_ids, NUM_FOLDS)

    kfold_split = {
        "seed": SEED,
        "mode": "5fold",
        "num_folds": NUM_FOLDS,
        "folds": {}
    }

    for fold_idx in range(NUM_FOLDS):
        val_ids = list(folds[fold_idx])
        train_ids = []

        for j in range(NUM_FOLDS):
            if j != fold_idx:
                train_ids.extend(list(folds[j]))

        kfold_split["folds"][f"fold_{fold_idx}"] = {
            "train": train_ids,
            "val": val_ids
        }

    with open(KFOLD_OUT, "w", encoding="utf-8") as f:
        json.dump(kfold_split, f, indent=4, ensure_ascii=False)

    print(f"✅ 已生成 5-fold split: {KFOLD_OUT}")
    for fold_idx in range(NUM_FOLDS):
        fold = kfold_split["folds"][f"fold_{fold_idx}"]
        print(f"fold_{fold_idx}: train={len(fold['train'])}, val={len(fold['val'])}")


if __name__ == "__main__":
    main()