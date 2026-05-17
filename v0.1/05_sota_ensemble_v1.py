import os
import json
import gc
import itertools
import importlib.util
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    EnsureTyped
)
from monai.data import CacheDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism


# =========================================================
# 0. 全局配置
# =========================================================

DATA_DIR = "./data/BraTS2021"
MODEL_DIR = "./model"
RESULT_DIR = "./result"

PRO_SCRIPT = "./2_pro_lstm_only.py"
SWIN_SCRIPT = "./3_swin_only.py"

SPLIT_JSON = "./splits_holdout_seed42.json"

CACHE_DIR = os.path.join(RESULT_DIR, "prob_cache_holdout_v1")

PATCH_SIZE = (128, 128, 128)
OVERLAP = 0.6

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED = 42
set_determinism(seed=SEED)

USE_TTA = True
TTA_MODE = "8flip"

SEARCH_CASE_LIMIT = 30

MODEL_NAMES = [
    "pro_lstm",
    "swin_unetr"
]

DEFAULT_THRESHOLDS = (0.45, 0.50, 0.35)  # TC, WT, ET
MIN_ET_VOXELS = 20

WEIGHT_STEP = 0.05

TC_GRID = [0.35, 0.40, 0.45, 0.50]
WT_GRID = [0.45, 0.50, 0.55]
ET_GRID = [0.25, 0.30, 0.35, 0.40, 0.45]

METRIC_SAMPLE_SIZE = 500000


# =========================================================
# 1. 动态导入已有模型脚本
# =========================================================

def import_python_file(module_name, file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到脚本: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pro_module = import_python_file("pro_lstm_module", PRO_SCRIPT)
swin_module = import_python_file("swin_module", SWIN_SCRIPT)

ConvertToMultiChannelBasedOnBratsClassesd = pro_module.ConvertToMultiChannelBasedOnBratsClassesd


# =========================================================
# 2. 数据构建
# =========================================================

def load_split():
    if not os.path.exists(SPLIT_JSON):
        raise FileNotFoundError(
            f"没有找到 {SPLIT_JSON}，请先运行: python 00_make_splits.py"
        )

    with open(SPLIT_JSON, "r", encoding="utf-8") as f:
        split = json.load(f)

    val_ids = split["val"]
    print(f"✅ 使用 split: {SPLIT_JSON}")
    print(f"✅ 验证集病例数: {len(val_ids)}")
    return val_ids


def build_case_dict(case_id):
    folder = os.path.join(DATA_DIR, case_id)

    image_paths = [
        os.path.join(folder, f"{case_id}_flair.nii.gz"),
        os.path.join(folder, f"{case_id}_t1ce.nii.gz"),
        os.path.join(folder, f"{case_id}_t1.nii.gz"),
        os.path.join(folder, f"{case_id}_t2.nii.gz")
    ]

    label_path = os.path.join(folder, f"{case_id}_seg.nii.gz")

    for p in image_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到图像文件: {p}")

    if not os.path.exists(label_path):
        raise FileNotFoundError(f"找不到标签文件: {label_path}")

    return {
        "case_id": case_id,
        "image": image_paths,
        "label": label_path
    }


def prepare_val_loader():
    val_ids = load_split()
    val_files = [build_case_dict(case_id) for case_id in val_ids]

    val_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest")
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"])
    ])

    dataset = CacheDataset(
        data=val_files,
        transform=val_transform,
        cache_num=5
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return loader


# =========================================================
# 3. 模型加载
# =========================================================

def find_weight_file(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "以下权重文件都不存在:\n" + "\n".join(candidates)
    )


def load_pro_lstm():
    print("🔧 加载 Pro-Res-Conv-LSTM...")
    model = pro_module.get_pro_lstm_model()

    weight_path = find_weight_file([
        os.path.join(MODEL_DIR, "pro_res_conv_lstm_best.pth"),
        os.path.join(MODEL_DIR, "pro_res_conv_lstm_last.pth"),
        os.path.join(MODEL_DIR, "pro_res_conv_lstm_standalone.pth")
    ])

    print(f"✅ Pro-LSTM 权重: {weight_path}")
    state = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def load_swin_unetr():
    print("🔧 加载 Swin-UNETR...")
    model = swin_module.get_swin_unetr()

    weight_path = find_weight_file([
        os.path.join(MODEL_DIR, "swin_unetr_best.pth"),
        os.path.join(MODEL_DIR, "swin_unetr_last.pth"),
        os.path.join(MODEL_DIR, "swin_unetr_standalone.pth")
    ])

    print(f"✅ Swin-UNETR 权重: {weight_path}")
    state = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def load_all_models():
    models = {}

    if "pro_lstm" in MODEL_NAMES:
        models["pro_lstm"] = load_pro_lstm()

    if "swin_unetr" in MODEL_NAMES:
        models["swin_unetr"] = load_swin_unetr()

    return models


# =========================================================
# 4. TTA 推理
# =========================================================

def get_tta_axes():
    if not USE_TTA:
        return [()]

    if TTA_MODE == "4flip":
        return [
            (),
            (2,),
            (3,),
            (4,)
        ]

    if TTA_MODE == "8flip":
        return [
            (),
            (2,),
            (3,),
            (4,),
            (2, 3),
            (2, 4),
            (3, 4),
            (2, 3, 4)
        ]

    raise ValueError(f"未知 TTA_MODE: {TTA_MODE}")


@torch.no_grad()
def predict_case_prob(model, inputs):
    """
    inputs: [1, 4, D, H, W]
    return: np.ndarray [3, D, H, W], float16
    """

    axes_list = get_tta_axes()
    prob_sum = None

    for axes in axes_list:
        if len(axes) > 0:
            x = torch.flip(inputs, dims=list(axes))
        else:
            x = inputs

        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            logits = sliding_window_inference(
                x,
                PATCH_SIZE,
                sw_batch_size=1,
                predictor=model,
                overlap=OVERLAP
            )

        if len(axes) > 0:
            logits = torch.flip(logits, dims=list(axes))

        probs = torch.sigmoid(logits).float()

        if prob_sum is None:
            prob_sum = probs
        else:
            prob_sum += probs

    probs = prob_sum / len(axes_list)
    probs_np = probs[0].detach().cpu().numpy().astype(np.float16)

    return probs_np


# =========================================================
# 5. 概率缓存
# =========================================================

def get_case_id_from_batch(batch):
    case_id = batch["case_id"]
    if isinstance(case_id, (list, tuple)):
        return case_id[0]
    return str(case_id)


def cache_path(model_name, case_id):
    return os.path.join(CACHE_DIR, f"{case_id}_{model_name}.npy")


def label_cache_path(case_id):
    return os.path.join(CACHE_DIR, f"{case_id}_label.npy")


def cache_model_probs(models, val_loader):
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("\n" + "=" * 80)
    print("阶段 1：缓存每个模型的 TTA 概率图")
    print("=" * 80)

    for batch in tqdm(val_loader, desc="[概率缓存]"):
        case_id = get_case_id_from_batch(batch)

        label_path = label_cache_path(case_id)
        if not os.path.exists(label_path):
            label_np = batch["label"][0].cpu().numpy().astype(np.uint8)
            np.save(label_path, label_np)

        inputs = batch["image"].to(DEVICE)

        for model_name, model in models.items():
            out_path = cache_path(model_name, case_id)

            if os.path.exists(out_path):
                continue

            probs_np = predict_case_prob(model, inputs)
            np.save(out_path, probs_np)

        torch.cuda.empty_cache()
        gc.collect()

    print(f"✅ 概率缓存完成: {CACHE_DIR}")


# =========================================================
# 6. 后处理与 Dice
# =========================================================

def keep_largest_components(mask, num_components=1, min_size=0):
    if not np.any(mask):
        return mask

    try:
        from scipy import ndimage as ndi
    except ImportError:
        return mask

    labeled, num = ndi.label(mask)

    if num == 0:
        return mask

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0

    valid_ids = np.where(sizes >= min_size)[0]
    valid_ids = valid_ids[valid_ids != 0]

    if valid_ids.size == 0:
        return np.zeros_like(mask, dtype=bool)

    if num_components is not None and valid_ids.size > num_components:
        valid_ids = valid_ids[np.argsort(sizes[valid_ids])[-num_components:]]

    return np.isin(labeled, valid_ids)


def build_pred_from_probs(probs, thresholds, use_cc=True):
    """
    probs: [3, D, H, W]
    channel order:
        0 = TC
        1 = WT
        2 = ET
    """

    th_tc, th_wt, th_et = thresholds

    tc = probs[0] > th_tc
    wt = probs[1] > th_wt
    et = probs[2] > th_et

    if use_cc:
        wt = keep_largest_components(wt, num_components=1, min_size=0)
        tc = keep_largest_components(tc, num_components=2, min_size=10)
        et = keep_largest_components(et, num_components=2, min_size=5)

    if et.sum() < MIN_ET_VOXELS:
        et[:] = False

    # 层级约束：ET ⊂ TC ⊂ WT
    tc = np.logical_or(tc, et)
    wt = np.logical_or(wt, tc)

    pred = np.zeros_like(probs, dtype=np.uint8)
    pred[0] = tc.astype(np.uint8)
    pred[1] = wt.astype(np.uint8)
    pred[2] = et.astype(np.uint8)

    return pred


def dice_binary(pred, gt, eps=1e-5):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    pred_sum = pred.sum()
    gt_sum = gt.sum()

    if pred_sum == 0 and gt_sum == 0:
        return 1.0

    if pred_sum == 0 or gt_sum == 0:
        return 0.0

    inter = np.logical_and(pred, gt).sum()
    return float((2.0 * inter + eps) / (pred_sum + gt_sum + eps))


def dice_channels(pred, label):
    dice_tc = dice_binary(pred[0], label[0])
    dice_wt = dice_binary(pred[1], label[1])
    dice_et = dice_binary(pred[2], label[2])
    dice_global = (dice_tc + dice_wt + dice_et) / 3.0

    return dice_global, dice_tc, dice_wt, dice_et


# =========================================================
# 7. 融合权重与阈值搜索
# =========================================================

def simplex_weights(n, step=0.05):
    """
    生成 n 个模型的权重组合，权重和为 1。
    当前两个模型时就是：
    [0,1], [0.05,0.95], ..., [1,0]
    """

    values = np.arange(0.0, 1.0 + 1e-8, step)

    if n == 1:
        yield [1.0]
        return

    if n == 2:
        for w in values:
            yield [round(float(w), 4), round(float(1.0 - w), 4)]
        return

    # 多模型情况下的递归搜索
    def rec(prefix, remaining, depth):
        if depth == n - 1:
            if 0.0 <= remaining <= 1.0:
                yield prefix + [round(float(remaining), 4)]
            return

        for v in values:
            if v <= remaining + 1e-8:
                yield from rec(prefix + [round(float(v), 4)], remaining - v, depth + 1)

    yield from rec([], 1.0, 0)


def load_prob(model_name, case_id):
    path = cache_path(model_name, case_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到缓存概率: {path}")
    return np.load(path).astype(np.float32)


def load_label(case_id):
    path = label_cache_path(case_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到缓存标签: {path}")
    return np.load(path).astype(np.uint8)


def ensemble_probs(case_id, model_names, weights):
    prob = None

    for model_name, weight in zip(model_names, weights):
        p = load_prob(model_name, case_id)

        if prob is None:
            prob = weight * p
        else:
            prob += weight * p

    return prob.astype(np.float32)


def search_best_config(val_loader):
    print("\n" + "=" * 80)
    print("阶段 2：搜索最优融合权重 + TC/WT/ET 阈值")
    print("=" * 80)

    all_case_ids = []

    for batch in val_loader:
        case_id = get_case_id_from_batch(batch)
        all_case_ids.append(case_id)

    search_case_ids = all_case_ids[:min(SEARCH_CASE_LIMIT, len(all_case_ids))]

    weight_candidates = list(simplex_weights(len(MODEL_NAMES), step=WEIGHT_STEP))

    threshold_candidates = [
        (tc, wt, et)
        for tc in TC_GRID
        for wt in WT_GRID
        for et in ET_GRID
    ]

    print(f"模型: {MODEL_NAMES}")
    print(f"权重组合数: {len(weight_candidates)}")
    print(f"阈值组合数: {len(threshold_candidates)}")
    print(f"搜索病例数: {len(search_case_ids)}")

    best_score = -1.0
    best_weights = None
    best_thresholds = None
    best_dice = None

    for weights in tqdm(weight_candidates, desc="[权重搜索]"):
        for thresholds in threshold_candidates:
            dice_tc_list = []
            dice_wt_list = []
            dice_et_list = []

            for case_id in search_case_ids:
                label = load_label(case_id)
                prob = ensemble_probs(case_id, MODEL_NAMES, weights)

                # 搜索阶段不做复杂 CC，速度更快
                pred = build_pred_from_probs(
                    prob,
                    thresholds=thresholds,
                    use_cc=False
                )

                _, dice_tc, dice_wt, dice_et = dice_channels(pred, label)

                dice_tc_list.append(dice_tc)
                dice_wt_list.append(dice_wt)
                dice_et_list.append(dice_et)

            mean_tc = float(np.mean(dice_tc_list))
            mean_wt = float(np.mean(dice_wt_list))
            mean_et = float(np.mean(dice_et_list))

            # 冲 SOTA 时建议更重视 TC 和 ET
            score = 0.4 * mean_tc + 0.2 * mean_wt + 0.4 * mean_et

            if score > best_score:
                best_score = score
                best_weights = weights
                best_thresholds = thresholds
                best_dice = {
                    "TC": mean_tc,
                    "WT": mean_wt,
                    "ET": mean_et
                }

    print("\n✅ 搜索完成")
    print(f"最佳权重: {dict(zip(MODEL_NAMES, best_weights))}")
    print(f"最佳阈值: TC={best_thresholds[0]}, WT={best_thresholds[1]}, ET={best_thresholds[2]}")
    print(f"搜索集 Dice: TC={best_dice['TC']:.4f}, WT={best_dice['WT']:.4f}, ET={best_dice['ET']:.4f}")
    print(f"Best SelectScore: {best_score:.4f}")

    config = {
        "model_names": MODEL_NAMES,
        "weights": dict(zip(MODEL_NAMES, best_weights)),
        "thresholds": {
            "TC": best_thresholds[0],
            "WT": best_thresholds[1],
            "ET": best_thresholds[2]
        },
        "search_dice": best_dice,
        "select_score": best_score,
        "search_case_limit": len(search_case_ids)
    }

    os.makedirs(RESULT_DIR, exist_ok=True)
    config_path = os.path.join(RESULT_DIR, "best_ensemble_config.json")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"✅ 最优配置已保存: {config_path}")

    return best_weights, best_thresholds


# =========================================================
# 8. 指标计算
# =========================================================

def compute_voxel_metrics(label, prob, pred):
    """
    这里的 Accuracy/Precision/Recall/AUC 是 voxel-level 指标。
    AUC 使用 prob 计算，不使用 hard mask。
    """

    y_true = label.reshape(-1).astype(np.uint8)
    y_pred = pred.reshape(-1).astype(np.uint8)
    y_prob = prob.reshape(-1).astype(np.float32)

    if len(y_true) > METRIC_SAMPLE_SIZE:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(y_true), METRIC_SAMPLE_SIZE, replace=False)

        y_true = y_true[idx]
        y_pred = y_pred[idx]
        y_prob = y_prob[idx]

    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5

    return acc, pre, rec, auc


def evaluate_final(val_loader, best_weights, best_thresholds):
    print("\n" + "=" * 80)
    print("阶段 3：全量验证集最终评估")
    print("=" * 80)

    all_acc = []
    all_pre = []
    all_rec = []
    all_auc = []

    all_dice_global = []
    all_dice_tc = []
    all_dice_wt = []
    all_dice_et = []

    for batch in tqdm(val_loader, desc="[最终评估]"):
        case_id = get_case_id_from_batch(batch)

        label = load_label(case_id)
        prob = ensemble_probs(case_id, MODEL_NAMES, best_weights)

        pred = build_pred_from_probs(
            prob,
            thresholds=best_thresholds,
            use_cc=True
        )

        dice_global, dice_tc, dice_wt, dice_et = dice_channels(pred, label)
        acc, pre, rec, auc = compute_voxel_metrics(label, prob, pred)

        all_dice_global.append(dice_global)
        all_dice_tc.append(dice_tc)
        all_dice_wt.append(dice_wt)
        all_dice_et.append(dice_et)

        all_acc.append(acc)
        all_pre.append(pre)
        all_rec.append(rec)
        all_auc.append(auc)

    final_res = {
        "Accuracy": float(np.mean(all_acc)),
        "Precision": float(np.mean(all_pre)),
        "Recall": float(np.mean(all_rec)),
        "AUC": float(np.mean(all_auc)),
        "Dice_Global": float(np.mean(all_dice_global)),
        "Dice_WT (整体)": float(np.mean(all_dice_wt)),
        "Dice_TC (核心)": float(np.mean(all_dice_tc)),
        "Dice_ET (增强)": float(np.mean(all_dice_et))
    }

    std_res = {
        "Accuracy_std": float(np.std(all_acc)),
        "Precision_std": float(np.std(all_pre)),
        "Recall_std": float(np.std(all_rec)),
        "AUC_std": float(np.std(all_auc)),
        "Dice_Global_std": float(np.std(all_dice_global)),
        "Dice_WT_std": float(np.std(all_dice_wt)),
        "Dice_TC_std": float(np.std(all_dice_tc)),
        "Dice_ET_std": float(np.std(all_dice_et))
    }

    print("\n" + "🔥" * 20)
    print("🏆 SOTA Ensemble v1.0 最终结果")
    print("🔥" * 20)

    df = pd.DataFrame({"SOTA_Ensemble_v1": final_res}).T
    print(df.round(4).to_string())

    os.makedirs(RESULT_DIR, exist_ok=True)

    out_csv = os.path.join(RESULT_DIR, "sota_ensemble_v1_results.csv")
    df.to_csv(out_csv)

    std_csv = os.path.join(RESULT_DIR, "sota_ensemble_v1_results_std.csv")
    pd.DataFrame({"SOTA_Ensemble_v1_std": std_res}).T.to_csv(std_csv)

    print(f"\n✅ 最终结果已保存: {out_csv}")
    print(f"✅ 标准差结果已保存: {std_csv}")

    return final_res


# =========================================================
# 9. 主入口
# =========================================================

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("\n" + "=" * 80)
    print("🚀 启动 Glioma SOTA Ensemble v1.0")
    print("=" * 80)

    val_loader = prepare_val_loader()

    models = load_all_models()

    cache_model_probs(models, val_loader)

    # 释放模型显存，后面只基于缓存概率搜索和评估
    for model in models.values():
        del model

    torch.cuda.empty_cache()
    gc.collect()

    best_weights, best_thresholds = search_best_config(val_loader)

    final_res = evaluate_final(
        val_loader=val_loader,
        best_weights=best_weights,
        best_thresholds=best_thresholds
    )

    return final_res


if __name__ == "__main__":
    main()