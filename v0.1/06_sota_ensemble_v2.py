import os
import glob
import json
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)


# =========================================================
# 0. 全局配置
# =========================================================

RESULT_DIR = "./result"

# 直接复用 v1 缓存
CACHE_DIR = "./result/prob_cache_holdout_v1"

# 如果你使用的是 holdout split
SPLIT_JSON = "./splits_holdout_seed42.json"

# v2 会自动检查哪些模型缓存存在
MODEL_CANDIDATES = [
    "pro_lstm",
    "swin_unetr",
    "nnunet",
    "mednext",
    "umamba"
]

# 如果你想强制只用部分模型，可以写成：
# FORCE_MODEL_NAMES = ["pro_lstm", "swin_unetr"]
FORCE_MODEL_NAMES = None

# 搜索用多少个病例，越大越准，越慢
SEARCH_CASE_LIMIT = 30

# 如果只想 debug，可以设为 5；正式评估设为 None
DEBUG_CASE_LIMIT = None

# 两模型时可以细一点；多模型时建议 0.1，否则组合太多
WEIGHT_STEP_2_MODELS = 0.02
WEIGHT_STEP_MULTI_MODELS = 0.10

# 默认阈值
DEFAULT_THRESHOLDS = {
    "TC": 0.45,
    "WT": 0.50,
    "ET": 0.35
}

# 阈值搜索空间
TC_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
WT_GRID = [0.40, 0.45, 0.50, 0.55, 0.60]
ET_GRID = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

# ET 最小体积搜索空间
MIN_ET_GRID = [0, 5, 10, 20, 30, 50, 80, 100]

# 最终是否做连通域后处理
USE_CONNECTED_COMPONENT = True

# 指标采样，避免 voxel-level AUC 太慢
METRIC_SAMPLE_SIZE = 500000

SEED = 42


# =========================================================
# 1. 基础路径工具
# =========================================================

def label_cache_path(case_id):
    return os.path.join(CACHE_DIR, f"{case_id}_label.npy")


def prob_cache_path(case_id, model_name):
    return os.path.join(CACHE_DIR, f"{case_id}_{model_name}.npy")


def load_label(case_id):
    path = label_cache_path(case_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到标签缓存: {path}")

    arr = np.load(path)

    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.shape[0] != 3:
        raise ValueError(f"{path} shape 异常，应为 [3,D,H,W]，实际为 {arr.shape}")

    return arr.astype(np.uint8)


def load_prob(case_id, model_name):
    path = prob_cache_path(case_id, model_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到概率缓存: {path}")

    arr = np.load(path)

    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.shape[0] != 3:
        raise ValueError(f"{path} shape 异常，应为 [3,D,H,W]，实际为 {arr.shape}")

    return arr.astype(np.float32)


def get_case_ids_from_cache():
    label_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*_label.npy")))

    if len(label_files) == 0:
        raise RuntimeError(
            f"缓存目录中没有找到 *_label.npy，请先确认 v1 是否已经生成缓存: {CACHE_DIR}"
        )

    case_ids = [
        os.path.basename(p).replace("_label.npy", "")
        for p in label_files
    ]

    return sorted(case_ids)


def load_case_ids():
    cached_case_ids = set(get_case_ids_from_cache())

    if os.path.exists(SPLIT_JSON):
        with open(SPLIT_JSON, "r", encoding="utf-8") as f:
            split = json.load(f)

        if "val" in split:
            case_ids = [c for c in split["val"] if c in cached_case_ids]
        else:
            case_ids = sorted(list(cached_case_ids))
    else:
        case_ids = sorted(list(cached_case_ids))

    if DEBUG_CASE_LIMIT is not None:
        case_ids = case_ids[:DEBUG_CASE_LIMIT]

    if len(case_ids) == 0:
        raise RuntimeError("没有可用病例，请检查缓存和 split 文件。")

    print(f"✅ 可用病例数: {len(case_ids)}")
    return case_ids


def discover_models(case_ids):
    if FORCE_MODEL_NAMES is not None:
        model_names = FORCE_MODEL_NAMES
    else:
        model_names = []

        for model_name in MODEL_CANDIDATES:
            total = len(case_ids)
            count = sum(
                os.path.exists(prob_cache_path(case_id, model_name))
                for case_id in case_ids
            )

            if count == total:
                model_names.append(model_name)
                print(f"✅ 检测到完整模型缓存: {model_name} ({count}/{total})")
            elif count > 0:
                print(f"⚠️ 模型 {model_name} 缓存不完整: {count}/{total}，本次跳过")

    if len(model_names) == 0:
        raise RuntimeError("没有发现任何完整模型概率缓存。")

    print(f"\n🔥 本次参与 v2 融合的模型: {model_names}")
    return model_names


# =========================================================
# 2. Dice 与后处理
# =========================================================

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


def build_pred_from_probs(
    probs,
    thresholds,
    min_et_voxels,
    use_cc=True
):
    """
    probs shape:
        [3, D, H, W]

    channel:
        0 = TC
        1 = WT
        2 = ET
    """

    th_tc = thresholds["TC"]
    th_wt = thresholds["WT"]
    th_et = thresholds["ET"]

    tc = probs[0] > th_tc
    wt = probs[1] > th_wt
    et = probs[2] > th_et

    if use_cc:
        wt = keep_largest_components(
            wt,
            num_components=1,
            min_size=0
        )

        tc = keep_largest_components(
            tc,
            num_components=2,
            min_size=10
        )

        et = keep_largest_components(
            et,
            num_components=2,
            min_size=5
        )

    if et.sum() < min_et_voxels:
        et[:] = False

    # 层级约束：ET ⊂ TC ⊂ WT
    tc = np.logical_or(tc, et)
    wt = np.logical_or(wt, tc)

    pred = np.zeros_like(probs, dtype=np.uint8)
    pred[0] = tc.astype(np.uint8)
    pred[1] = wt.astype(np.uint8)
    pred[2] = et.astype(np.uint8)

    return pred


# =========================================================
# 3. 权重生成与概率融合
# =========================================================

def simplex_weights(n, step):
    values = np.arange(0.0, 1.0 + 1e-8, step)

    if n == 1:
        yield [1.0]
        return

    if n == 2:
        for w in values:
            w = round(float(w), 4)
            yield [w, round(1.0 - w, 4)]
        return

    def rec(prefix, remain, depth):
        if depth == n - 1:
            if -1e-8 <= remain <= 1.0 + 1e-8:
                yield prefix + [round(float(remain), 4)]
            return

        for v in values:
            if v <= remain + 1e-8:
                yield from rec(
                    prefix + [round(float(v), 4)],
                    remain - v,
                    depth + 1
                )

    yield from rec([], 1.0, 0)


def get_weight_step(num_models):
    if num_models <= 2:
        return WEIGHT_STEP_2_MODELS
    return WEIGHT_STEP_MULTI_MODELS


def load_search_bank(case_ids, model_names):
    print("\n" + "=" * 80)
    print("阶段 1：加载搜索病例到内存")
    print("=" * 80)

    bank = []

    for case_id in tqdm(case_ids, desc="[加载 search bank]"):
        item = {
            "case_id": case_id,
            "label": load_label(case_id).astype(np.uint8),
            "probs": {}
        }

        for model_name in model_names:
            item["probs"][model_name] = load_prob(case_id, model_name).astype(np.float16)

        bank.append(item)

    return bank


def ensemble_prob_from_loaded(item, model_names, channel_weights):
    """
    channel_weights:
        {
            "TC": [w_model1, w_model2, ...],
            "WT": [...],
            "ET": [...]
        }
    """

    shape = item["label"].shape
    out = np.zeros(shape, dtype=np.float32)

    channel_keys = ["TC", "WT", "ET"]

    for ch_idx, ch_name in enumerate(channel_keys):
        weights = channel_weights[ch_name]

        for model_idx, model_name in enumerate(model_names):
            out[ch_idx] += (
                float(weights[model_idx])
                * item["probs"][model_name][ch_idx].astype(np.float32)
            )

    return out


def ensemble_prob_from_cache(case_id, model_names, channel_weights):
    first_prob = load_prob(case_id, model_names[0])
    out = np.zeros_like(first_prob, dtype=np.float32)

    channel_keys = ["TC", "WT", "ET"]

    for ch_idx, ch_name in enumerate(channel_keys):
        weights = channel_weights[ch_name]

        for model_idx, model_name in enumerate(model_names):
            p = load_prob(case_id, model_name)
            out[ch_idx] += float(weights[model_idx]) * p[ch_idx]

    return out


# =========================================================
# 4. v2 搜索：每个通道单独找模型权重
# =========================================================

def optimize_channel_weights(bank, model_names, channel_idx, channel_name, threshold):
    num_models = len(model_names)
    step = get_weight_step(num_models)
    candidates = list(simplex_weights(num_models, step))

    best_weights = None
    best_dice = -1.0

    for weights in tqdm(candidates, desc=f"[搜索 {channel_name} 通道权重]", leave=False):
        dices = []

        for item in bank:
            prob = np.zeros_like(item["label"][channel_idx], dtype=np.float32)

            for model_idx, model_name in enumerate(model_names):
                prob += (
                    float(weights[model_idx])
                    * item["probs"][model_name][channel_idx].astype(np.float32)
                )

            pred = prob > threshold
            gt = item["label"][channel_idx]

            dices.append(dice_binary(pred, gt))

        mean_dice = float(np.mean(dices))

        if mean_dice > best_dice:
            best_dice = mean_dice
            best_weights = weights

    print(
        f"✅ {channel_name} 最优模型权重: "
        f"{dict(zip(model_names, best_weights))} | Dice={best_dice:.4f}"
    )

    return best_weights, best_dice


def search_channel_weights(bank, model_names):
    print("\n" + "=" * 80)
    print("阶段 2：通道级模型权重搜索")
    print("=" * 80)

    tc_weights, tc_dice = optimize_channel_weights(
        bank=bank,
        model_names=model_names,
        channel_idx=0,
        channel_name="TC",
        threshold=DEFAULT_THRESHOLDS["TC"]
    )

    wt_weights, wt_dice = optimize_channel_weights(
        bank=bank,
        model_names=model_names,
        channel_idx=1,
        channel_name="WT",
        threshold=DEFAULT_THRESHOLDS["WT"]
    )

    et_weights, et_dice = optimize_channel_weights(
        bank=bank,
        model_names=model_names,
        channel_idx=2,
        channel_name="ET",
        threshold=DEFAULT_THRESHOLDS["ET"]
    )

    channel_weights = {
        "TC": tc_weights,
        "WT": wt_weights,
        "ET": et_weights
    }

    search_dice = {
        "TC": tc_dice,
        "WT": wt_dice,
        "ET": et_dice
    }

    return channel_weights, search_dice


# =========================================================
# 5. v2 搜索：阈值 + ET 最小体积
# =========================================================

def search_thresholds_and_et(bank, model_names, channel_weights):
    print("\n" + "=" * 80)
    print("阶段 3：搜索 TC/WT/ET 阈值 + ET 最小体积")
    print("=" * 80)

    # 先把搜索集 ensemble probability 预计算出来
    ens_bank = []

    for item in tqdm(bank, desc="[预计算 ensemble probability]"):
        ens_prob = ensemble_prob_from_loaded(
            item=item,
            model_names=model_names,
            channel_weights=channel_weights
        )

        ens_bank.append({
            "case_id": item["case_id"],
            "label": item["label"],
            "prob": ens_prob
        })

    best_score = -1.0
    best_thresholds = None
    best_min_et = None
    best_dice = None

    candidates = [
        (tc, wt, et, min_et)
        for tc in TC_GRID
        for wt in WT_GRID
        for et in ET_GRID
        for min_et in MIN_ET_GRID
    ]

    print(f"阈值组合数量: {len(candidates)}")

    for tc, wt, et, min_et in tqdm(candidates, desc="[阈值/ET搜索]"):
        thresholds = {
            "TC": tc,
            "WT": wt,
            "ET": et
        }

        dice_tc_list = []
        dice_wt_list = []
        dice_et_list = []

        for item in ens_bank:
            pred = build_pred_from_probs(
                probs=item["prob"],
                thresholds=thresholds,
                min_et_voxels=min_et,
                use_cc=False
            )

            _, dice_tc, dice_wt, dice_et = dice_channels(
                pred=pred,
                label=item["label"]
            )

            dice_tc_list.append(dice_tc)
            dice_wt_list.append(dice_wt)
            dice_et_list.append(dice_et)

        mean_tc = float(np.mean(dice_tc_list))
        mean_wt = float(np.mean(dice_wt_list))
        mean_et = float(np.mean(dice_et_list))

        # 比赛/论文里更重视 TC 和 ET
        score = 0.4 * mean_tc + 0.2 * mean_wt + 0.4 * mean_et

        if score > best_score:
            best_score = score
            best_thresholds = thresholds
            best_min_et = min_et
            best_dice = {
                "TC": mean_tc,
                "WT": mean_wt,
                "ET": mean_et
            }

    print("\n✅ 阈值搜索完成")
    print(f"最佳 thresholds: {best_thresholds}")
    print(f"最佳 min_et_voxels: {best_min_et}")
    print(
        f"搜索集 Dice: "
        f"TC={best_dice['TC']:.4f}, "
        f"WT={best_dice['WT']:.4f}, "
        f"ET={best_dice['ET']:.4f}"
    )
    print(f"Best SelectScore={best_score:.4f}")

    return best_thresholds, best_min_et, best_dice, best_score


# =========================================================
# 6. 指标计算
# =========================================================

def compute_voxel_metrics(label, prob, pred):
    """
    当前 Accuracy / Precision / Recall / AUC 是 voxel-level。
    AUC 使用 probability，不使用 hard mask。
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


def evaluate_final(
    case_ids,
    model_names,
    channel_weights,
    thresholds,
    min_et_voxels
):
    print("\n" + "=" * 80)
    print("阶段 4：v2 全量最终评估")
    print("=" * 80)

    records = []

    for case_id in tqdm(case_ids, desc="[v2 final eval]"):
        label = load_label(case_id)
        prob = ensemble_prob_from_cache(
            case_id=case_id,
            model_names=model_names,
            channel_weights=channel_weights
        )

        pred = build_pred_from_probs(
            probs=prob,
            thresholds=thresholds,
            min_et_voxels=min_et_voxels,
            use_cc=USE_CONNECTED_COMPONENT
        )

        dice_global, dice_tc, dice_wt, dice_et = dice_channels(
            pred=pred,
            label=label
        )

        acc, pre, rec, auc = compute_voxel_metrics(
            label=label,
            prob=prob,
            pred=pred
        )

        records.append({
            "case_id": case_id,
            "Accuracy": acc,
            "Precision": pre,
            "Recall": rec,
            "AUC": auc,
            "Dice_Global": dice_global,
            "Dice_TC": dice_tc,
            "Dice_WT": dice_wt,
            "Dice_ET": dice_et
        })

    df_case = pd.DataFrame(records)

    final_res = {
        "Accuracy": float(df_case["Accuracy"].mean()),
        "Precision": float(df_case["Precision"].mean()),
        "Recall": float(df_case["Recall"].mean()),
        "AUC": float(df_case["AUC"].mean()),
        "Dice_Global": float(df_case["Dice_Global"].mean()),
        "Dice_WT (整体)": float(df_case["Dice_WT"].mean()),
        "Dice_TC (核心)": float(df_case["Dice_TC"].mean()),
        "Dice_ET (增强)": float(df_case["Dice_ET"].mean())
    }

    std_res = {
        "Accuracy_std": float(df_case["Accuracy"].std()),
        "Precision_std": float(df_case["Precision"].std()),
        "Recall_std": float(df_case["Recall"].std()),
        "AUC_std": float(df_case["AUC"].std()),
        "Dice_Global_std": float(df_case["Dice_Global"].std()),
        "Dice_WT_std": float(df_case["Dice_WT"].std()),
        "Dice_TC_std": float(df_case["Dice_TC"].std()),
        "Dice_ET_std": float(df_case["Dice_ET"].std())
    }

    os.makedirs(RESULT_DIR, exist_ok=True)

    case_csv = os.path.join(RESULT_DIR, "sota_ensemble_v2_per_case.csv")
    summary_csv = os.path.join(RESULT_DIR, "sota_ensemble_v2_summary.csv")
    std_csv = os.path.join(RESULT_DIR, "sota_ensemble_v2_std.csv")

    df_case.to_csv(case_csv, index=False)

    df_summary = pd.DataFrame({"SOTA_Ensemble_v2": final_res}).T
    df_summary.to_csv(summary_csv)

    pd.DataFrame({"SOTA_Ensemble_v2_std": std_res}).T.to_csv(std_csv)

    print("\n" + "🔥" * 20)
    print("🏆 SOTA Ensemble v2 最终结果")
    print("🔥" * 20)
    print(df_summary.round(4).to_string())

    print(f"\n✅ per-case 结果已保存: {case_csv}")
    print(f"✅ summary 结果已保存: {summary_csv}")
    print(f"✅ std 结果已保存: {std_csv}")

    return final_res, df_case


# =========================================================
# 7. 保存配置
# =========================================================

def save_config(
    model_names,
    channel_weights,
    search_weight_dice,
    thresholds,
    min_et_voxels,
    threshold_search_dice,
    select_score
):
    config = {
        "version": "SOTA Ensemble v2",
        "cache_dir": CACHE_DIR,
        "model_names": model_names,
        "channel_weights": {
            "TC": dict(zip(model_names, channel_weights["TC"])),
            "WT": dict(zip(model_names, channel_weights["WT"])),
            "ET": dict(zip(model_names, channel_weights["ET"]))
        },
        "search_weight_dice": search_weight_dice,
        "thresholds": thresholds,
        "min_et_voxels": min_et_voxels,
        "threshold_search_dice": threshold_search_dice,
        "select_score": select_score,
        "use_connected_component": USE_CONNECTED_COMPONENT,
        "search_case_limit": SEARCH_CASE_LIMIT
    }

    os.makedirs(RESULT_DIR, exist_ok=True)

    config_path = os.path.join(RESULT_DIR, "best_ensemble_config_v2.json")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"✅ v2 最优配置已保存: {config_path}")


# =========================================================
# 8. 主入口
# =========================================================

def main():
    print("\n" + "=" * 80)
    print("🚀 启动 Glioma SOTA Ensemble v2")
    print("=" * 80)

    if not os.path.exists(CACHE_DIR):
        raise FileNotFoundError(
            f"找不到缓存目录: {CACHE_DIR}。请先跑通 v1 并生成概率缓存。"
        )

    case_ids = load_case_ids()
    model_names = discover_models(case_ids)

    search_case_ids = case_ids[:min(SEARCH_CASE_LIMIT, len(case_ids))]

    bank = load_search_bank(
        case_ids=search_case_ids,
        model_names=model_names
    )

    channel_weights, search_weight_dice = search_channel_weights(
        bank=bank,
        model_names=model_names
    )

    thresholds, min_et_voxels, threshold_search_dice, select_score = search_thresholds_and_et(
        bank=bank,
        model_names=model_names,
        channel_weights=channel_weights
    )

    save_config(
        model_names=model_names,
        channel_weights=channel_weights,
        search_weight_dice=search_weight_dice,
        thresholds=thresholds,
        min_et_voxels=min_et_voxels,
        threshold_search_dice=threshold_search_dice,
        select_score=select_score
    )

    del bank
    gc.collect()

    final_res, df_case = evaluate_final(
        case_ids=case_ids,
        model_names=model_names,
        channel_weights=channel_weights,
        thresholds=thresholds,
        min_et_voxels=min_et_voxels
    )

    return final_res


if __name__ == "__main__":
    main()