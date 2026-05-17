import numpy as np
from scipy import ndimage


def postprocess_brats(
    prob,
    threshold=0.5,
    et_threshold_voxels=200,
    et_min_prob=0.5,
    enforce_hierarchy=True,
):
    """
    BraTS 后处理。
    输入：prob shape (3, D, H, W)，概率，通道顺序 TC, WT, ET
    输出：原始 BraTS label 体积 (D, H, W) ∈ {0, 1, 2, 4}

    关键步骤：
    1. 阈值化
    2. ET 假阳性删除（小连通域改为 NCR/NET）—— BraTS 提分关键
    3. 层级一致性 ET ⊆ TC ⊆ WT
    """
    if hasattr(prob, "cpu"):
        prob = prob.cpu().numpy()
    prob = np.asarray(prob)

    tc_mask = prob[0] >= threshold
    wt_mask = prob[1] >= threshold
    et_mask = prob[2] >= threshold

    if enforce_hierarchy:
        et_mask = et_mask & tc_mask
        tc_mask = tc_mask & wt_mask

    # ET 假阳性后处理：若 ET 体积过小，把它当作 NCR/NET
    et_voxels = int(et_mask.sum())
    if et_voxels < et_threshold_voxels:
        # 如果 ET 概率最大值仍较低，整体删除
        if prob[2].max() < et_min_prob:
            et_mask = np.zeros_like(et_mask)

    # 组装原始 BraTS label
    # WT \ TC -> ED (2)
    # TC \ ET -> NCR/NET (1)
    # ET -> 4
    out = np.zeros(prob.shape[1:], dtype=np.uint8)
    out[wt_mask] = 2
    out[tc_mask] = 1
    out[et_mask] = 4

    # WT 最大连通域过滤（去除散点假阳性）
    labeled, n = ndimage.label(out > 0)
    if n > 1:
        sizes = ndimage.sum(out > 0, labeled, range(1, n + 1))
        max_label = int(np.argmax(sizes)) + 1
        keep = labeled == max_label
        out = np.where(keep, out, 0).astype(np.uint8)

    return out