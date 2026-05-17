import torch
from monai.inferers import sliding_window_inference


# 8 种翻转组合
FLIP_DIMS = [
    (),
    (2,),
    (3,),
    (4,),
    (2, 3),
    (2, 4),
    (3, 4),
    (2, 3, 4),
]


@torch.no_grad()
def sliding_window_tta(
    image,
    model,
    roi_size,
    sw_batch_size=4,
    overlap=0.5,
    mode="gaussian",
    flips=True,
    sigmoid=True,
):
    """
    支持 TTA 翻转的滑窗推理。返回概率图。
    image: (B, C, D, H, W)
    """
    def predictor(x):
        out = model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out

    if not flips:
        logits = sliding_window_inference(
            inputs=image,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=predictor,
            overlap=overlap,
            mode=mode,
        )
        return torch.sigmoid(logits) if sigmoid else logits

    probs = None
    for dims in FLIP_DIMS:
        x = torch.flip(image, dims=dims) if dims else image
        logits = sliding_window_inference(
            inputs=x,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=predictor,
            overlap=overlap,
            mode=mode,
        )
        p = torch.sigmoid(logits) if sigmoid else logits
        if dims:
            p = torch.flip(p, dims=dims)
        probs = p if probs is None else probs + p

    probs /= len(FLIP_DIMS)
    return probs