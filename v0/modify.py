import os
import numpy as np

with open("3_swin_only.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Imports
content = content.replace(
    '''import torch.nn as nn
import pandas as pd''',
    '''import torch.nn as nn
import torch.nn.functional as F
import pandas as pd'''
)

content = content.replace(
    '''    KeepLargestConnectedComponent
)''',
    '''    KeepLargestConnectedComponent,
    RandCropByLabelClassesd
)'''
)

# 2. Globals
content = content.replace(
    '''DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
set_determinism(seed=42)''',
    '''DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
set_determinism(seed=42)

VAL_EVERY = 5
DEFAULT_THRESHOLDS = (0.45, 0.50, 0.35)
MIN_ET_VOXELS = 20
TUNE_THRESHOLDS = True
THRESHOLD_TUNE_CASES = 20'''
)

# Loss Function & Additional Utility Functions
loss_and_util_funcs = '''
class BratsWeightedTverskyBCELoss(nn.Module):
    def __init__(self, channel_weight=(1.5, 1.0, 2.5), pos_weight=(2.0, 1.0, 4.0), alpha=0.3, beta=0.7, gamma=1.33, lambda_bce=0.3, lambda_hierarchy=0.05, smooth=1e-5):
        super().__init__()
        self.register_buffer("channel_weight", torch.tensor(channel_weight, dtype=torch.float32).view(1, 3))
        self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float32).view(1, 3, 1, 1, 1))
        self.alpha = alpha; self.beta = beta; self.gamma = gamma; self.lambda_bce = lambda_bce; self.lambda_hierarchy = lambda_hierarchy; self.smooth = smooth
    def forward(self, logits, target):
        target = target.float()
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3, 4)
        tp = torch.sum(probs * target, dim=dims)
        fp = torch.sum(probs * (1.0 - target), dim=dims)
        fn = torch.sum((1.0 - probs) * target, dim=dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        channel_weight = self.channel_weight.to(logits.device, logits.dtype).view(3)
        tversky_loss = torch.sum(channel_weight * torch.pow(1.0 - tversky, self.gamma)) / torch.sum(channel_weight)
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight.to(logits.device, logits.dtype), reduction="mean")
        hierarchy_loss = F.relu(probs[:, 2] - probs[:, 0]).mean() + F.relu(probs[:, 0] - probs[:, 1]).mean()
        return tversky_loss + self.lambda_bce * bce_loss + self.lambda_hierarchy * hierarchy_loss

def keep_largest_components(mask, num_components=1, min_size=0):
    if not np.any(mask): return mask
    try: from scipy import ndimage as ndi
    except ImportError: return mask
    labeled, num = ndi.label(mask)
    if num == 0: return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    valid_ids = np.where(sizes >= min_size)[0]
    valid_ids = valid_ids[valid_ids != 0]
    if valid_ids.size == 0: return np.zeros_like(mask, dtype=bool)
    if num_components is not None and valid_ids.size > num_components:
        valid_ids = valid_ids[np.argsort(sizes[valid_ids])[-num_components:]]
    return np.isin(labeled, valid_ids)

def post_process_brats_probs(probs, thresholds=DEFAULT_THRESHOLDS, min_et_voxels=MIN_ET_VOXELS):
    device = probs.device
    probs_np = probs.detach().float().cpu().numpy()
    preds_np = np.zeros_like(probs_np, dtype=np.float32)
    th_tc, th_wt, th_et = thresholds
    for b in range(probs_np.shape[0]):
        tc, wt, et = probs_np[b, 0] > th_tc, probs_np[b, 1] > th_wt, probs_np[b, 2] > th_et
        wt = keep_largest_components(wt, num_components=1, min_size=0)
        tc = keep_largest_components(tc, num_components=1, min_size=0)
        et = keep_largest_components(et, num_components=2, min_size=5)
        if et.sum() < min_et_voxels: et[:] = False
        tc = np.logical_or(tc, et)
        wt = np.logical_or(wt, tc)
        preds_np[b, 0], preds_np[b, 1], preds_np[b, 2] = tc.astype(np.float32), wt.astype(np.float32), et.astype(np.float32)
    return torch.from_numpy(preds_np).to(device=device, dtype=torch.float32)

def validate_swin(model, val_loader, thresholds=DEFAULT_THRESHOLDS, max_cases=None):
    model.eval()
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    with torch.no_grad():
        for idx, val_data in enumerate(tqdm(val_loader, desc="[验证中]", leave=False)):
            if max_cases is not None and idx >= max_cases: break
            val_inputs = val_data["image"].to(DEVICE)
            val_labels = val_data["label"].to(DEVICE)
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                val_outputs = sliding_window_inference(val_inputs, PATCH_SIZE, 1, model, overlap=0.6)
            probs = torch.sigmoid(val_outputs).float()
            preds = post_process_brats_probs(probs, thresholds=thresholds)
            dice_metric(y_pred=preds, y=val_labels)
    metric_batch = dice_metric.aggregate()
    dice_metric.reset()
    tc_dice, wt_dice, et_dice = metric_batch[0].item(), metric_batch[1].item(), metric_batch[2].item()
    mean_dice = metric_batch.mean().item()
    model.train()
    return mean_dice, tc_dice, wt_dice, et_dice

def dice_per_channel_np(pred, target, eps=1e-5):
    pred, target = pred.astype(np.float32), target.astype(np.float32)
    axes = (0, 2, 3, 4)
    inter = np.sum(pred * target, axis=axes)
    denom = np.sum(pred, axis=axes) + np.sum(target, axis=axes)
    return (2.0 * inter + eps) / (denom + eps)

def build_pred_np_without_cc(probs_np, thresholds):
    th_tc, th_wt, th_et = thresholds
    pred = np.zeros_like(probs_np, dtype=bool)
    pred[:, 0], pred[:, 1], pred[:, 2] = probs_np[:, 0] > th_tc, probs_np[:, 1] > th_wt, probs_np[:, 2] > th_et
    pred[:, 0] = np.logical_or(pred[:, 0], pred[:, 2])
    pred[:, 1] = np.logical_or(pred[:, 1], pred[:, 0])
    return pred

def quick_search_thresholds(model, val_loader, max_cases=20):
    print("\n🔍 正在快速搜索 TC/ET 最优阈值...")
    model.eval()
    candidates = [(tc, wt, et) for tc in [0.35, 0.40, 0.45, 0.50] for wt in [0.45, 0.50, 0.55] for et in [0.25, 0.30, 0.35, 0.40, 0.45]]
    score_sum = {c: np.zeros(3, dtype=np.float64) for c in candidates}
    case_count = 0
    with torch.no_grad():
        for idx, val_data in enumerate(tqdm(val_loader, desc="[阈值搜索]", leave=False)):
            if idx >= max_cases: break
            val_inputs = val_data["image"].to(DEVICE)
            val_labels = val_data["label"].cpu().numpy().astype(bool)
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                val_outputs = sliding_window_inference(val_inputs, PATCH_SIZE, 1, model, overlap=0.6)
            probs_np = torch.sigmoid(val_outputs).float().cpu().numpy()
            for th in candidates:
                pred_np = build_pred_np_without_cc(probs_np, th)
                score_sum[th] += dice_per_channel_np(pred_np, val_labels)
            case_count += 1
    best_th, best_score, best_dice = DEFAULT_THRESHOLDS, -1.0, None
    for th in candidates:
        dice_ch = score_sum[th] / max(case_count, 1)
        score = 0.4 * dice_ch[0] + 0.2 * dice_ch[1] + 0.4 * dice_ch[2]
        if score > best_score:
            best_score, best_th, best_dice = score, th, dice_ch
    print(f"✅ 最优阈值: TC={best_th[0]:.2f}, WT={best_th[1]:.2f}, ET={best_th[2]:.2f} | TC={best_dice[0]:.4f}, WT={best_dice[1]:.4f}, ET={best_dice[2]:.4f}")
    model.train()
    return best_th

# ==========================================
# 2. 获取 Swin-UNETR 模型
# =========================================='''

content = content.replace(
    '''# ==========================================
# 2. 获取 Swin-UNETR 模型
# ==========================================''',
    loss_and_util_funcs
)

# 3. Data Shuffle
content = content.replace(
    '''        })
    
    split_idx = int(len(data_dicts) * 0.8)''',
    '''        })
    
    rng = np.random.RandomState(42)
    rng.shuffle(data_dicts)
    
    split_idx = int(len(data_dicts) * 0.8)'''
)

# 4. Train Transform
content = content.replace(
    '''    train_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]), 
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=PATCH_SIZE, pos=1, neg=1, num_samples=1),''',
    '''    train_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]), 
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=PATCH_SIZE,
            ratios=[0.05, 2.0, 1.0, 0.0, 4.0],
            num_classes=5,
            num_samples=1,
            allow_smaller=False
        ),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),'''
)

# 5. Training Loop modifications
content = content.replace(
    '''    loss_function = DiceCELoss(smooth_nr=1e-5, smooth_dr=1e-5, squared_pred=False, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    scheduler = WarmupCosineSchedule(
        optimizer=optimizer,
        warmup_steps=5,         # 🌟 已修复的参数
        warmup_multiplier=0.1,
        t_total=MAX_EPOCHS
    )
    
    scaler = torch.amp.GradScaler('cuda')
    save_path = os.path.join(MODEL_DIR, "swin_unetr_standalone.pth")
    history_loss, history_lr = [], []
    post_process_cc = KeepLargestConnectedComponent(applied_labels=[1], independent=False)''',
    '''    loss_function = BratsWeightedTverskyBCELoss().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    scheduler = WarmupCosineSchedule(
        optimizer=optimizer,
        warmup_steps=5,         # 🌟 已修复的参数
        warmup_multiplier=0.1,
        t_total=MAX_EPOCHS
    )
    
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
    best_path = os.path.join(MODEL_DIR, "swin_unetr_best.pth")
    last_path = os.path.join(MODEL_DIR, "swin_unetr_last.pth")
    
    best_score = -1.0
    best_epoch = -1
    thresholds = DEFAULT_THRESHOLDS
    
    history_loss, history_lr = [], []'''
)

content = content.replace(
    '''            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = loss_function(outputs, labels) / GRAD_ACCUM_STEPS''',
    '''            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                outputs = model(inputs)
                loss = loss_function(outputs, labels) / GRAD_ACCUM_STEPS'''
)

content = content.replace(
    '''        history_loss.append(epoch_loss / actual_steps)
        history_lr.append(optimizer.param_groups[0]['lr'])
        
        if (epoch + 1) % 5 == 0:
            print(f"\\n👀 正在生成 Epoch {epoch+1} 的 2D 监控切片...")
            model.eval()
            with torch.no_grad():
                mon_inputs = monitor_data["image"].to(DEVICE)
                mon_labels = monitor_data["label"] 
                with torch.amp.autocast('cuda'):
                    mon_outputs = sliding_window_inference(mon_inputs, PATCH_SIZE, 1, model, overlap=0.6)
                mon_preds = (torch.sigmoid(mon_outputs) > 0.5).float().cpu()
                visualize_2d_slice(mon_inputs.cpu()[0].numpy(), mon_labels[0].numpy(), mon_preds[0].numpy(), epoch+1, GRAPH_DIR)
            model.train()
            
    torch.save(model.state_dict(), save_path)
    print(f"✅ 训练完成！Swin-UNETR 权重已保存至 {save_path}")''',
    '''        history_loss.append(epoch_loss / actual_steps)
        history_lr.append(optimizer.param_groups[0]['lr'])
        
        if (epoch + 1) % 5 == 0:
            print(f"\\n👀 正在生成 Epoch {epoch+1} 的 2D 监控切片...")
            model.eval()
            with torch.no_grad():
                mon_inputs = monitor_data["image"].to(DEVICE)
                mon_labels = monitor_data["label"] 
                with torch.amp.autocast('cuda', enabled=(DEVICE.type == "cuda")):
                    mon_outputs = sliding_window_inference(mon_inputs, PATCH_SIZE, 1, model, overlap=0.6)
                mon_preds = (torch.sigmoid(mon_outputs) > 0.5).float().cpu()
                visualize_2d_slice(mon_inputs.cpu()[0].numpy(), mon_labels[0].numpy(), mon_preds[0].numpy(), epoch+1, GRAPH_DIR)
            model.train()
            
        if (epoch + 1) % VAL_EVERY == 0 or (epoch + 1) == MAX_EPOCHS:
            mean_dice, tc_dice, wt_dice, et_dice = validate_swin(
                model,
                val_loader,
                thresholds=thresholds
            )

            select_score = 0.4 * tc_dice + 0.2 * wt_dice + 0.4 * et_dice

            print(
                f"\\n📌 Epoch {epoch+1} Val Dice | "
                f"Mean={mean_dice:.4f}, TC={tc_dice:.4f}, WT={wt_dice:.4f}, ET={et_dice:.4f}, "
                f"SelectScore={select_score:.4f}"
            )

            if select_score > best_score:
                best_score = select_score
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_path)
                print(f"✅ 保存当前最佳模型: epoch={best_epoch}, score={best_score:.4f}")
            
    torch.save(model.state_dict(), last_path)
    print(f"✅ 训练完成！Swin-UNETR 权重已保存至 {last_path}")
    
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        print(f"✅ 已加载最佳模型进行最终评估: {best_path}, best_epoch={best_epoch}")
    else:
        print("⚠️ 未找到最佳模型，使用最后一个 epoch 模型评估。")
        
    if TUNE_THRESHOLDS:
        thresholds = quick_search_thresholds(
            model,
            val_loader,
            max_cases=min(THRESHOLD_TUNE_CASES, len(val_loader))
        )
    else:
        thresholds = DEFAULT_THRESHOLDS
'''
)

# 6. Evaluation processing modifications
content = content.replace(
    '''    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    all_acc, all_pre, all_rec, all_auc = [], [], [], []  # 🌟 还原记录列表''',
    '''    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    all_acc, all_pre, all_rec, all_auc = [], [], [], []  # 🌟 还原记录列表'''
)

content = content.replace(
    '''            with torch.amp.autocast('cuda'):
                val_outputs = inference_with_tta(val_inputs, model, PATCH_SIZE)
            
            # 概率矩阵
            probs = torch.sigmoid(val_outputs).float()
            # 初始二分类
            preds = (probs > 0.5).float() 
            # 连通域后处理
            preds = post_process_cc(preds[0]).unsqueeze(0) 
            
            et_mask_sum = preds[0, 2, ...].sum()
            tc_mask_sum = preds[0, 0, ...].sum()
            if et_mask_sum < 73 or tc_mask_sum == 0: 
                preds[0, 2, ...] = 0.0 ''',
    '''            with torch.amp.autocast('cuda', enabled=(DEVICE.type == "cuda")):
                val_outputs = inference_with_tta(val_inputs, model, PATCH_SIZE)
            
            probs = torch.sigmoid(val_outputs).float()

            # BraTS 专用阈值 + 连通域 + 层级后处理
            preds = post_process_brats_probs(
                probs,
                thresholds=thresholds,
                min_et_voxels=MIN_ET_VOXELS
            )'''
)

with open("3_swin_only.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Replacement complete.")
