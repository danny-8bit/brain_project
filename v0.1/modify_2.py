import re

with open("2_pro_lstm_only.py", "r") as f:
    content = f.read()

# 1. Imports
content = content.replace(
"""    EnsureTyped, Spacingd,
    RandFlipd, RandRotate90d, RandScaleIntensityd, RandShiftIntensityd,
    KeepLargestConnectedComponent
)""",
"""    EnsureTyped, Spacingd,
    RandFlipd, RandRotate90d, RandScaleIntensityd, RandShiftIntensityd,
    KeepLargestConnectedComponent,
    RandCropByLabelClassesd, RandGaussianNoised, RandAdjustContrastd
)""")

# 2. Config
content = content.replace(
"""MAX_EPOCHS = 50                
TRAIN_BATCH_SIZE = 1""",
"""MAX_EPOCHS = 100                
TRAIN_BATCH_SIZE = 1           
VAL_EVERY = 5
DEFAULT_THRESHOLDS = (0.45, 0.50, 0.35)
MIN_ET_VOXELS = 20
TUNE_THRESHOLDS = True
THRESHOLD_TUNE_CASES = 20""")

# 3. Data prep
content = content.replace(
"""def prepare_data():
    patient_folders = sorted(glob.glob(os.path.join(DATA_DIR, "BraTS2021_*")))
    if MAX_SAMPLES: patient_folders = patient_folders[:MAX_SAMPLES]
    data_dicts = [{"image": [os.path.join(folder, f"{os.path.basename(folder)}_{m}.nii.gz") for m in ["flair", "t1ce", "t1", "t2"]],
                   "label": os.path.join(folder, f"{os.path.basename(folder)}_seg.nii.gz")} for folder in patient_folders]
    
    split_idx = int(len(data_dicts) * 0.8)""",
"""def prepare_data():
    patient_folders = sorted(glob.glob(os.path.join(DATA_DIR, "BraTS2021_*")))
    data_dicts = [{"image": [os.path.join(folder, f"{os.path.basename(folder)}_{m}.nii.gz") for m in ["flair", "t1ce", "t1", "t2"]],
                   "label": os.path.join(folder, f"{os.path.basename(folder)}_seg.nii.gz")} for folder in patient_folders]
    
    rng = np.random.RandomState(42)
    rng.shuffle(data_dicts)
    
    if MAX_SAMPLES: data_dicts = data_dicts[:MAX_SAMPLES]
    
    split_idx = int(len(data_dicts) * 0.8)""")

# 4. Transforms
content = content.replace(
"""    train_transform = Compose([
        LoadImaged(keys=["image", "label"]), EnsureChannelFirstd(keys=["image", "label"]), 
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"), Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=PATCH_SIZE, pos=3, neg=1, num_samples=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0), RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2), RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5), RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        EnsureTyped(keys=["image", "label"]),
    ])""",
"""    train_transform = Compose([
        LoadImaged(keys=["image", "label"]), EnsureChannelFirstd(keys=["image", "label"]), 
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandCropByLabelClassesd(keys=["image", "label"], label_key="label", spatial_size=PATCH_SIZE, ratios=[0.05, 2.0, 1.0, 0.0, 4.0], num_classes=5, num_samples=1, allow_smaller=False),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0), RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2), RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5), RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        RandGaussianNoised(keys="image", prob=0.1, mean=0.0, std=0.1),
        RandAdjustContrastd(keys="image", prob=0.15, gamma=(0.5, 2.0)),
        EnsureTyped(keys=["image", "label"]),
    ])""")

# 5. Functions & Classes
addition = """
class BratsWeightedTverskyBCELoss(nn.Module):
    def __init__(self, channel_weight=(1.6, 1.0, 2.8), pos_weight=(2.0, 1.0, 5.0), alpha=0.3, beta=0.7, gamma=1.33, lambda_bce=0.25, lambda_hierarchy=0.05, smooth=1e-5):
        super().__init__()
        self.register_buffer("channel_weight", torch.tensor(channel_weight, dtype=torch.float32))
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
        channel_weight = self.channel_weight.to(device=logits.device, dtype=logits.dtype)
        tversky_loss = torch.sum(channel_weight * torch.pow(1.0 - tversky, self.gamma)) / torch.sum(channel_weight)
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight.to(device=logits.device, dtype=logits.dtype), reduction="mean")
        hierarchy_loss = F.relu(probs[:, 2] - probs[:, 0]).mean() + F.relu(probs[:, 0] - probs[:, 1]).mean()
        return tversky_loss + self.lambda_bce * bce_loss + self.lambda_hierarchy * hierarchy_loss

def deep_supervision_brats_loss(outputs, labels, criterion, ds_weights=(1.0, 0.5, 0.25)):
    if not isinstance(outputs, (tuple, list)): outputs = [outputs]
    used_weights = ds_weights[:len(outputs)]
    loss = 0.0
    for w, out in zip(used_weights, outputs):
        target = F.interpolate(labels, size=out.shape[2:], mode="nearest") if out.shape[2:] != labels.shape[2:] else labels
        loss = loss + w * criterion(out, target)
    return loss / sum(used_weights)

def keep_largest_components(mask, num_components=1, min_size=0):
    if not np.any(mask): return mask
    try:
        from scipy import ndimage as ndi
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
        tc = probs_np[b, 0] > th_tc
        wt = probs_np[b, 1] > th_wt
        et = probs_np[b, 2] > th_et
        wt = keep_largest_components(wt, num_components=1, min_size=0)
        tc = keep_largest_components(tc, num_components=2, min_size=10)
        et = keep_largest_components(et, num_components=2, min_size=5)
        if et.sum() < min_et_voxels: et[:] = False
        tc = np.logical_or(tc, et)
        wt = np.logical_or(wt, tc)
        preds_np[b, 0] = tc.astype(np.float32)
        preds_np[b, 1] = wt.astype(np.float32)
        preds_np[b, 2] = et.astype(np.float32)
    return torch.from_numpy(preds_np).to(device=device, dtype=torch.float32)

def inference_with_tta(inputs, model, patch_size, overlap=0.6):
    outputs = sliding_window_inference(inputs, patch_size, 1, model, overlap=overlap)
    for dim in [2, 3, 4]:
        flipped_inputs = torch.flip(inputs, dims=[dim])
        flipped_outputs = sliding_window_inference(flipped_inputs, patch_size, 1, model, overlap=overlap)
        outputs += torch.flip(flipped_outputs, dims=[dim])
    return outputs / 4.0

def validate_pro_lstm(model, val_loader, thresholds=DEFAULT_THRESHOLDS, max_cases=None):
    model.eval()
    all_dice_tc, all_dice_wt, all_dice_et = [], [], []
    with torch.no_grad():
        for idx, val_data in enumerate(tqdm(val_loader, desc="[验证中]", leave=False)):
            if max_cases is not None and idx >= max_cases: break
            val_inputs = val_data["image"].as_tensor().to(DEVICE)
            val_labels = val_data["label"].cpu().numpy()
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                val_outputs = sliding_window_inference(val_inputs, PATCH_SIZE, 1, model, overlap=0.5)
            probs = torch.sigmoid(val_outputs).float()
            preds = post_process_brats_probs(probs, thresholds=thresholds, min_et_voxels=MIN_ET_VOXELS)
            preds_np = preds.cpu().numpy()
            all_dice_tc.append(calculate_brats_dice(preds_np[0, 0], val_labels[0, 0]))
            all_dice_wt.append(calculate_brats_dice(preds_np[0, 1], val_labels[0, 1]))
            all_dice_et.append(calculate_brats_dice(preds_np[0, 2], val_labels[0, 2]))
    tc_dice, wt_dice, et_dice = np.mean(all_dice_tc), np.mean(all_dice_wt), np.mean(all_dice_et)
    model.train()
    return np.mean([tc_dice, wt_dice, et_dice]), tc_dice, wt_dice, et_dice

def dice_per_channel_np(pred, target, eps=1e-5):
    pred, target = pred.astype(np.float32), target.astype(np.float32)
    axes = (0, 2, 3, 4)
    inter = np.sum(pred * target, axis=axes)
    denom = np.sum(pred, axis=axes) + np.sum(target, axis=axes)
    return (2.0 * inter + eps) / (denom + eps)

def build_pred_np_without_cc(probs_np, thresholds):
    th_tc, th_wt, th_et = thresholds
    pred = np.zeros_like(probs_np, dtype=bool)
    pred[:, 0] = probs_np[:, 0] > th_tc
    pred[:, 1] = probs_np[:, 1] > th_wt
    pred[:, 2] = probs_np[:, 2] > th_et
    pred[:, 0] = np.logical_or(pred[:, 0], pred[:, 2])
    pred[:, 1] = np.logical_or(pred[:, 1], pred[:, 0])
    return pred

def quick_search_thresholds(model, val_loader, max_cases=20):
    print("\\n🔍 正在快速搜索 TC/ET 阈值...")
    model.eval()
    tc_grid, wt_grid, et_grid = [0.35, 0.40, 0.45, 0.50], [0.45, 0.50, 0.55], [0.25, 0.30, 0.35, 0.40, 0.45]
    candidates = [(tc, wt, et) for tc in tc_grid for wt in wt_grid for et in et_grid]
    score_sum = {c: np.zeros(3, dtype=np.float64) for c in candidates}
    case_count = 0
    with torch.no_grad():
        for idx, val_data in enumerate(tqdm(val_loader, desc="[阈值搜索]", leave=False)):
            if idx >= max_cases: break
            val_inputs = val_data["image"].as_tensor().to(DEVICE)
            labels_np = val_data["label"].cpu().numpy().astype(bool)
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                val_outputs = sliding_window_inference(val_inputs, PATCH_SIZE, 1, model, overlap=0.5)
            probs_np = torch.sigmoid(val_outputs).float().cpu().numpy()
            for th in candidates:
                pred_np = build_pred_np_without_cc(probs_np, th)
                score_sum[th] += dice_per_channel_np(pred_np, labels_np)
            case_count += 1
    best_th, best_score, best_dice = DEFAULT_THRESHOLDS, -1.0, None
    for th in candidates:
        dice_ch = score_sum[th] / max(case_count, 1)
        score = 0.4 * dice_ch[0] + 0.2 * dice_ch[1] + 0.4 * dice_ch[2]
        if score > best_score: best_score, best_th, best_dice = score, th, dice_ch
    print(f"✅ 最优阈值: TC={best_th[0]:.2f}, WT={best_th[1]:.2f}, ET={best_th[2]:.2f} | TC={best_dice[0]:.4f}, WT={best_dice[1]:.4f}, ET={best_dice[2]:.4f}")
    model.train()
    return best_th
"""
content = content.replace("# 5. 单体训练与评估核心", addition + "\n# 5. 单体训练与评估核心")

# In train loop
content = content.replace(
"""    loss_function = DiceFocalLoss(smooth_nr=1e-5, smooth_dr=1e-5, squared_pred=False, to_onehot_y=False, sigmoid=True, gamma=2.0)""",
"""    loss_function = BratsWeightedTverskyBCELoss().to(DEVICE)""")

content = content.replace(
"""    scaler = torch.amp.GradScaler('cuda')
    ds_weights = [1.0, 0.5, 0.25] 
    history_loss, history_lr = [], []
    post_process_cc = KeepLargestConnectedComponent(applied_labels=[1], independent=False)
    
    monitor_data = next(iter(val_loader))""",
"""    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == "cuda"))
    ds_weights = [1.0, 0.5, 0.25] 
    history_loss, history_lr = [], []
    
    best_path = os.path.join(MODEL_DIR, "pro_res_conv_lstm_best.pth")
    last_path = os.path.join(MODEL_DIR, "pro_res_conv_lstm_last.pth")
    best_score = -1.0
    best_epoch = -1
    thresholds = DEFAULT_THRESHOLDS

    monitor_data = next(iter(val_loader))""")

content = content.replace(
"""            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = 0.0
                for i, out in enumerate(outputs):
                    out_resized = F.interpolate(out, size=labels.shape[2:], mode="trilinear", align_corners=False)
                    loss_tc = loss_function(out_resized[:, 0:1, ...], labels[:, 0:1, ...])
                    loss_wt = loss_function(out_resized[:, 1:2, ...], labels[:, 1:2, ...])
                    loss_et = loss_function(out_resized[:, 2:3, ...], labels[:, 2:3, ...])
                    
                    weighted_loss = (1.5 * loss_tc) + (1.0 * loss_wt) + (2.5 * loss_et)
                    loss += ds_weights[i] * weighted_loss
                
                loss = loss / GRAD_ACCUM_STEPS""",
"""            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                outputs = model(inputs)
                loss = deep_supervision_brats_loss(outputs, labels, loss_function, ds_weights=ds_weights)
                loss = loss / GRAD_ACCUM_STEPS""")

# model valid in loop
content = re.sub(r"        # 🟢 2D 监控切片：还是每隔 5 个 epoch 抽查一次.*?(?=    # 🌟 修改点：跳出 for epoch 循环后)", """        if (epoch + 1) % VAL_EVERY == 0 or (epoch + 1) == MAX_EPOCHS:
            mean_dice, tc_dice, wt_dice, et_dice = validate_pro_lstm(model, val_loader, thresholds=thresholds)
            select_score = 0.4 * tc_dice + 0.2 * wt_dice + 0.4 * et_dice
            print(f"\\n📌 Epoch {epoch+1} Val Dice | Mean={mean_dice:.4f}, TC={tc_dice:.4f}, WT={wt_dice:.4f}, ET={et_dice:.4f}, SelectScore={select_score:.4f}")
            if select_score > best_score:
                best_score = select_score
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_path)
                print(f"✅ 保存当前最佳模型: epoch={best_epoch}, score={best_score:.4f}")

        # 🟢 2D 监控切片
        if (epoch + 1) % 5 == 0:
            print(f"👀 正在生成 Epoch {epoch+1} 的 2D 监控切片...")
            model.eval()
            with torch.no_grad():
                mon_inputs = monitor_data["image"].to(DEVICE)
                mon_labels = monitor_data["label"]
                with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
                    mon_outputs = sliding_window_inference(mon_inputs, PATCH_SIZE, 1, model, overlap=0.5)
                
                mon_preds = (torch.sigmoid(mon_outputs) > 0.5).float().cpu()
                visualize_2d_slice(mon_inputs.cpu()[0].numpy(), mon_labels[0].numpy(), mon_preds[0].numpy(), epoch+1, GRAPH_DIR)
            model.train()
""", content, flags=re.DOTALL)


# evaluation
content = content.replace(
"""    # 🚨 保存模型权重
    save_path = os.path.join(MODEL_DIR, "pro_res_conv_lstm_standalone.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ 训练完成！模型权重已安全保存至: {save_path}")""",
"""    torch.save(model.state_dict(), last_path)
    print(f"✅ 最后一轮模型已保存至: {last_path}")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        print(f"✅ 已加载最佳模型进行最终评估: {best_path}, best_epoch={best_epoch}")
    else:
        print("⚠️ 未找到最佳模型，使用最后一轮模型评估。")
        
    if TUNE_THRESHOLDS:
        thresholds = quick_search_thresholds(model, val_loader, max_cases=min(THRESHOLD_TUNE_CASES, len(val_loader)))
    else:
        thresholds = DEFAULT_THRESHOLDS""")


content = content.replace(
"""            with torch.amp.autocast('cuda'):
                val_outputs = sliding_window_inference(val_inputs, PATCH_SIZE, 1, model, overlap=0.5)
            probs = torch.sigmoid(val_outputs).cpu()
            
            preds = torch.zeros_like(probs)
            preds[0, 0, ...] = (probs[0, 0, ...] > 0.50).float() 
            preds[0, 1, ...] = (probs[0, 1, ...] > 0.50).float() 
            preds[0, 2, ...] = (probs[0, 2, ...] > 0.45).float() 
            
            preds[0, 0, ...] = torch.logical_or(preds[0, 0, ...], preds[0, 2, ...]).float()
            preds[0, 1, ...] = torch.logical_or(preds[0, 1, ...], preds[0, 0, ...]).float()
            
            wt_cleaned = post_process_cc(preds[0, 1:2, ...]) 
            main_tumor_mask = (wt_cleaned > 0).float()
            preds[0] = preds[0] * main_tumor_mask
            
            if preds[0, 2, ...].sum() < 50: preds[0, 2, ...] = 0.0 
            
            preds_np = preds.numpy()""",
"""            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                val_outputs = inference_with_tta(val_inputs, model, PATCH_SIZE, overlap=0.6)
            probs = torch.sigmoid(val_outputs).float()
            preds = post_process_brats_probs(probs, thresholds=thresholds, min_et_voxels=MIN_ET_VOXELS)
            preds_np = preds.cpu().numpy()""")

with open("2_pro_lstm_only.py", "w") as f:
    f.write(content)
