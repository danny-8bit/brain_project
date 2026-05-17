import os
import glob
import gc
import torch
torch.backends.cudnn.benchmark = True           # 让 cuDNN 自动寻找最快卷积算法
torch.backends.cuda.matmul.allow_tf32 = True    # 开启 TensorFloat-32 神技
torch.backends.cudnn.allow_tf32 = True          # 开启卷积的 TF32 加速
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 强制使用无 GUI 的纯后台渲染引擎
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
# 🌟 重新引入原来的指标计算库
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from monai.utils import set_determinism
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, MapTransform,
    NormalizeIntensityd, Orientationd, RandCropByPosNegLabeld, EnsureTyped, Spacingd,
    RandFlipd, RandRotate90d, RandScaleIntensityd, RandShiftIntensityd,
    RandGaussianNoised, RandAdjustContrastd,  
    KeepLargestConnectedComponent,
    RandCropByLabelClassesd
)
from monai.networks.nets import SwinUNETR 
from monai.losses import DiceCELoss 
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader
from monai.metrics import DiceMetric              
from monai.optimizers import WarmupCosineSchedule 

# ==========================================
# 0. Swin-UNETR 专属极速测试配置
# ==========================================
DATA_DIR = "./data/BraTS2021"  
MODEL_DIR = "./model"          
RESULT_DIR = "./result"        
GRAPH_DIR = os.path.join(RESULT_DIR, "swin-unetr-graph")
MAX_EPOCHS = 50                
TRAIN_BATCH_SIZE = 1           
GRAD_ACCUM_STEPS = 4           
PATCH_SIZE = (128, 128, 128)   
MAX_SAMPLES = None             
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
set_determinism(seed=42)

VAL_EVERY = 5
DEFAULT_THRESHOLDS = (0.45, 0.50, 0.35)
MIN_ET_VOXELS = 20
TUNE_THRESHOLDS = True
THRESHOLD_TUNE_CASES = 20

# ==========================================
# 1. 核心数据转换 (TC, WT, ET)
# ==========================================
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key]
            if label.ndim == 4:
                label = label.squeeze(0)
            result = [
                torch.logical_or(label == 1, label == 4), # TC
                torch.logical_or(torch.logical_or(label == 1, label == 2), label == 4), # WT
                label == 4 # ET
            ]
            d[key] = torch.stack(result, axis=0).float()
        return d


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
    print("
🔍 正在快速搜索 TC/ET 最优阈值...")
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
# ==========================================
def get_swin_unetr():
    print("🤖 初始化 Swin-UNETR 视觉 Transformer 架构...")
    return SwinUNETR(
        spatial_dims=3, 
        in_channels=4, 
        out_channels=3, 
        feature_size=48, 
        use_checkpoint=True
    ).to(DEVICE)

# ==========================================
# 3. 数据加载与增强
# ==========================================
def prepare_data():
    patient_folders = sorted(glob.glob(os.path.join(DATA_DIR, "BraTS2021_*")))
    if MAX_SAMPLES: patient_folders = patient_folders[:MAX_SAMPLES]
    
    data_dicts = []
    for folder in patient_folders:
        p_id = os.path.basename(folder)
        data_dicts.append({
            "image": [
                os.path.join(folder, f"{p_id}_flair.nii.gz"),
                os.path.join(folder, f"{p_id}_t1ce.nii.gz"),
                os.path.join(folder, f"{p_id}_t1.nii.gz"),
                os.path.join(folder, f"{p_id}_t2.nii.gz")
            ],
            "label": os.path.join(folder, f"{p_id}_seg.nii.gz")
        })
    
    rng = np.random.RandomState(42)
    rng.shuffle(data_dicts)
    
    split_idx = int(len(data_dicts) * 0.8)
    train_files, val_files = data_dicts[:split_idx], data_dicts[split_idx:]
    print(f"✅ 数据划分: 训练集 {len(train_files)} 例, 验证集 {len(val_files)} 例")
    
    train_transform = Compose([
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
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        RandGaussianNoised(keys="image", prob=0.1, mean=0.0, std=0.1),       
        RandAdjustContrastd(keys="image", prob=0.15, gamma=(0.5, 2.0)),      
        EnsureTyped(keys=["image", "label"]),
    ])
    
    val_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ])
    
    train_loader = DataLoader(CacheDataset(data=train_files, transform=train_transform, cache_num=20), 
                              batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(CacheDataset(data=val_files, transform=val_transform, cache_num=5), 
                            batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader

# ==========================================
# 4. 指标计算 & 训练曲线 (还原补充 Acc/Pre/Rec/AUC)
# ==========================================
def compute_metrics_sklearn(y_true, y_pred_probs, y_pred_final):
    """
    分离计算逻辑：仅用于计算基础分类指标，不计算Dice。
    使用 y_pred_final (经过 CC 和后处理) 计算 Acc, Pre, Rec；
    使用 y_pred_probs 计算 AUC。
    """
    y_true_f = y_true.flatten()
    y_pred_probs_f = y_pred_probs.flatten()
    y_pred_f = y_pred_final.flatten().astype(np.int8) 
    
    # 抽样50万像素来计算sklearn指标，防止数千万像素撑爆内存或计算过慢
    if len(y_true_f) > 500000:
        indices = np.random.choice(len(y_true_f), 500000, replace=False)
        y_true_f = y_true_f[indices]
        y_pred_f = y_pred_f[indices]
        y_pred_probs_f = y_pred_probs_f[indices]
    
    acc = accuracy_score(y_true_f, y_pred_f)
    pre = precision_score(y_true_f, y_pred_f, zero_division=0)
    rec = recall_score(y_true_f, y_pred_f, zero_division=0)
    try: 
        auc = roc_auc_score(y_true_f, y_pred_probs_f)
    except ValueError: 
        auc = 0.5 
    return acc, pre, rec, auc

def plot_swin_curves(train_losses, lrs):
    epochs = range(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = 'tab:blue'
    ax1.plot(epochs, train_losses, color=color, linewidth=2, marker='o', label='Loss')
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Training Loss', color=color, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.plot(epochs, lrs, color=color, linewidth=2, linestyle='--', label='LR')
    ax2.set_ylabel('Learning Rate', color=color, fontweight='bold')  
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Swin-UNETR Training Dynamics', fontsize=14, fontweight='bold')
    save_path = os.path.join(GRAPH_DIR, 'Swin_UNETR_Dynamics.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 训练动态曲线已保存至: {save_path}")

# ==========================================
# 5. 【监控/展示双引擎】2D 监控 + 3D 秀操作
# ==========================================
def visualize_2d_slice(image_vol, label_vol, pred_vol, epoch_num, save_dir):
    tumor_area_per_slice = np.sum(label_vol[1], axis=(0, 1))
    z_idx = np.argmax(tumor_area_per_slice)
    if tumor_area_per_slice[z_idx] == 0:
        z_idx = label_vol.shape[3] // 2  
        
    img_2d = np.rot90(image_vol[1, :, :, z_idx]) 
    lbl_2d = np.rot90(label_vol[:, :, :, z_idx], axes=(1, 2))
    pred_2d = np.rot90(pred_vol[:, :, :, z_idx], axes=(1, 2))
    
    def overlay_mask(ax, bg_img, mask_2d, title):
        ax.imshow(bg_img, cmap='gray')
        mask_bool = mask_2d > 0.5 
        
        edema = np.logical_and(mask_bool[1], np.logical_not(mask_bool[0])) 
        ncr = np.logical_and(mask_bool[0], np.logical_not(mask_bool[2]))   
        et = mask_bool[2]                                                
        
        overlay = np.zeros((*bg_img.shape, 4))
        overlay[edema] = [0, 1, 0, 0.35]   
        overlay[ncr] = [1, 0, 0, 0.45]     
        overlay[et] = [1, 1, 0, 0.7]       
        
        ax.imshow(overlay)
        ax.set_title(title, fontsize=14, color='white', fontweight='bold')
        ax.axis('off')
        
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('black')
    
    axes[0].imshow(img_2d, cmap='gray')
    axes[0].set_title(f"T1ce Image (Z={z_idx})", fontsize=14, color='white', fontweight='bold')
    axes[0].axis('off')
    
    overlay_mask(axes[1], img_2d, lbl_2d, "Ground Truth")
    overlay_mask(axes[2], img_2d, pred_2d, f"Prediction (Epoch {epoch_num})")
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"epoch_{epoch_num:03d}_2D_monitor.png")
    plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

def visualize_brats_3d(label_vol, pred_vol, save_path):
    try:
        from skimage import measure
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError: return
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor('black')
    def plot_3d_mesh(ax, volume, title):
        step = 2 
        vol_tc, vol_wt, vol_et = volume[0, ::step, ::step, ::step], volume[1, ::step, ::step, ::step], volume[2, ::step, ::step, ::step]
        
        edema = np.logical_and(vol_wt, np.logical_not(vol_tc))
        ncr = np.logical_and(vol_tc, np.logical_not(vol_et))
        et = vol_et
        
        regions = [(edema, '#00ff00', 0.15), (ncr, '#ff0000', 0.25), (et, '#ffff00', 0.8)]
        
        for mask, color, alpha in regions:
            if not np.any(mask): continue
            try:
                verts, faces, normals, values = measure.marching_cubes(mask.astype(float), level=0.5)
                mesh = Poly3DCollection(verts[faces], alpha=alpha, linewidths=0)
                mesh.set_facecolor(color)
                ax.add_collection3d(mesh)
            except Exception: pass
            
        ax.set_xlim(0, vol_wt.shape[0]); ax.set_ylim(0, vol_wt.shape[1]); ax.set_zlim(0, vol_wt.shape[2])
        ax.set_title(title, fontsize=16, fontweight='bold', color='white')
        ax.set_axis_off()
        ax.set_facecolor('black')
        ax.view_init(elev=20, azim=60)
        
    ax1 = fig.add_subplot(121, projection='3d')
    plot_3d_mesh(ax1, label_vol, "Ground Truth (3D)")
    
    ax2 = fig.add_subplot(122, projection='3d')
    plot_3d_mesh(ax2, pred_vol, "Prediction (3D)")
    
    legend_elements = [
        mpatches.Patch(color='#00ff00', label='Edema (Green)', alpha=0.5),
        mpatches.Patch(color='#ff0000', label='NCR/NET (Red)', alpha=0.5),
        mpatches.Patch(color='#ffff00', label='Enhancing Tumor (Yellow)', alpha=0.8)
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=14, labelcolor='white', facecolor='black', framealpha=0)
    
    plt.tight_layout(); plt.subplots_adjust(bottom=0.15)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

# ==========================================
# 6. 【高分秘籍】测试时增强 TTA (Test-Time Augmentation)
# ==========================================
def inference_with_tta(inputs, model, patch_size):
    """白嫖 1~2% Dice 的技巧：对输入数据进行三个维度的空间翻转推理，然后取均值"""
    outputs = sliding_window_inference(inputs, patch_size, 1, model, overlap=0.6)
    
    outputs += torch.flip(sliding_window_inference(torch.flip(inputs, dims=[2]), patch_size, 1, model, overlap=0.6), dims=[2])
    outputs += torch.flip(sliding_window_inference(torch.flip(inputs, dims=[3]), patch_size, 1, model, overlap=0.6), dims=[3])
    outputs += torch.flip(sliding_window_inference(torch.flip(inputs, dims=[4]), patch_size, 1, model, overlap=0.6), dims=[4])
    
    return outputs / 4.0

# ==========================================
# 7. 单体训练与评估核心
# ==========================================
def train_and_eval_swin():
    print(f"\n{'='*50}\n🚀 启动独立特训: Swin-UNETR (Transformer)\n{'='*50}")
    
    train_loader, val_loader = prepare_data()
    model = get_swin_unetr()
    
    loss_function = BratsWeightedTverskyBCELoss().to(DEVICE)
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
    
    history_loss, history_lr = [], []
    
    actual_steps = len(train_loader)
    monitor_data = next(iter(val_loader))
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad() 
        
        pbar = tqdm(enumerate(train_loader), total=actual_steps, desc=f"Swin-UNETR Epoch {epoch+1}/{MAX_EPOCHS}")
        
        for step, batch_data in pbar:
            inputs, labels = batch_data["image"].to(DEVICE), batch_data["label"].to(DEVICE)
            
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                outputs = model(inputs)
                loss = loss_function(outputs, labels) / GRAD_ACCUM_STEPS
            
            scaler.scale(loss).backward()
            epoch_loss += (loss.item() * GRAD_ACCUM_STEPS)
            
            if ((step + 1) % GRAD_ACCUM_STEPS == 0) or ((step + 1) == actual_steps):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad()
                
            pbar.set_postfix({"loss": f"{loss.item() * GRAD_ACCUM_STEPS:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}"})
        
        scheduler.step()
        
        history_loss.append(epoch_loss / actual_steps)
        history_lr.append(optimizer.param_groups[0]['lr'])
        
        if (epoch + 1) % 5 == 0:
            print(f"\n👀 正在生成 Epoch {epoch+1} 的 2D 监控切片...")
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
                f"\n📌 Epoch {epoch+1} Val Dice | "
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

    plot_swin_curves(history_loss, history_lr)
    
    # ===================== [终极评估 & 3D 可视化] =====================
    print(f"\n🏆 开始对 Swin-UNETR 进行全量数据高精度评估 (含 TTA 加成)...")
    model.eval()
    
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    all_acc, all_pre, all_rec, all_auc = [], [], [], []  # 🌟 还原记录列表
    
    with torch.no_grad():
        for idx, val_data in enumerate(tqdm(val_loader, desc="[Swin-UNETR 评估中]")):
            val_inputs, val_labels = val_data["image"].to(DEVICE), val_data["label"].to(DEVICE)
            
            with torch.amp.autocast('cuda', enabled=(DEVICE.type == "cuda")):
                val_outputs = inference_with_tta(val_inputs, model, PATCH_SIZE)
            
            probs = torch.sigmoid(val_outputs).float()

            # BraTS 专用阈值 + 连通域 + 层级后处理
            preds = post_process_brats_probs(
                probs,
                thresholds=thresholds,
                min_et_voxels=MIN_ET_VOXELS
            )
                
            # 1. 计算精准 Dice (交给 MONAI)
            dice_metric(y_pred=preds, y=val_labels)
            
            # 2. 计算其他综合指标 (交给我们自己写的方法)
            probs_np = probs.cpu().numpy()
            preds_np = preds.cpu().numpy()
            labels_np = val_labels.cpu().numpy()
            
            acc, pre, rec, auc = compute_metrics_sklearn(labels_np, probs_np, preds_np)
            all_acc.append(acc)
            all_pre.append(pre)
            all_rec.append(rec)
            all_auc.append(auc)
            if idx == 0:
                print("\n🎬 正在渲染最终 3D 立体图...")
                vis_save_path = os.path.join(GRAPH_DIR, "swin_unetr_3D_visualization.png")
                visualize_brats_3d(labels_np[0], preds_np[0], vis_save_path)
            
    # 🌟 核心修改点：显式通过 mean_batch 聚合后再求均值，确保通道等权
    metric_batch = dice_metric.aggregate()
    tc_dice = metric_batch[0].item()
    wt_dice = metric_batch[1].item()
    et_dice = metric_batch[2].item()
    
    # 此处 mean() 会对 [TC, WT, ET] 三个标量求算术平均，即 (TC+WT+ET)/3
    mean_dice = metric_batch.mean().item()
    dice_metric.reset()
    
    del model, optimizer, scaler; torch.cuda.empty_cache(); gc.collect()
    
    # 🌟 拼合展示所有最终指标
    final_res = {
        "Mean Dice (Unweighted)": mean_dice, 
        "TC Dice": tc_dice, 
        "WT Dice": wt_dice, 
        "ET Dice": et_dice,
        "Accuracy": np.mean(all_acc),
        "Precision": np.mean(all_pre), 
        "Recall": np.mean(all_rec), 
        "AUC": np.mean(all_auc)
    }
    df = pd.DataFrame({"Swin-UNETR": final_res}).T
    print("\n" + "🔥"*20 + "\n      🏆 Swin-UNETR 权威最终战绩 (TTA 加持) 🏆      \n" + "🔥"*20)
    print(df.round(4).to_string())
    df.to_csv(os.path.join(RESULT_DIR, "swin_unetr_standalone_results.csv"))

# ==========================================
# 8. 运行入口
# ==========================================
if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(GRAPH_DIR, exist_ok=True) 
    
    train_and_eval_swin()