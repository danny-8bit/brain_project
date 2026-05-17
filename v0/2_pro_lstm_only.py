import os
# ==========================================
# ⚡ 终极稳定补丁 1：限制底层 C++ 库的多核抢占
# ==========================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import glob
import gc
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
# ==========================================
# ⚡ 终极稳定补丁 2：强制 Matplotlib 使用纯计算无头模式
# ==========================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from monai.utils import set_determinism
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, MapTransform,
    NormalizeIntensityd, Orientationd, RandCropByPosNegLabeld, EnsureTyped, Spacingd,
    RandFlipd, RandRotate90d, RandScaleIntensityd, RandShiftIntensityd,
    KeepLargestConnectedComponent
)
from monai.losses import DiceFocalLoss 
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, ThreadDataLoader 

# ==========================================
# 0. Pro-Res-Conv-LSTM 专属配置
# ==========================================
DATA_DIR = "./data/BraTS2021"  
MODEL_DIR = "./model"          
RESULT_DIR = "./result"        
GRAPH_DIR = os.path.join(RESULT_DIR, "conv-lstm-graph")
MAX_EPOCHS = 100                
TRAIN_BATCH_SIZE = 1           
VAL_EVERY = 5
DEFAULT_THRESHOLDS = (0.45, 0.50, 0.35)
MIN_ET_VOXELS = 20
TUNE_THRESHOLDS = True
THRESHOLD_TUNE_CASES = 20           
GRAD_ACCUM_STEPS = 4           
PATCH_SIZE = (128, 128, 128)   
MAX_SAMPLES = None             
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
set_determinism(seed=42)

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
                torch.logical_or(label == 1, label == 4), # TC (红+黄)
                torch.logical_or(torch.logical_or(label == 1, label == 2), label == 4), # WT (绿+红+黄)
                label == 4 # ET (黄)
            ]
            d[key] = torch.stack(result, axis=0).float()
        return d

# ==========================================
# 2. 网络架构: Pro-ResUNet-ConvLSTM 3D 
# ==========================================
class SingleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class ProResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = SingleConv3D(in_ch, out_ch)
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch)
        )
        self.relu = nn.LeakyReLU(inplace=True)
        self.shortcut = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        res = self.shortcut(x)
        return self.relu(self.conv2(self.conv1(x)) + res)

class ProConvLSTM3DCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv3d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size=3, padding=1)
    def forward(self, x, hidden_state):
        h_cur, c_cur = hidden_state
        gates = self.conv(torch.cat([x, h_cur], dim=1))
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        c_next = torch.sigmoid(f) * c_cur + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next

class ProResUNetConvLSTM3D(nn.Module):
    def __init__(self, in_ch=4, out_ch=3, init_filters=32, lstm_steps=2):
        super().__init__()
        self.lstm_steps = lstm_steps 
        f = init_filters
        self.enc1 = ProResBlock3D(in_ch, f)
        self.down1 = nn.Conv3d(f, f*2, kernel_size=2, stride=2)
        self.enc2 = ProResBlock3D(f*2, f*2)
        self.down2 = nn.Conv3d(f*2, f*4, kernel_size=2, stride=2)
        self.enc3 = ProResBlock3D(f*4, f*4)
        self.down3 = nn.Conv3d(f*4, f*8, kernel_size=2, stride=2)
        self.enc4 = ProResBlock3D(f*8, f*8)
        
        self.lstm_cell = ProConvLSTM3DCell(input_dim=f*8, hidden_dim=f*8)
        
        self.up3 = nn.ConvTranspose3d(f*8, f*4, kernel_size=2, stride=2)
        self.dec3 = ProResBlock3D(f*8, f*4) 
        self.up2 = nn.ConvTranspose3d(f*4, f*2, kernel_size=2, stride=2)
        self.dec2 = ProResBlock3D(f*4, f*2)
        self.up1 = nn.ConvTranspose3d(f*2, f, kernel_size=2, stride=2)
        self.dec1 = ProResBlock3D(f*2, f)
        
        self.out3 = nn.Conv3d(f*4, out_ch, kernel_size=1) 
        self.out2 = nn.Conv3d(f*2, out_ch, kernel_size=1) 
        self.out1 = nn.Conv3d(f, out_ch, kernel_size=1)   
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        
        b, c, d, h, w = e4.size()
        h_t, c_t = torch.zeros_like(e4), torch.zeros_like(e4)
        for _ in range(self.lstm_steps):
            h_t, c_t = self.lstm_cell(e4, (h_t, c_t))
            
        d3 = self.up3(h_t)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        out1 = self.out1(d1)
        if self.training: return out1, self.out2(d2), self.out3(d3) 
        else: return out1             

def get_pro_lstm_model():
    return ProResUNetConvLSTM3D(in_ch=4, out_ch=3, init_filters=32, lstm_steps=2).to(DEVICE)

# ==========================================
# 3. 数据加载与增强
# ==========================================
def prepare_data():
    patient_folders = sorted(glob.glob(os.path.join(DATA_DIR, "BraTS2021_*")))
    data_dicts = [{"image": [os.path.join(folder, f"{os.path.basename(folder)}_{m}.nii.gz") for m in ["flair", "t1ce", "t1", "t2"]],
                   "label": os.path.join(folder, f"{os.path.basename(folder)}_seg.nii.gz")} for folder in patient_folders]
    
    rng = np.random.RandomState(42)
    rng.shuffle(data_dicts)
    
    if MAX_SAMPLES: data_dicts = data_dicts[:MAX_SAMPLES]
    
    split_idx = int(len(data_dicts) * 0.8)
    train_files, val_files = data_dicts[:split_idx], data_dicts[split_idx:]
    print(f"✅ 数据划分: 训练集 {len(train_files)} 例, 验证集 {len(val_files)} 例")
    
    train_transform = Compose([
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
    ])
    
    val_transform = Compose([
        LoadImaged(keys=["image", "label"]), EnsureChannelFirstd(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"), Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True), EnsureTyped(keys=["image", "label"]),
    ])
    
    train_loader = ThreadDataLoader(CacheDataset(data=train_files, transform=train_transform, cache_num=20), batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = ThreadDataLoader(CacheDataset(data=val_files, transform=val_transform, cache_num=5), batch_size=1, shuffle=False, num_workers=0)
    return train_loader, val_loader

# ==========================================
# 4. 评估指标与绘图可视化引擎
# ==========================================
def calculate_brats_dice(pred, true):
    pred, true = pred.astype(np.float32), true.astype(np.float32)
    return (2. * np.sum(pred * true) + 1e-5) / (np.sum(pred) + np.sum(true) + 1e-5)

def compute_fast_global_metrics(y_true, y_pred):
    y_t, y_p = y_true.astype(bool), y_pred.astype(bool)
    tp = np.logical_and(y_t, y_p).sum()
    tn = np.logical_and(~y_t, ~y_p).sum()
    fp = np.logical_and(~y_t, y_p).sum()
    fn = np.logical_and(y_t, ~y_p).sum()
    
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-5)
    pre = tp / (tp + fp + 1e-5)
    rec = tp / (tp + fn + 1e-5)
    return acc, pre, rec

def plot_pro_lstm_curves(train_losses, lrs):
    epochs = range(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = 'tab:blue'
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Training Loss', color=color, fontweight='bold')
    ax1.plot(epochs, train_losses, color=color, linewidth=2, marker='o', markersize=4, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Learning Rate', color=color, fontweight='bold')  
    ax2.plot(epochs, lrs, color=color, linewidth=2, linestyle='--', label='LR')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Pro-Res-Conv-LSTM Training Dynamics', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_name = os.path.join(GRAPH_DIR, 'Pro_LSTM_Dynamics.png')
    plt.savefig(save_name, dpi=300)
    plt.close(fig) 
    print(f"\n📊 最终训练曲线已生成并保存至: {save_name}")

def visualize_2d_slice(image_vol, label_vol, pred_vol, epoch_num, save_dir):
    tumor_area_per_slice = np.sum(label_vol[1], axis=(0, 1))
    z_idx = np.argmax(tumor_area_per_slice)
    if tumor_area_per_slice[z_idx] == 0: z_idx = label_vol.shape[3] // 2  
        
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
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch_num:03d}_2D_monitor.png"), dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
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
        ax.set_axis_off(); ax.set_facecolor('black'); ax.view_init(elev=20, azim=60)
        
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
    print("\n🔍 正在快速搜索 TC/ET 阈值...")
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

# 5. 单体训练与评估核心
# ==========================================
def train_and_eval_pro_lstm():
    train_loader, val_loader = prepare_data()
    model = get_pro_lstm_model()
    
    loss_function = BratsWeightedTverskyBCELoss().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    actual_steps = len(train_loader) 
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=4e-4, epochs=MAX_EPOCHS, steps_per_epoch=math.ceil(actual_steps / GRAD_ACCUM_STEPS), pct_start=0.2 
    )
    
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == "cuda"))
    ds_weights = [1.0, 0.5, 0.25] 
    history_loss, history_lr = [], []
    
    best_path = os.path.join(MODEL_DIR, "pro_res_conv_lstm_best.pth")
    last_path = os.path.join(MODEL_DIR, "pro_res_conv_lstm_last.pth")
    best_score = -1.0
    best_epoch = -1
    thresholds = DEFAULT_THRESHOLDS

    monitor_data = next(iter(val_loader)) 
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad() 
        pbar = tqdm(enumerate(train_loader), total=actual_steps, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}")
        
        for step, batch_data in pbar:
            inputs = batch_data["image"].as_tensor().to(DEVICE)
            labels = batch_data["label"].as_tensor().to(DEVICE)
            
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                outputs = model(inputs)
                loss = deep_supervision_brats_loss(outputs, labels, loss_function, ds_weights=ds_weights)
                loss = loss / GRAD_ACCUM_STEPS
            
            scaler.scale(loss).backward()
            epoch_loss += (loss.item() * GRAD_ACCUM_STEPS)
            
            if ((step + 1) % GRAD_ACCUM_STEPS == 0) or ((step + 1) == actual_steps):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
            pbar.set_postfix({"loss": f"{loss.item() * GRAD_ACCUM_STEPS:.4f}"})
        
        history_loss.append(epoch_loss / actual_steps)
        history_lr.append(optimizer.param_groups[0]['lr'])
        
        if (epoch + 1) % VAL_EVERY == 0 or (epoch + 1) == MAX_EPOCHS:
            mean_dice, tc_dice, wt_dice, et_dice = validate_pro_lstm(model, val_loader, thresholds=thresholds)
            select_score = 0.4 * tc_dice + 0.2 * wt_dice + 0.4 * et_dice
            print(f"
📌 Epoch {epoch+1} Val Dice | Mean={mean_dice:.4f}, TC={tc_dice:.4f}, WT={wt_dice:.4f}, ET={et_dice:.4f}, SelectScore={select_score:.4f}")
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
    # 🌟 修改点：跳出 for epoch 循环后，在这里一次性绘制最终的学习率和 Loss 曲线
    plot_pro_lstm_curves(history_loss, history_lr)

    torch.save(model.state_dict(), last_path)
    print(f"✅ 最后一轮模型已保存至: {last_path}")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        print(f"✅ 已加载最佳模型进行最终评估: {best_path}, best_epoch={best_epoch}")
    else:
        print("⚠️ 未找到最佳模型，使用最后一轮模型评估。")
        
    if TUNE_THRESHOLDS:
        thresholds = quick_search_thresholds(model, val_loader, max_cases=min(THRESHOLD_TUNE_CASES, len(val_loader)))
    else:
        thresholds = DEFAULT_THRESHOLDS
    
    # ===================== [终极评估] =====================
    model.eval()
    all_dice_tc, all_dice_wt, all_dice_et = [], [], []
    all_acc, all_pre, all_rec = [], [], []
    
    with torch.no_grad():
        for idx, val_data in enumerate(tqdm(val_loader, desc="[全力评估中]")):
            val_inputs = val_data["image"].as_tensor().to(DEVICE)
            val_labels = val_data["label"].cpu().numpy()
            
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                val_outputs = inference_with_tta(val_inputs, model, PATCH_SIZE, overlap=0.6)
            probs = torch.sigmoid(val_outputs).float()
            preds = post_process_brats_probs(probs, thresholds=thresholds, min_et_voxels=MIN_ET_VOXELS)
            preds_np = preds.cpu().numpy()
            
            if idx == 0:
                print("\n🎬 正在渲染 3D 效果图...")
                visualize_brats_3d(val_labels[0], preds_np[0], os.path.join(GRAPH_DIR, "eval_3D_visualization.png"))
            
            all_dice_tc.append(calculate_brats_dice(preds_np[0, 0], val_labels[0, 0]))
            all_dice_wt.append(calculate_brats_dice(preds_np[0, 1], val_labels[0, 1]))
            all_dice_et.append(calculate_brats_dice(preds_np[0, 2], val_labels[0, 2]))
            
            acc, pre, rec = compute_fast_global_metrics(val_labels[0], preds_np[0])
            all_acc.append(acc); all_pre.append(pre); all_rec.append(rec)
            
    final_res = {
        "Accuracy": np.mean(all_acc), 
        "Precision": np.mean(all_pre), 
        "Recall": np.mean(all_rec),
        "Dice_Global_Avg": np.mean([np.mean(all_dice_wt), np.mean(all_dice_tc), np.mean(all_dice_et)]),
        "Dice_WT (整体)": np.mean(all_dice_wt), 
        "Dice_TC (核心)": np.mean(all_dice_tc), 
        "Dice_ET (增强)": np.mean(all_dice_et)
    }
    
    print("\n" + "🔥"*20)
    print("  🏆 优化后独立特训战绩 🏆  ")
    df = pd.DataFrame({"Pro-LSTM": final_res}).T
    print(df.round(4).to_string())
    df.to_csv(os.path.join(RESULT_DIR, "pro_lstm_standalone_results.csv"))

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(GRAPH_DIR, exist_ok=True) 
    train_and_eval_pro_lstm()