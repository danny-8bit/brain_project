import os
# ==========================================
# ⚡ 稳定补丁
# ==========================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import glob
import gc
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

from monai.utils import set_determinism
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, MapTransform,
    NormalizeIntensityd, Orientationd, EnsureTyped, Spacingd,
    KeepLargestConnectedComponent
)
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, ThreadDataLoader 

# ==========================================
# 0. 配置与路径 (保持与训练一致)
# ==========================================
DATA_DIR = "./data/BraTS2021"  
MODEL_PATH = "./model/pro_res_conv_lstm_standalone.pth" # 直接指向权重文件
RESULT_DIR = "./result"        
GRAPH_DIR = os.path.join(RESULT_DIR, "conv-lstm-graph")
PATCH_SIZE = (128, 128, 128)   
MAX_SAMPLES = None             
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 🌟 必须固定随机种子！保证划分出的验证集和训练时一模一样！
set_determinism(seed=42)

# ==========================================
# 1. 核心数据转换
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
# 2. 满血版网络架构 (必须完整保留以加载权重)
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
        out = self.conv1(x)
        out = self.conv2(out)
        return self.relu(out + res)

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
    def __init__(self, in_ch=4, out_ch=3, init_filters=32, lstm_steps=3):
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
        h_t = torch.zeros(b, c, d, h, w, device=x.device)
        c_t = torch.zeros(b, c, d, h, w, device=x.device)
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

# ==========================================
# 3. 数据加载 (仅加载验证集)
# ==========================================
def prepare_val_data():
    patient_folders = sorted(glob.glob(os.path.join(DATA_DIR, "BraTS2021_*")))
    if MAX_SAMPLES: patient_folders = patient_folders[:MAX_SAMPLES]
    data_dicts = [{"image": [os.path.join(folder, f"{os.path.basename(folder)}_{m}.nii.gz") for m in ["flair", "t1ce", "t1", "t2"]],
                   "label": os.path.join(folder, f"{os.path.basename(folder)}_seg.nii.gz")} for folder in patient_folders]
    
    split_idx = int(len(data_dicts) * 0.8)
    val_files = data_dicts[split_idx:] # 严格取后 20% 作为验证集
    print(f"✅ 已加载验证集: {len(val_files)} 例进行纯评估")
    
    val_transform = Compose([
        LoadImaged(keys=["image", "label"]), EnsureChannelFirstd(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"), Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True), EnsureTyped(keys=["image", "label"]),
    ])
    
    val_loader = ThreadDataLoader(
        CacheDataset(data=val_files, transform=val_transform, cache_num=5), 
        batch_size=1, shuffle=False, num_workers=0
    )
    return val_loader

# ==========================================
# 4. 绘图与指标计算
# ==========================================
def compute_metrics(y_true, y_pred_probs, is_binary_pred=False):
    y_true_f = y_true.flatten()
    if is_binary_pred: y_pred_f = y_pred_probs.flatten().astype(np.int8)
    else: y_pred_probs_f = y_pred_probs.flatten(); y_pred_f = (y_pred_probs_f > 0.5).astype(np.int8) 
    
    if len(y_true_f) > 250000:
        indices = np.random.choice(len(y_true_f), 250000, replace=False)
        y_true_f, y_pred_f = y_true_f[indices], y_pred_f[indices]
        if not is_binary_pred: y_pred_probs_f = y_pred_probs_f[indices]
            
    acc = accuracy_score(y_true_f, y_pred_f)
    pre = precision_score(y_true_f, y_pred_f, zero_division=0)
    rec = recall_score(y_true_f, y_pred_f, zero_division=0)
    dice = (2. * np.sum(y_true_f * y_pred_f)) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-5)
    try: auc = roc_auc_score(y_true_f, y_pred_f if is_binary_pred else y_pred_probs_f)
    except ValueError: auc = 0.5 
    return acc, pre, rec, auc, dice

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
# 5. 纯评估主函数 (包含最新版后处理)
# ==========================================
def evaluate_only():
    print(f"\n{'='*60}\n🚀 启动纯评估模式 (加载权重直接验证)\n{'='*60}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型权重文件: {MODEL_PATH}")
        return

    val_loader = prepare_val_data()
    
    print("🤖 正在实例化模型并加载权重...")
    model = ProResUNetConvLSTM3D(in_ch=4, out_ch=3, init_filters=32, lstm_steps=3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ 权重加载成功！")
    post_process_cc = KeepLargestConnectedComponent(applied_labels=[1], independent=False)
    all_acc, all_pre, all_rec, all_auc, all_dice = [], [], [], [], []
    
    with torch.no_grad():
        for idx, val_data in enumerate(tqdm(val_loader, desc="[正在全力评估中]")):
            val_inputs = val_data["image"].as_tensor().to(DEVICE)
            val_labels = val_data["label"].as_tensor().to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                val_outputs = sliding_window_inference(val_inputs, PATCH_SIZE, 1, model, overlap=0.5)
            
            probs = torch.sigmoid(val_outputs)
            
            # 👇 ========================================== 👇
            # 🌟 终极进化版：独立阈值瘦身 + 生物学层级后处理 🌟
            # ========================================== 
            
            # 0. 独立阈值控制：拯救被黄色吞噬的红色核心！
            preds = torch.zeros_like(probs)
            preds[0, 0, ...] = (probs[0, 0, ...] > 0.5).float() # TC (红+黄核心) 保持 0.5，保证整体内核不缩水
            preds[0, 1, ...] = (probs[0, 1, ...] > 0.5).float() # WT (绿色主体) 保持 0.5，保证超高 Recall
            
            # 🔥 绝杀魔法：把 ET(黄色) 的准入门槛拉高到 0.65！
            # 逼迫过度膨胀的黄色向最亮的区域收缩，把内部空间“吐”出来，还给真正的红色坏死区！
            # (如果觉得出来的红色还不够大，可以改成 0.70；如果红色太大，改回 0.60)
            preds[0, 2, ...] = (probs[0, 2, ...] > 0.65).float() 
            
            # 1. 层级包含融合 (维持生物学正确性)
            # 黄色(ET)一定是红色(TC)的一部分
            preds[0, 0, ...] = torch.logical_or(preds[0, 0, ...], preds[0, 2, ...]).float()
            # 红色(TC)一定是绿色(WT)的一部分
            preds[0, 1, ...] = torch.logical_or(preds[0, 1, ...], preds[0, 0, ...]).float()
            
            # 2. 提取真正的主体外壳 (基于融合后的 WT 通道)
            wt_pred = preds[0, 1:2, ...] 
            wt_cleaned = post_process_cc(wt_pred) # 清理主体外围的孤立噪点
            main_tumor_mask = (wt_cleaned > 0).float() # 生成绝对干净的掩码
            
            # 3. 完美过滤：用干净的外壳过滤所有 3 个通道，消灭悬空碎片
            preds[0] = preds[0] * main_tumor_mask
            
            # 4. BraTS 竞赛官方要求：ET 极小则清零
            if preds[0, 2, ...].sum() < 50: 
                preds[0, 2, ...] = 0.0 
            
            # 👆 ========================================== 👆
                
            preds_np = preds.cpu().numpy()
            labels_np = val_labels.cpu().numpy()
            
            # 只为第一个样本生成 3D 渲染图用于肉眼验证
            if idx == 0:
                print("\n🎬 正在渲染 3D 效果图 (见证奇迹：红色核心重现)...")
                os.makedirs(GRAPH_DIR, exist_ok=True)
                vis_save_path = os.path.join(GRAPH_DIR, "eval_only_3D_visualization.png")
                visualize_brats_3d(labels_np[0], preds_np[0], vis_save_path)
            
            # 计算指标
            acc, pre, rec, auc, dice = compute_metrics(labels_np, preds_np, is_binary_pred=True)
            all_acc.append(acc); all_pre.append(pre); all_rec.append(rec); all_auc.append(auc); all_dice.append(dice)
            
    # 清理内存
    del model; torch.cuda.empty_cache(); gc.collect()
    
    # 汇总成绩
    final_res = {
        "Accuracy": np.mean(all_acc), "Precision": np.mean(all_pre),
        "Recall": np.mean(all_rec), "AUC": np.mean(all_auc), "Dice": np.mean(all_dice)
    }
    
    print("\n" + "🚀"*20)
    print("  🏆 修复后处理逻辑 - 纯验证最终战绩 🏆  ")
    print("🚀"*20)
    
    df = pd.DataFrame({"Pro-LSTM-Eval": final_res}).T
    print(df.round(4).to_string())
    
    # 保存结果
    os.makedirs(RESULT_DIR, exist_ok=True)
    csv_path = os.path.join(RESULT_DIR, "pro_lstm_eval_only_results.csv")
    df.to_csv(csv_path)
    print(f"\n✅ 评估指标已保存至 '{csv_path}'")

if __name__ == "__main__":
    evaluate_only()