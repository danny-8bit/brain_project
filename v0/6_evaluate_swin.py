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
import warnings
warnings.filterwarnings("ignore")
from monai.utils import set_determinism
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, MapTransform,
    NormalizeIntensityd, Orientationd, EnsureTyped, Spacingd,
    KeepLargestConnectedComponent
)
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader

# 🚀 引入 MONAI 官方的 Swin UNETR
from monai.networks.nets import SwinUNETR

# ==========================================
# 0. 配置与路径 (Swin-UNETR 专属)
# ==========================================
DATA_DIR = "./data/BraTS2021"  
MODEL_PATH = "./model/swin_unetr_standalone.pth" 
RESULT_DIR = "./result"        
GRAPH_DIR = os.path.join(RESULT_DIR, "swin-unetr-graph")
PATCH_SIZE = (128, 128, 128)   
MAX_SAMPLES = None             
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
                torch.logical_or(label == 1, label == 4), # TC (红+黄) -> Channel 0
                torch.logical_or(torch.logical_or(label == 1, label == 2), label == 4), # WT (绿+红+黄) -> Channel 1
                label == 4 # ET (黄) -> Channel 2
            ]
            d[key] = torch.stack(result, axis=0).float()
        return d

# ==========================================
# 2. 数据加载
# ==========================================
def prepare_val_data():
    patient_folders = sorted(glob.glob(os.path.join(DATA_DIR, "BraTS2021_*")))
    if MAX_SAMPLES: patient_folders = patient_folders[:MAX_SAMPLES]
    data_dicts = [{"image": [os.path.join(folder, f"{os.path.basename(folder)}_{m}.nii.gz") for m in ["flair", "t1ce", "t1", "t2"]],
                   "label": os.path.join(folder, f"{os.path.basename(folder)}_seg.nii.gz")} for folder in patient_folders]
    
    split_idx = int(len(data_dicts) * 0.8)
    val_files = data_dicts[split_idx:]
    print(f"✅ 已加载验证集: {len(val_files)} 例进行纯评估")
    
    val_transform = Compose([
        LoadImaged(keys=["image", "label"]), EnsureChannelFirstd(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"), Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True), EnsureTyped(keys=["image", "label"]),
    ])
    
    val_dataset = Dataset(data=val_files, transform=val_transform)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4,       
        pin_memory=True,     
        prefetch_factor=2    
    )
    return val_loader

# ==========================================
# 3. 绘图与指标计算 (🌟 扩充各区域指标 🌟)
# ==========================================
def compute_metrics_gpu(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    eps = 1e-5
    
    # --- 1. 计算全局指标 ---
    tp = (y_true_f * y_pred_f).sum()
    tn = ((1 - y_true_f) * (1 - y_pred_f)).sum()
    fp = ((1 - y_true_f) * y_pred_f).sum()
    fn = (y_true_f * (1 - y_pred_f)).sum()
    
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    pre = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    dice = (2. * tp) / (y_true_f.sum() + y_pred_f.sum() + eps)
    
    tpr = rec
    tnr = tn / (tn + fp + eps) 
    auc = (tpr + tnr) / 2.0
    
    # --- 2. 计算各区域特定指标 (BraTS 标准评测) ---
    def calc_dice_per_channel(c):
        t = y_true[:, c, ...]
        p = y_pred[:, c, ...]
        return (2. * (t * p).sum()) / (t.sum() + p.sum() + eps)
    
    dice_tc = calc_dice_per_channel(0)
    dice_wt = calc_dice_per_channel(1)
    dice_et = calc_dice_per_channel(2)
    
    return acc.item(), pre.item(), rec.item(), auc.item(), dice.item(), dice_tc.item(), dice_wt.item(), dice_et.item()

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
# 4. 纯评估主函数
# ==========================================
def evaluate_only():
    print(f"\n{'='*60}\n🚀 启动纯评估模式 (加载权重直接验证 Swin-UNETR)\n{'='*60}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型权重文件: {MODEL_PATH}")
        return
    val_loader = prepare_val_data()
    
    print("🤖 正在实例化 Swin-UNETR 模型并加载权重...")
    # ⚠️ 注意: 如果你训练时用的 feature_size 不是 48（例如 24），请在此处修改。
    model = SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=48, 
        use_checkpoint=False
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ 权重加载成功！")
    
    post_process_cc = KeepLargestConnectedComponent(applied_labels=[1], independent=False)
    
    all_acc, all_pre, all_rec, all_auc, all_dice_global = [], [], [], [], []
    all_dice_tc, all_dice_wt, all_dice_et = [], [], []
    
    with torch.no_grad():
        for idx, val_data in enumerate(tqdm(val_loader, desc="[正在全力评估中]")):
            val_inputs = val_data["image"].to(DEVICE, non_blocking=True)
            val_labels = val_data["label"].to(DEVICE, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                val_outputs = sliding_window_inference(val_inputs, PATCH_SIZE, 1, model, overlap=0.25)
            
            probs = torch.sigmoid(val_outputs)
            
            preds = torch.zeros_like(probs)
            preds[0, 0, ...] = (probs[0, 0, ...] > 0.5).float() 
            preds[0, 1, ...] = (probs[0, 1, ...] > 0.5).float() 
            preds[0, 2, ...] = (probs[0, 2, ...] > 0.60).float() 
            
            preds[0, 0, ...] = torch.logical_or(preds[0, 0, ...], preds[0, 2, ...]).float()
            preds[0, 1, ...] = torch.logical_or(preds[0, 1, ...], preds[0, 0, ...]).float()
            
            wt_pred = preds[0, 1:2, ...] 
            wt_cleaned = post_process_cc(wt_pred).to(DEVICE)
            main_tumor_mask = (wt_cleaned > 0).float() 
            
            preds[0] = preds[0] * main_tumor_mask
            
            if preds[0, 2, ...].sum() < 50: 
                preds[0, 2, ...] = 0.0 
                
            if idx == 0:
                print("\n🎬 正在渲染 3D 效果图 ")
                os.makedirs(GRAPH_DIR, exist_ok=True)
                vis_save_path = os.path.join(GRAPH_DIR, "eval_only_3D_visualization.png")
                visualize_brats_3d(val_labels.cpu().numpy()[0], preds.cpu().numpy()[0], vis_save_path)
            
            # 分区域接收指标
            acc, pre, rec, auc, dice, dice_tc, dice_wt, dice_et = compute_metrics_gpu(val_labels, preds)
            
            all_acc.append(acc); all_pre.append(pre); all_rec.append(rec); all_auc.append(auc); all_dice_global.append(dice)
            all_dice_tc.append(dice_tc)
            all_dice_wt.append(dice_wt)
            all_dice_et.append(dice_et)
            
    # 清理内存
    del model; torch.cuda.empty_cache(); gc.collect()
    
    # 汇总成绩
    final_res = {
        "Accuracy": np.mean(all_acc), 
        "Precision": np.mean(all_pre),
        "Recall": np.mean(all_rec), 
        "AUC": np.mean(all_auc), 
        "Dice_Global": np.mean(all_dice_global),
        "Dice_WT (整体)": np.mean(all_dice_wt),
        "Dice_TC (核心)": np.mean(all_dice_tc),
        "Dice_ET (增强)": np.mean(all_dice_et)
    }
    
    print("\n" + "🚀"*20)
    print("  🏆 Swin-UNETR - 纯验证最终战绩 🏆  ")
    print("🚀"*20)
    
    # 报告名称也改为 Swin-UNETR-Eval
    df = pd.DataFrame({"Swin-UNETR-Eval": final_res}).T
    print(df.round(4).to_string())
    
    # 保存结果
    os.makedirs(RESULT_DIR, exist_ok=True)
    csv_path = os.path.join(RESULT_DIR, "swin_unetr_eval_only_results.csv")
    df.to_csv(csv_path)
    print(f"\n✅ 评估指标已保存至 '{csv_path}'")

if __name__ == "__main__":
    evaluate_only()