import os
import glob
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # 新增：用于绘制优美的统计图表
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
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader

# ==========================================
# 0. 全局配置
# ==========================================
DATA_DIR = "./data/BraTS2021"  
MODEL_DIR = "./model"          # 统一模型读取路径
RESULT_DIR = "./result"        # 统一结果保存路径
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
set_determinism(seed=42)

# ==========================================
# 1. 数据转换与加载
# ==========================================
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key]
            if label.ndim == 4: label = label.squeeze(0)
            result = [
                torch.logical_or(label == 1, label == 4), 
                torch.logical_or(torch.logical_or(label == 1, label == 2), label == 4), 
                label == 4 
            ]
            d[key] = torch.stack(result, axis=0).float()
        return d

def prepare_val_data():
    patient_folders = sorted(glob.glob(os.path.join(DATA_DIR, "BraTS2021_*")))
    data_dicts = [{"image": [os.path.join(folder, f"{os.path.basename(folder)}_{m}.nii.gz") for m in ["flair", "t1ce", "t1", "t2"]],
                   "label": os.path.join(folder, f"{os.path.basename(folder)}_seg.nii.gz")} for folder in patient_folders]
    
    val_files = data_dicts[int(len(data_dicts) * 0.8):]
    
    val_transform = Compose([
        LoadImaged(keys=["image", "label"]), EnsureChannelFirstd(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"), Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True), EnsureTyped(keys=["image", "label"]),
    ])
    return DataLoader(CacheDataset(data=val_files, transform=val_transform, cache_num=5), batch_size=1, shuffle=False, num_workers=0)

# ==========================================
# 2. 满血版模型 definition (Pro-Res-Conv-LSTM & Swin-UNETR)
# ==========================================
# --- 模型A：Pro-Res-Conv-LSTM 核心组件 ---
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
        if self.training:
            out2 = self.out2(d2)
            out3 = self.out3(d3)
            return out1, out2, out3 
        else:
            return out1             

# --- 模型获取接口 ---
def get_model_a():
    return ProResUNetConvLSTM3D(in_ch=4, out_ch=3, init_filters=32).to(DEVICE)
def get_model_b():
    return SwinUNETR(spatial_dims=3, in_channels=4, out_channels=3, feature_size=48, use_checkpoint=True).to(DEVICE)

# ==========================================
# 3. 指标与验证辅助函数 (已修改为通道等权平均)
# ==========================================
def compute_metrics(y_true, y_pred_probs, is_binary_pred=False):
    # 先处理基础分类指标
    y_true_f = y_true.flatten()
    if is_binary_pred:
        y_pred_f = y_pred_probs.flatten().astype(np.int8)
    else:
        y_pred_probs_f = y_pred_probs.flatten()
        y_pred_f = (y_pred_probs_f > 0.5).astype(np.int8) 
    
    if len(y_true_f) > 250000:
        indices = np.random.choice(len(y_true_f), 250000, replace=False)
        y_true_f = y_true_f[indices]
        y_pred_f = y_pred_f[indices]
        if not is_binary_pred:
            y_pred_probs_f = y_pred_probs_f[indices]
            
    acc = accuracy_score(y_true_f, y_pred_f)
    pre = precision_score(y_true_f, y_pred_f, zero_division=0)
    rec = recall_score(y_true_f, y_pred_f, zero_division=0)
    try: auc = roc_auc_score(y_true_f, y_pred_f if is_binary_pred else y_pred_probs_f)
    except ValueError: auc = 0.5 

    # 🌟 关键修改：三通道分别计算 Dice 然后取算术平均
    dices = []
    for c in range(3):
        # 如果是 probs 输入则需要阈值化
        y_p = y_pred_probs[c] if is_binary_pred else (y_pred_probs[c] > 0.5).astype(np.int8)
        y_t = y_true[c]
        d = (2. * np.sum(y_t * y_p)) / (np.sum(y_t) + np.sum(y_p) + 1e-5)
        dices.append(d)
        
    return acc, pre, rec, auc, np.mean(dices), dices[0], dices[1], dices[2] # 返回 Mean, TC, WT, ET

def get_fast_dice(y_true, y_pred_prob):
    # 🌟 寻优过程也采用算术平均以保证目标一致
    y_pred = (y_pred_prob > 0.5).astype(np.int8)
    dices = []
    for c in range(3):
        d = (2. * np.sum(y_true[c] * y_pred[c])) / (np.sum(y_true[c]) + np.sum(y_pred[c]) + 1e-5)
        dices.append(d)
    return np.mean(dices)

# ==========================================
# 新增：绘图专用辅助函数
# ==========================================
def plot_grid_search_curve(weights, dices, best_w, best_dice):
    """绘制融合权重寻优曲线"""
    plt.figure(figsize=(9, 5))
    sns.set_theme(style="whitegrid")
    
    plt.plot(weights, dices, marker='o', linestyle='-', color='tab:blue', linewidth=2, label='Grid Search Dice')
    # 标出最高点
    plt.plot(best_w, best_dice, marker='*', markersize=18, color='tab:red', 
             label=f'Best Proportion (Pro-LSTM={best_w:.2f})\nMax Dice={best_dice:.4f}')
    
    plt.title('Ensemble Weight Grid Search Optimization', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Weight Assigned to Pro-Res-Conv-LSTM', fontsize=12)
    plt.ylabel('Validation Fast Dice Score', fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    
    save_path = os.path.join(RESULT_DIR, "ensemble_grid_search.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 网格寻优曲线图已保存至: '{save_path}'")

def plot_final_metrics(df):
    """绘制最终评估指标柱状图"""
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # 转置一下方便画柱状图
    df_plot = df.T.reset_index()
    df_plot.columns = ['Metric', 'Score']
    
    ax = sns.barplot(x='Metric', y='Score', data=df_plot, palette='viridis')
    plt.title('Final Ensemble Model Performance (Pro-LSTM + Swin-UNETR)', fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylim(0, 1.1)
    
    # 在柱子上标注具体数值
    for i, score in enumerate(df_plot['Score']):
        ax.text(i, score + 0.01, f'{score:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, "ensemble_final_metrics.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 最终评价指标柱状图已保存至: '{save_path}'")

# ==========================================
# 4. 终极动态寻优融合引擎
# ==========================================
def run_dynamic_ensemble():
    print("\n" + "="*60)
    print("🚀 启动工业级智能融合引擎: 满血 Pro-Res-Conv-LSTM ⚔️ Swin-UNETR")
    print("="*60)
    
    val_loader = prepare_val_data()
    
    print("\n⏳ 加载模型 A: 满血 Pro-Res-Conv-LSTM (视野 128) ...")
    model_a = get_model_a()
    model_a.load_state_dict(torch.load(os.path.join(MODEL_DIR, "pro_res_conv_lstm_standalone.pth")))
    model_a.eval()
    
    print("⏳ 加载模型 B: Swin-UNETR (视野 128) ...")
    model_b = get_model_b()
    model_b.load_state_dict(torch.load(os.path.join(MODEL_DIR, "swin_unetr_standalone.pth")))
    model_b.eval()
    
    # ========================================================
    # 阶段一：防爆内存策略！只抽取前 30 个病人做权重寻优测试
    # ========================================================
    print("\n[阶段一] 提取 30 例数据进行极速网格寻优 (防内存溢出)...")
    probs_a_list, probs_b_list, labels_list = [], [], []
    
    search_limit = 30
    with torch.no_grad():
        for i, val_data in enumerate(tqdm(val_loader, desc="[缓存先遣队数据]", total=search_limit)):
            if i >= search_limit: break
                
            val_inputs = val_data["image"].as_tensor().to(DEVICE)
            val_labels = val_data["label"].as_tensor().to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                out_a = sliding_window_inference(val_inputs, (128,128,128), 1, model_a, overlap=0.5)
                out_b = sliding_window_inference(val_inputs, (128,128,128), 1, model_b, overlap=0.5)
                
            probs_a_list.append(torch.sigmoid(out_a).cpu().numpy().astype(np.float16))
            probs_b_list.append(torch.sigmoid(out_b).cpu().numpy().astype(np.float16))
            labels_list.append(val_labels.cpu().numpy().astype(np.int8))
            
    # ========================================================
    # 阶段二：毫秒级光速网格搜索寻优
    # ========================================================
    print("\n[阶段二] 开始光速扫描最佳融合权重比例 (0.0 ~ 1.0)...")
    search_space = np.arange(0.0, 1.05, 0.05)
    best_w_a, best_dice = 0.5, 0.0
    
    # 记录数据用于画图
    history_weights, history_dices = [], []
    
    for w_a in search_space:
        w_b = 1.0 - w_a
        dices = []
        for p_a, p_b, y in zip(probs_a_list, probs_b_list, labels_list):
            ens_p = (w_a * p_a[0]) + (w_b * p_b[0]) # 移除 batch 维度
            dices.append(get_fast_dice(y[0], ens_p))
        
        mean_dice = np.mean(dices)
        history_weights.append(w_a)
        history_dices.append(mean_dice)
        
        print(f"尝试权重 -> Pro-LSTM: {w_a:.2f} | Swin-UNETR: {w_b:.2f} => 快速 Mean Dice: {mean_dice:.4f}")
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_w_a = w_a
            
    best_w_b = 1.0 - best_w_a
    print(f"\n🎉 寻优完成！最佳黄金比例定档 -> Pro-LSTM({best_w_a:.2f}) : Swin-UNETR({best_w_b:.2f})")
    
    plot_grid_search_curve(history_weights, history_dices, best_w_a, best_dice)
    del probs_a_list, probs_b_list, labels_list; gc.collect()
    
    # ========================================================
    # 阶段三：持黄金比例，对全量病人进行大考 + 终极去噪
    # ========================================================
    print(f"\n[阶段三] 工业级最终大考 (全量评估 + 连通域去噪过滤)")
    all_acc, all_pre, all_rec, all_auc, all_dice = [], [], [], [], []
    all_tc, all_wt, all_et = [], [], [] # 🌟 新增各通道记录
    post_process_cc = KeepLargestConnectedComponent(applied_labels=[1], independent=False)
    
    with torch.no_grad():
        for val_data in tqdm(val_loader, desc="[融合双王·全景扫描中]"):
            val_inputs = val_data["image"].as_tensor().to(DEVICE)
            val_labels = val_data["label"].as_tensor().to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                out_a = sliding_window_inference(val_inputs, (128,128,128), 1, model_a, overlap=0.5)
                out_b = sliding_window_inference(val_inputs, (128,128,128), 1, model_b, overlap=0.5)
                
            prob_a, prob_b = torch.sigmoid(out_a), torch.sigmoid(out_b)
            ens_prob = (best_w_a * prob_a) + (best_w_b * prob_b)
            preds = (ens_prob > 0.5).float()
            
            preds = post_process_cc(preds[0]).unsqueeze(0) 
            if preds[0, 2, ...].sum() < 50: preds[0, 2, ...] = 0.0 
            
            p_np, l_np = preds.cpu().numpy()[0], val_labels.cpu().numpy()[0]
            
            acc, pre, rec, auc, m_dice, tc, wt, et = compute_metrics(l_np, p_np, is_binary_pred=True)
            all_acc.append(acc); all_pre.append(pre); all_rec.append(rec); all_auc.append(auc)
            all_dice.append(m_dice); all_tc.append(tc); all_wt.append(wt); all_et.append(et)
            
    final_res = {
        "Accuracy": np.mean(all_acc), "Precision": np.mean(all_pre),
        "Recall": np.mean(all_rec), "AUC": np.mean(all_auc), 
        "Dice_Mean": np.mean(all_dice), "Dice_TC": np.mean(all_tc), 
        "Dice_WT": np.mean(all_wt), "Dice_ET": np.mean(all_et)
    }
    
    print("\n" + "🔥"*20)
    print("   🏆 巅峰双王 Auto-Ensemble 最终大考战绩 🏆   ")
    print("🔥"*20)
    df = pd.DataFrame({f"Ensemble": final_res}).T
    print(df.round(4).to_string())
    
    plot_final_metrics(df)
    df.to_csv(os.path.join(RESULT_DIR, "final_ensemble_results.csv"))
    print(f"\n✅ 终极记录表格已保存。大功告成！")

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(RESULT_DIR, exist_ok=True)
    run_dynamic_ensemble()