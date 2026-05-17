#五个模型的横向对比程序
import os
import glob
import gc
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import warnings
import matplotlib.pyplot as plt
import seaborn as sns  # 新增：用于绘制优美的统计图表
warnings.filterwarnings("ignore") # 屏蔽切片警告
from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete, Compose, LoadImaged, EnsureChannelFirstd, MapTransform,
    NormalizeIntensityd, Orientationd, RandCropByPosNegLabeld, EnsureTyped, Spacingd
)
from monai.networks.nets import UNet, SwinUNETR, SegResNet
from monai.losses import DiceCELoss 
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch 

# ==========================================
# 0. 全局性能配置 (针对 RTX 5070 Ti 16GB 优化)
# ==========================================
DATA_DIR = "./data/BraTS2021"  # 请确保路径正确
MODEL_DIR = "./model"          # 模型保存路径
RESULT_DIR = "./result"        # 结果保存路径
MAX_EPOCHS = 50                # 对比测试轮数
TRAIN_BATCH_SIZE = 1           # 5070Ti 16GB显存应对复杂3D模型的安全基准
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
set_determinism(seed=42)

# ==========================================
# 1. 核心数据转换：生成 TC, WT, ET 三通道
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

# ==========================================
# 2. 自定义 3D 模型库 (R2U-Net & Conv-LSTM)
# ==========================================
class RecurrentBlock3D(nn.Module):
    def __init__(self, out_ch, t=2):
        super().__init__()
        self.t = t
        self.conv = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        out = self.conv(x)
        for i in range(1, self.t):
            out = self.conv(x + out)
        return out

class RRCNNBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, t=2):
        super().__init__()
        self.conv1x1 = nn.Conv3d(in_ch, out_ch, 1)
        self.rrcnn = RecurrentBlock3D(out_ch, t)
    def forward(self, x):
        x1 = self.conv1x1(x)
        return x1 + self.rrcnn(x1)

class R2UNet3D(nn.Module):
    def __init__(self, in_ch=4, out_ch=3, t=2):
        super().__init__()
        self.enc1 = RRCNNBlock3D(in_ch, 16, t)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = RRCNNBlock3D(16, 32, t)
        self.pool2 = nn.MaxPool3d(2)
        self.bottleneck = RRCNNBlock3D(32, 64, t)
        self.up1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec1 = RRCNNBlock3D(64, 32, t)
        self.up2 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.dec2 = RRCNNBlock3D(32, 16, t)
        self.final = nn.Conv3d(16, out_ch, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d1 = self.up1(b)
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        return self.final(d2)

class ConvLSTM3DCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv3d(input_dim + hidden_dim, 4 * hidden_dim, 3, padding=1)
    def forward(self, x, hidden_state):
        h_cur, c_cur = hidden_state
        gates = self.conv(torch.cat([x, h_cur], dim=1))
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        c_next = torch.sigmoid(f) * c_cur + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next

class UNetConvLSTM3D(nn.Module):
    def __init__(self, in_ch=4, out_ch=3):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv3d(in_ch, 16, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = nn.Sequential(nn.Conv3d(16, 32, 3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool3d(2)
        
        self.lstm_bn = ConvLSTM3DCell(32, 32)
        
        self.up1 = nn.ConvTranspose3d(32, 32, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv3d(64, 16, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose3d(16, 16, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv3d(32, 16, 3, padding=1), nn.ReLU())
        self.final = nn.Conv3d(16, out_ch, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        bn_in = self.pool2(e2)
        
        device = x.device
        h_t = torch.zeros(x.size(0), 32, bn_in.size(2), bn_in.size(3), bn_in.size(4), device=device)
        c_t = torch.zeros(x.size(0), 32, bn_in.size(2), bn_in.size(3), bn_in.size(4), device=device)
        
        h_t, c_t = self.lstm_bn(bn_in, (h_t, c_t))
        
        d1 = self.up1(h_t)
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        return self.final(d2)

# ==========================================
# 3. 模型工厂
# ==========================================
def get_model(name):
    common_args = {"in_channels": 4, "out_channels": 3}
    if name == "UNet":
        return UNet(spatial_dims=3, channels=(16, 32, 64, 128), strides=(2, 2, 2), **common_args).to(DEVICE)
    elif name == "SegResNet":
        return SegResNet(blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], init_filters=32, dropout_prob=0.2, **common_args).to(DEVICE)
    elif name == "R2U-Net":
        return R2UNet3D(in_ch=4, out_ch=3).to(DEVICE)
    elif name == "Conv-LSTM":
        return UNetConvLSTM3D(in_ch=4, out_ch=3).to(DEVICE)
    elif name == "Swin-Unet":
        return SwinUNETR(feature_size=24, patch_size=2, **common_args).to(DEVICE)

# ==========================================
# 4. 数据加载准备
# ==========================================
def prepare_data():
    patient_folders = sorted(glob.glob(os.path.join(DATA_DIR, "BraTS2021_*")))[:30] 
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
    
    train_files, val_files = data_dicts[:24], data_dicts[24:]
    
    train_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]), 
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=1),
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
    
    train_loader = DataLoader(CacheDataset(data=train_files, transform=train_transform, cache_num=24), 
                              batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(CacheDataset(data=val_files, transform=val_transform, cache_num=6), 
                            batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader

# ==========================================
# 5. 指标计算与评估逻辑
# ==========================================
def compute_metrics(y_true, y_pred_probs):
    y_true_f = y_true.flatten()
    y_pred_probs_f = y_pred_probs.flatten()
    y_pred_f = (y_pred_probs_f > 0.5).astype(np.int8) 
    
    if len(y_true_f) > 200000:
        indices = np.random.choice(len(y_true_f), 200000, replace=False)
        y_true_f, y_pred_probs_f, y_pred_f = y_true_f[indices], y_pred_probs_f[indices], y_pred_f[indices]
        
    acc = accuracy_score(y_true_f, y_pred_f)
    pre = precision_score(y_true_f, y_pred_f, zero_division=0)
    rec = recall_score(y_true_f, y_pred_f, zero_division=0)
    
    intersection = np.sum(y_true_f * y_pred_f)
    dice = (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-5)
    
    try:
        auc = roc_auc_score(y_true_f, y_pred_probs_f)
    except ValueError:
        auc = 0.5 
    return acc, pre, rec, auc, dice

def train_and_eval(model_name, train_loader, val_loader):
    print(f"\n" + "="*40)
    print(f"🚀 开始测试模型: {model_name}")
    print("="*40)
    
    model = get_model(model_name)
    loss_function = DiceCELoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    
    # --- 训练阶段 ---
    for epoch in range(MAX_EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Train]")
        
        for batch_data in pbar:
            inputs = batch_data["image"].as_tensor().to(DEVICE)
            labels = batch_data["label"].as_tensor().to(DEVICE)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
    # --- 保存模型 ---
    model_save_path = os.path.join(MODEL_DIR, f"{model_name}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"\n💾 模型 {model_name} 已成功保存至: {model_save_path}")

    # --- 验证阶段 ---
    print(f"[{model_name}] 训练完成，正在评估各项指标...")
    model.eval()
    all_acc, all_pre, all_rec, all_auc, all_dice = [], [], [], [], []
    
    with torch.no_grad():
        for val_data in tqdm(val_loader, desc="[Validating]"):
            val_inputs = val_data["image"].as_tensor().to(DEVICE)
            val_labels = val_data["label"].as_tensor().to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                val_outputs = sliding_window_inference(
                    val_inputs, 
                    roi_size=(96, 96, 96), 
                    sw_batch_size=1,
                    predictor=model, 
                    overlap=0.25
                )
            
            probs = torch.sigmoid(val_outputs).cpu().numpy()
            labels_np = val_labels.cpu().numpy()
            
            acc, pre, rec, auc, dice = compute_metrics(labels_np, probs)
            all_acc.append(acc); all_pre.append(pre); all_rec.append(rec); all_auc.append(auc); all_dice.append(dice)
            
    # 释放显存防爆
    del model, optimizer, scaler, inputs, labels, val_inputs, val_outputs, val_labels
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        "Accuracy": np.mean(all_acc),
        "Precision": np.mean(all_pre),
        "Recall": np.mean(all_rec),
        "AUC": np.mean(all_auc),
        "Dice": np.mean(all_dice)
    }

# ==========================================
# 7. 可视化绘图模块 
# ==========================================
def plot_results(df):
    """
    接收DataFrame对象并生成对比分析图表
    """
    print("\n" + "="*40)
    print("📊 正在生成可视化图表...")
    
    # 设置Seaborn主题风格
    sns.set_theme(style="whitegrid")
    
    # 1. 综合指标分组柱状图
    fig, ax = plt.subplots(figsize=(12, 6))
    df.plot(kind='bar', ax=ax, width=0.8, colormap='Set2')
    ax.set_title('BraTS 2021 Performance Comparison Across 5 Models', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylim(0, 1.1)  # 限制Y轴为0-1.1，留出图例空间
    plt.xticks(rotation=15, fontsize=11)
    # 将图例移出图表避免遮挡数据
    plt.legend(title='Metrics', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    
    bar_chart_path = os.path.join(RESULT_DIR, "brats_models_bar_chart.png")
    plt.savefig(bar_chart_path, dpi=300)
    print(f"✅ 综合柱状图已保存至: '{bar_chart_path}'")
    
    # 2. 指标数值热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, cmap='YlGnBu', fmt=".4f", vmin=0, vmax=1, linewidths=.5)
    plt.title('Metrics Heatmap', fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('Models')
    plt.tight_layout()
    
    heatmap_path = os.path.join(RESULT_DIR, "brats_models_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    print(f"✅ 数据热力图已保存至: '{heatmap_path}'")
    
    # 3. Dice 指标单独对比图 (医学分割的核心指标)
    plt.figure(figsize=(8, 5))
    # 为排序后的Dice单独作图
    dice_df = df[['Dice']].sort_values(by='Dice', ascending=False)
    ax = sns.barplot(x=dice_df.index, y=dice_df['Dice'], palette='Blues_r')
    plt.title('Dice Score Ranking', fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('Dice Score', fontsize=12)
    plt.xlabel('Models', fontsize=12)
    plt.ylim(0, 1.0)
    
    # 在柱子上标注具体数值
    for i, score in enumerate(dice_df['Dice']):
        ax.text(i, score + 0.01, f'{score:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    plt.tight_layout()
    
    dice_chart_path = os.path.join(RESULT_DIR, "brats_models_dice_chart.png")
    plt.savefig(dice_chart_path, dpi=300)
    print(f"✅ Dice 得分图已保存至: '{dice_chart_path}'")
    plt.close('all')

# ==========================================
# 8. 主循环与表格生成
# ==========================================
if __name__ == "__main__":
    # 检查并创建对应的文件夹
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    print(f"检测到极速设备: {DEVICE} | Batch Size: {TRAIN_BATCH_SIZE}")
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ 错误: 找不到数据集路径 '{DATA_DIR}'。请修改 DATA_DIR。")
        exit()
        
    train_loader, val_loader = prepare_data()
    
    # 包含全阵容的 5 个模型
    models_to_test = ["UNet", "R2U-Net", "Conv-LSTM", "Swin-Unet", "SegResNet"]
    final_results = {}
    
    for m_name in models_to_test:
        torch.cuda.empty_cache()
        gc.collect()
        try:
            results = train_and_eval(m_name, train_loader, val_loader)
            final_results[m_name] = results
        except Exception as e:
            print(f"❌ 模型 {m_name} 训练失败，跳过。错误信息: {e}")
            final_results[m_name] = {"Accuracy": 0, "Precision": 0, "Recall": 0, "AUC": 0, "Dice": 0}
            torch.cuda.empty_cache()
            
    # 打印和保存最终表格
    print("\n" + "🌟"*20)
    print("      BraTS 2021 终极横向对比结果      ")
    print("🌟"*20)
    
    df = pd.DataFrame(final_results).T
    df = df.sort_values(by="Dice", ascending=False)
    
    print(df.round(4).to_string())
    
    # 保存CSV到 result 文件夹
    csv_path = os.path.join(RESULT_DIR, "brats_ultimate_comparison.csv")
    df.to_csv(csv_path)
    print(f"\n✅ 数据表格已保存至 '{csv_path}'")
    
    # 生成可视化图表，并保存到 result 文件夹
    plot_results(df)