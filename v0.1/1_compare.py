import os
import glob
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")
from monai.utils import set_determinism
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, MapTransform,
    NormalizeIntensityd, Orientationd, RandCropByPosNegLabeld, EnsureTyped, Spacingd,
    RandFlipd, RandRotate90d, RandScaleIntensityd, RandShiftIntensityd
)
from monai.networks.nets import SegResNetDS
from monai.losses import DiceCELoss 
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader

# ==========================================
# 0. 比赛级全局配置
# ==========================================
DATA_DIR = "./data/BraTS2021"  
MAX_EPOCHS = 50                # 总 Epoch 数
STEPS_PER_EPOCH = 50           # 每个 Epoch 限制训练的 Iteration (Batch) 数量
TRAIN_BATCH_SIZE = 1           # 5070Ti 16GB 安全基准
MAX_SAMPLES = None             # 设为 None 使用全量数据
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
set_determinism(seed=42)

# ==========================================
# 1. 核心数据转换 (TC, WT, ET 三通道生成)
# ==========================================
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key]
            if label.ndim == 4:
                label = label.squeeze(0)
            result = [
                torch.logical_or(label == 1, label == 4), # TC (肿瘤核心)
                torch.logical_or(torch.logical_or(label == 1, label == 2), label == 4), # WT (整个肿瘤)
                label == 4 # ET (增强肿瘤)
            ]
            d[key] = torch.stack(result, axis=0).float()
        return d

# ==========================================
# 2. 模型库 A: Res-Conv-LSTM 3D 
# ==========================================
class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm3d(out_ch) 
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm3d(out_ch)
        self.shortcut = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        res = self.shortcut(x)
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)

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

class ResUNetConvLSTM3D(nn.Module):
    def __init__(self, in_ch=4, out_ch=3):
        super().__init__()
        self.enc1 = ResBlock3D(in_ch, 16)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ResBlock3D(16, 32)
        self.pool2 = nn.MaxPool3d(2)
        
        self.lstm_bn = ConvLSTM3DCell(32, 32)
        
        self.up1 = nn.ConvTranspose3d(32, 32, 2, stride=2)
        self.dec1 = ResBlock3D(64, 16) 
        self.up2 = nn.ConvTranspose3d(16, 16, 2, stride=2)
        self.dec2 = ResBlock3D(32, 16) 
        
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
# 3. 模型获取工厂 [已修复]
# ==========================================
def get_model(name):
    if name == "SegResNetDS (Champion)":
        return SegResNetDS(
            blocks_down=[1, 2, 2, 4], 
            blocks_up=[1, 1, 1], 
            init_filters=32, 
            in_channels=4, 
            out_channels=3,
            dsdepth=4
        ).to(DEVICE)
    elif name == "Res-Conv-LSTM":
        return ResUNetConvLSTM3D(in_ch=4, out_ch=3).to(DEVICE)

# ==========================================
# 4. 赛事级数据加载与增强
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
    
    split_idx = int(len(data_dicts) * 0.8)
    train_files, val_files = data_dicts[:split_idx], data_dicts[split_idx:]
    print(f"✅ 数据划分: 训练集 {len(train_files)} 例, 验证集 {len(val_files)} 例")
    
    train_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]), 
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
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
# 5. 指标计算
# ==========================================
def compute_metrics(y_true, y_pred_probs):
    y_true_f, y_pred_probs_f = y_true.flatten(), y_pred_probs.flatten()
    y_pred_f = (y_pred_probs_f > 0.5).astype(np.int8) 
    
    if len(y_true_f) > 250000:
        indices = np.random.choice(len(y_true_f), 250000, replace=False)
        y_true_f, y_pred_probs_f, y_pred_f = y_true_f[indices], y_pred_probs_f[indices], y_pred_f[indices]
    
    acc = accuracy_score(y_true_f, y_pred_f)
    pre = precision_score(y_true_f, y_pred_f, zero_division=0)
    rec = recall_score(y_true_f, y_pred_f, zero_division=0)
    dice = (2. * np.sum(y_true_f * y_pred_f)) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-5)
    try: auc = roc_auc_score(y_true_f, y_pred_probs_f)
    except ValueError: auc = 0.5 
    return acc, pre, rec, auc, dice

# ==========================================
# 6. 可视化绘图辅助函数
# ==========================================
def plot_training_curves(model_name, train_losses, lrs):
    sanitized_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    epochs = range(1, len(train_losses) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # 绘制 Loss 曲线 (左Y轴)
    color = 'tab:blue'
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Training Loss', color=color, fontweight='bold')
    ax1.plot(epochs, train_losses, color=color, linewidth=2, marker='o', markersize=4, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 绘制 Learning Rate 曲线 (右Y轴)
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Learning Rate', color=color, fontweight='bold')  
    ax2.plot(epochs, lrs, color=color, linewidth=2, linestyle='--', label='LR')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Training Dynamics: {model_name}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(f'{sanitized_name}_training_curves.png', dpi=300)
    plt.close()
    print(f"📊 训练曲线已保存至: {sanitized_name}_training_curves.png")

# ==========================================
# 7. 统一的自适应核心训练引擎 (无中途验证版)
# ==========================================
def train_and_eval(model_name, train_loader, val_loader):
    print(f"\n{'='*50}\n🚀 开始训练 SOTA 级模型: {model_name}\n{'='*50}")
    model = get_model(model_name)
    loss_function = DiceCELoss(smooth_nr=1e-5, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    actual_steps = min(STEPS_PER_EPOCH, len(train_loader))
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=5e-4, 
        epochs=MAX_EPOCHS, 
        steps_per_epoch=actual_steps,
        pct_start=0.3 
    )
    
    scaler = torch.amp.GradScaler('cuda')
    sanitized_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    save_path = f"pro_best_{sanitized_name}.pth"
    ds_weights = [1.0, 0.5, 0.25, 0.125] 
    
    # 记录数据的列表
    history_loss = []
    history_lr = []
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(enumerate(train_loader), total=actual_steps, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}")
        
        for step, batch_data in pbar:
            if step >= actual_steps:
                break
                
            inputs = batch_data["image"].as_tensor().to(DEVICE)
            labels = batch_data["label"].as_tensor().to(DEVICE)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                
                if isinstance(outputs, tuple) or isinstance(outputs, list):
                    loss = 0.0
                    for i, out in enumerate(outputs):
                        out_resized = F.interpolate(out, size=labels.shape[2:], mode="trilinear", align_corners=False)
                        loss += ds_weights[i] * loss_function(out_resized, labels)
                else:
                    loss = loss_function(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}"})
        
        # 记录每轮平均 Loss 和 LR
        history_loss.append(epoch_loss / actual_steps)
        history_lr.append(optimizer.param_groups[0]['lr'])
        
    # ===================== [训练结束，保存模型并评估] =====================
    torch.save(model.state_dict(), save_path)
    print(f"✅ 训练完成，模型已保存至 {save_path}")
    
    # 绘制该模型的曲线
    plot_training_curves(model_name, history_loss, history_lr)
    
    print(f"\n🏆 开始对 {model_name} 进行最终高精度全面评估 (overlap=0.5)...")
    model.eval()
    all_acc, all_pre, all_rec, all_auc, all_dice = [], [], [], [], []
    
    with torch.no_grad():
        for val_data in tqdm(val_loader, desc=f"[{model_name} 终极评估]"):
            val_inputs = val_data["image"].as_tensor().to(DEVICE)
            val_labels = val_data["label"].as_tensor().to(DEVICE)
            with torch.amp.autocast('cuda'):
                val_outputs = sliding_window_inference(val_inputs, (96,96,96), 1, model, overlap=0.5)
            
            probs = torch.sigmoid(val_outputs).cpu().numpy()
            labels_np = val_labels.cpu().numpy()
            acc, pre, rec, auc, dice = compute_metrics(labels_np, probs)
            all_acc.append(acc); all_pre.append(pre); all_rec.append(rec); all_auc.append(auc); all_dice.append(dice)
            
    del model, optimizer, scaler; torch.cuda.empty_cache(); gc.collect()
    
    return {
        "Accuracy": np.mean(all_acc),
        "Precision": np.mean(all_pre),
        "Recall": np.mean(all_rec),
        "AUC": np.mean(all_auc),
        "Dice": np.mean(all_dice)
    }

# ==========================================
# 8. 绘制多模型性能对比柱状图
# ==========================================
def plot_final_comparison(results_dict):
    metrics = ["Accuracy", "Precision", "Recall", "AUC", "Dice"]
    model_names = list(results_dict.keys())
    
    # 提取数据
    model_A_scores = [results_dict[model_names[0]][m] for m in metrics]
    model_B_scores = [results_dict[model_names[1]][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35  
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, model_A_scores, width, label=model_names[0], color='#4C72B0')
    rects2 = ax.bar(x + width/2, model_B_scores, width, label=model_names[1], color='#DD8452')

    ax.set_ylabel('Scores', fontweight='bold')
    ax.set_title('Ultimate Models Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontweight='bold')
    ax.legend()
    
    # 柱状图上添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig('Final_Models_Comparison.png', dpi=300)
    plt.close()
    print("📊 双王对决多指标对比柱状图已生成: Final_Models_Comparison.png")

# ==========================================
# 9. 主程序执行入口
# ==========================================
if __name__ == "__main__":
    train_loader, val_loader = prepare_data()
    final_results = {}
    
    models_to_fight = ["Res-Conv-LSTM", "SegResNetDS (Champion)"]
    
    for m_name in models_to_fight:
        torch.cuda.empty_cache(); gc.collect()
        final_results[m_name] = train_and_eval(m_name, train_loader, val_loader)
            
    print("\n" + "🔥"*20)
    print("      🏆 Pro 级: 巅峰双王终极对决结果 🏆      ")
    print("🔥"*20)
    
    df = pd.DataFrame(final_results).T
    df = df.sort_values(by="Dice", ascending=False)
    print(df.round(4).to_string())
    
    df.to_csv("pro_showdown_ultimate.csv")
    print("\n✅ 结果数据已保存至 'pro_showdown_ultimate.csv'")
    
    plot_final_comparison(final_results)
    print("✅ 所有的曲线图表均已生成在当前目录下！")