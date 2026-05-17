import torch
import torch.nn as nn
import numpy as np
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from typing import Tuple

######################################################################
# 1. 多模态 Cross-Attention 融合模块
######################################################################
class MultimodalCrossAttention(nn.Module):
    """
    针对 BraTS 的 4 种模态 (T1, T1c, T2, FLAIR) 分别提取初步特征，
    然后使用 Cross-Attention (MultiHeadAttention) 对它们进行信息交互融合。
    """
    def __init__(self, in_channels=1, embed_dim=48, num_heads=4):
        super().__init__()
        # 对 4 个模态 (T1, T1c, T2, FLAIR) 分别使用一个轻量卷积进行 Embedding
        self.modal_stems = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels, embed_dim, kernel_size=3, padding=1),
                nn.InstanceNorm3d(embed_dim),
                nn.LeakyReLU(inplace=True)
            ) for _ in range(4)
        ])
        
        # 多头注意力机制
        self.cross_modal_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        # 将 [B, 4, C, D, H, W] 的融合特征投影回 [B, embed_dim, D, H, W] 交给主干网络
        self.proj = nn.Conv3d(embed_dim * 4, embed_dim, kernel_size=1)
        
    def forward(self, t1, t1c, t2, flair):
        # features[i] 形状: (B, C, D, H, W)
        features = [
            self.modal_stems[0](t1),
            self.modal_stems[1](t1c),
            self.modal_stems[2](t2),
            self.modal_stems[3](flair)
        ]
        
        B, C, D, H, W = features[0].shape
        
        # 将空间维度拍平，准备进行 Attention 处理
        # -> (B, 4, C, L) where L = D*H*W
        flattened_feats = torch.stack([f.view(B, C, -1) for f in features], dim=1)
        # 转置和重塑为 Attention 期待的形状: (B * L, 4, C)
        flattened_feats = flattened_feats.permute(0, 3, 1, 2).reshape(-1, 4, C)
        
        # 自注意力/交叉注意力
        attn_out, _ = self.cross_modal_attn(flattened_feats, flattened_feats, flattened_feats)
        
        # 还原回 3D 空间特征图形状: (B, 4, C, L) -> (B, 4*C, D, H, W)
        attn_out = attn_out.reshape(B, -1, 4, C).permute(0, 2, 3, 1) # B, 4, C, L
        concat_feats = attn_out.reshape(B, 4 * C, D, H, W)
        
        # 投影到单一分支用于后续 UNETR 处理
        out = self.proj(concat_feats)
        return out


######################################################################
# 2. 具备多任务分类头的高级分割网络 (Swin-UNETR + Classification)
######################################################################
class AdvancedBraTSMultiTaskNet(nn.Module):
    """
    集成了 Cross-Attention 的多模态融合，Swin-UNETR 主干分割，
    以及深层特征引导的 ROI 分类头（用于肿瘤分级、IDH突变预测等）。
    """
    def __init__(self, img_size=(96, 96, 96), num_classes_seg=3, num_classes_roi=3, embed_dim=48):
        super().__init__()
        
        self.modal_fusion = MultimodalCrossAttention(in_channels=1, embed_dim=embed_dim)
        
        # 内置 SwinUNETR
        self.backbone = SwinUNETR(
            img_size=img_size,
            in_channels=embed_dim, # 注意此处接收融合后的单特征流
            out_channels=num_classes_seg, # 预测 WT, TC, ET 的 logits
            feature_size=embed_dim,
            use_checkpoint=True,
            drop_rate=0.1
        )
        
        # 提取最深层的特征。对于 feature_size=48，深层特征通常为 48 * 2**4 = 768 或更大。
        # SwinUNETR 的最深处特征图大小较小，我们用 AdaptiveAvgPool3d 转为 1D 向量
        self.bottleneck_dim = embed_dim * 16  # 对于 SwinUNETR 通常缩放 16 倍
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        
        # 分类器头（比如 3 对应: 1=IDH_status, 2=MGMT_status, 3=Grade_LGG_HGG）
        self.roi_classifier = nn.Sequential(
            nn.Linear(self.bottleneck_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes_roi)
        )

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x 为包含4个模态的张量 (B, 4, D, H, W)
        """
        t1, t1c, t2, flair = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
        
        # 1. Attention 融合
        fused_in = self.modal_fusion(t1, t1c, t2, flair)
        
        # 2. Backbone 分割前向 (我们需要 Hack 一下深入获取 hidden_states，
        # MONAI 的 SwinUNETR 内部调用 swinBiTR，我们以其隐藏输出获取 bottleneck)
        hidden_states_out = self.backbone.swinViT(fused_in, self.backbone.normalize)
        enc0 = self.backbone.encoder1(fused_in)
        enc1 = self.backbone.encoder2(hidden_states_out[0])
        enc2 = self.backbone.encoder3(hidden_states_out[1])
        enc3 = self.backbone.encoder4(hidden_states_out[2])
        dec4 = self.backbone.encoder10(hidden_states_out[4]) # bottleneck
        
        # Decoder 过程 (此处调用其自带解码块简化展示，实际可以直接 self.backbone(fused_in))
        # 但为了拿到 dec4，上面拆解了。为了代码稳健性，我们可以直接进行预测。
        seg_output = self.backbone(fused_in)
        
        # 3. 提取全局深层特征
        # dec4 形状约为 (B, 768, D/32, H/32, W/32)
        pooled_feat = self.avg_pool(dec4).squeeze(-1).squeeze(-1).squeeze(-1)
        
        # 4. 预测分子标志物和分级
        cls_output = self.roi_classifier(pooled_feat)
        
        return seg_output, cls_output


######################################################################
# 3. 自监督预训练 (Masked Image Modeling) 的 Wrapper 示例
######################################################################
class SSLMaskedPretrainer(nn.Module):
    def __init__(self, backbone, mask_ratio=0.3):
        super().__init__()
        self.backbone = backbone
        self.mask_ratio = mask_ratio
        
    def forward(self, x):
        # 简单的随机 Mask 代码逻辑示意
        B, C, D, H, W = x.shape
        mask = torch.rand((B, 1, D, H, W), device=x.device) < self.mask_ratio
        x_masked = x.clone()
        x_masked[mask.expand_as(x)] = 0.0 # 被遮挡用 0 填充
        
        # 将遮挡后的图像送入主干特征提取与重构
        reconstructed = self.backbone(x_masked) 
        
        return reconstructed, mask


######################################################################
# 4. 高级模型集成 (加权集成概率并后处理)
######################################################################
def advanced_weighted_ensemble(preds_dict, weights_dict):
    """
    preds_dict: {'nnunet': (C, D, H, W) tensor, 'swin': (...), 'lstm': (...)}
    weights_dict: {'nnunet': 0.5, 'swin': 0.3, 'lstm': 0.2}
    """
    ensemble_prob = 0.0
    for key in preds_dict:
        ensemble_prob += preds_dict[key] * weights_dict[key]
        
    # BraTS 通常有 3 个通道: WT, TC, ET (可能通过 sigmoid 输出，非 softmax互斥)
    # 若模型输出经过 sigmoid (代表概率 [0,1])
    pred_seg = np.zeros_like(ensemble_prob[0].cpu().numpy(), dtype=np.uint8)
    
    wt_prob = ensemble_prob[0].cpu().numpy()
    tc_prob = ensemble_prob[1].cpu().numpy()
    et_prob = ensemble_prob[2].cpu().numpy()
    
    # 阈值决策与优先级 (ET > TC > WT)
    pred_seg[wt_prob > 0.5] = 2  # WT 对应标签 2 (Edema, 组合起来是 WT)
    pred_seg[tc_prob > 0.5] = 1  # TC 对应标签 1 (Necrosis & Non-enhancing) 
    pred_seg[et_prob > 0.5] = 3  # ET 对应标签 4 (此处简写用 3，按官方竞赛具体映射，常见是 4 为增强)
    
    # 后处理: 比如清除总体积 < 50 像素的 ET
    et_mask = (pred_seg == 3)
    if np.sum(et_mask) < 50:
        # 如果增强肿瘤过小，可能是假阳性，用非增强(TC)替代
        pred_seg[et_mask] = 1 
        
    return pred_seg

if __name__ == "__main__":
    # 功能单元测试
    print("Testing Architecture Build...")
    model = AdvancedBraTSMultiTaskNet(img_size=(96, 96, 96), num_classes_seg=3, num_classes_roi=3)
    
    # 模拟 [Batch, Modality=4, D, H, W]
    dummy_input = torch.randn(2, 4, 96, 96, 96)
    
    seg_logits, cls_logits = model(dummy_input)
    print("Input Shape:", dummy_input.shape)
    print("Seg Output Shape:", seg_logits.shape) # 应为 (2, 3, 96, 96, 96)
    print("Classification Output Shape:", cls_logits.shape) # 应为 (2, 3)
    print("Architecture built successfully!")
