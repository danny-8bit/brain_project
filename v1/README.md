# BraTS2021 脑胶质瘤分割 SOTA 训练框架

下面是一个完整的、面向比赛冠军和论文级成果的训练框架。

## 框架核心 SOTA 技术

1. **Region-based 预测**：直接预测 WT/TC/ET 三个区域（sigmoid），这是 BraTS 冠军方案的标配
2. **Deep Supervision**：多尺度监督，加快收敛、提升精度
3. **5-Fold 交叉验证**：标准做法
4. **多模型异构融合**：nnU-Net v2 + SegResNet + SwinUNETR + (可选 MedNeXt)
5. **EMA (Exponential Moving Average)**：稳定提分
6. **强数据增强**：spatial + intensity + modality dropout
7. **TTA**：测试时翻转增强
8. **滑窗高斯加权推理**
9. **ET 阈值后处理**：BraTS 提分关键
10. **AMP 混合精度训练**

请参考其中的 Python 脚本和配置文件使用该框架。