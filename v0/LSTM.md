# 技术报告：基于 Pro-ResUNet-ConvLSTM 3D 的多模态脑胶质瘤分割系统

## 1. 算法架构总览 (Architectural Overview)
本研究提出了一种结合残差连接（Residual Connection）与卷积长短期记忆网络（ConvLSTM）的深度 3D 编码器-解码器架构——**Pro-ResUNet-ConvLSTM 3D**。该架构专为 BraTS 2021 挑战赛中的多模态磁共振成像（MRI）分割任务设计，旨在通过时空特征融合增强对肿瘤核心（TC）、整体肿瘤（WT）和增强肿瘤（ET）的捕捉能力。

### 1.1 核心组件设计
*   **Pro-ResBlock3D (残差块)**：每个编码/解码层均采用改进的残差模块。通过引入 `InstanceNorm3d` 和 `LeakyReLU` 激活函数，模型能够有效缓解深度网络中的梯度消失问题，并加速模型收敛。
*   **Pro-ConvLSTM3D 瓶颈层**：在 U-Net 架构的最底层（Bottleneck）引入了 3D ConvLSTM 单元。不同于传统的静态特征提取，ConvLSTM 通过迭代循环机制（`lstm_steps=2`）在特征空间内进行非线性精炼，捕获更深层的前后向空间依赖关系。
*   **Deep Supervision (深度监督)**：模型在解码器的不同尺度输出端均设有预测头（out1, out2, out3）。在训练阶段，通过对多尺度预测结果进行加权损失计算，引导模型学习更具鲁棒性的层次特征。

## 2. 训练策略与参数配置 (Training Strategy & Hyperparameters)

### 2.1 损失函数设计
为了应对脑肿瘤分割中极端的类别不平衡问题，系统采用了复合损失函数 **DiceFocalLoss**。
*   **Dice Loss**：关注预测掩码与真实标签的重叠度，优化全局结构。
*   **Focal Loss**：通过调整难易样本的权重，使模型更专注于难以分割的肿瘤边界和细小病灶。
*   **加权方案**：针对 BraTS 特定类别进行了加权处理（TC: 1.5, WT: 1.0, ET: 2.5），显著提升了高难度区域（如增强肿瘤 ET）的精度。

### 2.2 优化器与调度
*   **优化算法**：使用 `AdamW` 优化器，设置权重衰减（Weight Decay）为 $1 \times 10^{-4}$ 以防止过拟合。
*   **学习率策略**：采用 `OneCycleLR` 调度器。该策略通过“热身-退火”机制，在训练初期快速寻找最优解空间，后期平稳收敛。
*   **计算优化**：
    *   **混合精度训练 (AMP)**：利用 `torch.amp` 提升显存效率及计算速度。
    *   **梯度累积 (Gradient Accumulation)**：通过设置 `GRAD_ACCUM_STEPS=4`，在小批量（Batch Size=1）的情况下模拟大批量训练的稳定性。

### 2.3 数据增强与预处理
基于 **MONAI** 框架构建了高性能数据管线：
*   **强度归一化**：针对 MRI 非零区域进行 Z-Score 归一化。
*   **几何变换**：包括随机 3D 旋转、三轴翻转、随机缩放及平移，增强模型的平移/旋转不变性。
*   **采样策略**：采用 `RandCropByPosNegLabeld` 保证训练切片中正负样本比例为 3:1，确保模型能学习到足够的肿瘤特征。

## 3. 核心算法参数表 (Key Parameters)

| 参数名称 | 数值 | 说明 |
| :--- | :--- | :--- |
| 输入维度 (Input Channels) | 4 | FLAIR, T1ce, T1, T2 四种模态 |
| 输出维度 (Output Channels) | 3 | TC, WT, ET (Sigmoid 激活) |
| 初始滤波器数 (Init Filters) | 32 | 编码器首层通道数 |
| 块大小 (Patch Size) | (128, 128, 128) | 3D 空间裁剪尺寸 |
| 迭代步数 (LSTM Steps) | 2 | ConvLSTM 循环精炼次数 |
| 训练周期 (Max Epochs) | 50 | 完整数据集迭代次数 |
| 优化器 (Optimizer) | AdamW | 学习率 3e-4, 带权重衰减 |

## 4. 后处理与评估指标 (Post-processing & Metrics)

### 4.1 评估指标
系统通过以下指标对模型进行全方位评估：
*   **Dice Similarity Coefficient (DSC)**：评估分割精度。
*   **Accuracy / Precision / Recall**：评估像素级分类准确度。

### 4.2 鲁棒性增强
*   **最大连通域提取 (Keep Largest Connected Component)**：在预测阶段过滤孤立的假阳性噪点。
*   **类别相关性约束**：根据 BraTS 逻辑强制约束分割层次（如：ET 必须是 TC 的子集，TC 必须是 WT 的子集），纠正解剖学不合理的预测。

## 5. 原理总结
本模型通过 **"残差结构保证深度 + ConvLSTM 提取时空特征 + 深度监督精炼多尺度特征"** 的组合拳，解决了 3D 医学影像中病灶形态多变、边界模糊、类别失衡的三大痛点。该方案在保持较低计算开销的同时，实现了高精度的自动化肿瘤分割。