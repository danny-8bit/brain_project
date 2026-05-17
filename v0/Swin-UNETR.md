# 技术报告：基于 Swin-UNETR 的多模态脑胶质瘤 3D 分割算法

## 1. 算法架构：Swin-UNETR (Swin Transformer Embedding)
本研究采用了 **Swin-UNETR** 架构，这是一种将层次化视觉 Transformer（Swin Transformer）作为编码器并结合对称 CNN 解码器的混合模型。

### 1.1 编码器（Encoder）
*   **层次化特征提取**：编码器通过位移窗口（Shifted Window）机制处理 3D 图像块（Patches），在不同尺度上捕获长程依赖关系（Long-range Dependencies）。
*   **自注意力机制**：相比于传统卷积神经网络（CNN），Transformer 的自注意力机制允许模型学习全局上下文信息，这对于确定肿瘤的宏观边界至关重要。

### 1.2 解码器与跳跃连接（Decoder & Skip Connections）
*   **多尺度融合**：解码器通过跳跃连接接收编码器不同阶段的特征图，将 Transformer 提取的语义信息与 CNN 提取的空间细节进行融合。
*   **Spatio-Temporal Consistency**：在 3D 空间（$128 \times 128 \times 128$）内进行操作，保证了切片间（Inter-slice）的解剖结构连续性。

## 2. 核心技术特性 (Core Technical Enhancements)

### 2.1 硬件级加速优化
为了提升在大型 3D 体素上的训练效率，代码集成了多项计算优化技术：
*   **TF32 加速**：启用 `allow_tf32`，在 NVIDIA Ampere 架构（及后续架构）上利用 TensorFloat-32 提供接近于 FP32 的精度，同时获得数倍的计算性能。
*   **cuDNN Auto-tuner**：通过 `cudnn.benchmark = True` 自动为当前硬件配置寻找最优卷积算法。
*   **混合精度训练 (AMP)**：利用 `torch.amp` 减少显存占用并加速梯度反向传播。

### 2.2 测试时增强 (Test-Time Augmentation, TTA)
在评估阶段，本算法实现了一套空间翻转 TTA 流程：
*   对输入体积在三个空间维度上分别进行水平/垂直翻转并进行多次推理。
*   通过对多次预测结果取均值（Averaging Ensemble），显著降低了预测方差，实验表明该方法能提升约 1%~2% 的 Dice 系数。

### 2.3 鲁棒性后处理
*   **连通域分析 (CC Analysis)**：利用 `KeepLargestConnectedComponent` 消除预测掩码中的孤立虚假阳性病灶。
*   **阈值过滤**：针对 BraTS 2021 数据集的特性，设定了最小体积阈值（如 ET 区域少于 73 像素归零），模拟放射科医生的诊断逻辑，降低过拟合风险。

## 3. 损失函数与训练配置

### 3.1 混合损失函数 (DiceCELoss)
采用 **Dice Loss** 与 **Cross Entropy Loss** 的加权组合。
*   **Dice Loss**：直接优化预测区域与真实区域的重叠率。
*   **Cross Entropy**：提供稳定的像素级分类梯度，辅助模型处理边界不确定性。

### 3.2 学习率调度策略
*   **Warmup Cosine Schedule**：初始阶段进行 5 个 Epoch 的热身（Warmup），随后进入余弦退火阶段。这种策略能有效防止模型在初始化阶段因梯度过大而震荡，并确保后期平稳收敛。

## 4. 关键参数设置 (Technical Parameters)

| 类别 | 参数 | 数值/设置 |
| :--- | :--- | :--- |
| **模型设置** | 架构 | Swin-UNETR (Feature Size=48) |
| | 空间维度 | 3D (128x128x128) |
| **数据流** | 输入模态 | 4 模态 (FLAIR, T1ce, T1, T2) |
| | 增强方法 | 旋转, 翻转, 强度缩放, 高斯噪声, 对比度调节 |
| **训练优化** | 优化器 | AdamW (Weight Decay: 1e-5) |
| | 梯度累积 | 4 Steps (模拟 Batch Size = 4) |
| | 调度器 | Warmup Cosine Schedule |
| **推理技巧** | 重叠率 (Overlap) | 0.6 (Sliding Window) |
| | 后处理 | TTA (3-axis flips) + 最大连通域提取 |

## 5. 评估指标
系统通过以下四个核心维度评估 Swin-UNETR 的泛化性能：
1.  **Dice Similarity Coefficient (DSC)**：衡量重叠度（核心指标）。
2.  **Accuracy / Precision / Recall**：评估像素分类性能。
3.  **AUC (Area Under Curve)**：通过对采样 50 万像素进行概率分析，衡量模型的区分能力。