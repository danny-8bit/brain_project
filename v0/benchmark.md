# 学术报告：基于 BraTS 2021 数据集的多模态脑肿瘤分割基准测试

## 1. 实验综述 (Experiment Overview)
本实验旨在评估不同深度学习架构在三维医学影像分割任务中的表现。通过在 BraTS 2021 挑战赛数据集上，对五种代表性模型（UNet, SegResNet, R2U-Net, Conv-LSTM, Swin-Unet）进行端到端的横向对比，分析其在捕捉局部细节、全局上下文以及特征迭代演化方面的优劣。

## 2. 对比模型架构说明 (Model Architectures)

本研究选择了涵盖三种主要技术路径的模型：

### 2.1 经典卷积路径 (Convolutional Baselines)
*   **UNet (Baseline)**：作为医学影像分割的基准模型，采用对称的编码器-解码器结构与跳跃连接（Skip Connections），用于提取多尺度空间特征。
*   **SegResNet (Advanced CNN)**：在 UNet 基础上引入了残差连接（Residual Connections）和深度可分离卷积的思想，旨在增强深层网络中的梯度流动，提高模型在复杂体素环境下的鲁棒性。

### 2.2 循环与残差增强路径 (Recurrent & Residual Enhancement)
*   **R2U-Net (Recurrent Residual UNet)**：引入了循环残差卷积单元（RRCNN）。通过在同一层内多次迭代累积特征，该模型能够更有效地提取细微病灶的边界信息，增强了特征的辨别力。
*   **Conv-LSTM (CNN-RNN Hybrid)**：在 UNet 的瓶颈层（Bottleneck）嵌入了 3D ConvLSTM 单元。该模块将中间层特征视为序列进行时空演化建模，通过遗忘门和输入门机制，动态精炼关键肿瘤特征并抑制背景噪声。

### 2.3 视觉 Transformer 路径 (Transformer-based)
*   **Swin-Unet (Swin-UNETR)**：代表了当前最前沿的视觉 Transformer 技术。它利用位移窗口（Shifted Window）自注意力机制，摆脱了传统卷积感受野受限的问题，能够在超大范围（96³-128³ 空间）内建立全局长程依赖关系。

## 3. 实验配置与实施 (Experimental Configuration)

### 3.1 数据处理管线 (Data Pipeline)
*   **输入模态**：模型接收 FLAIR, T1ce, T1, T2 四种 MRI 模态，输入维度为 $(4, 96, 96, 96)$。
*   **预处理**：包括 1.0mm 各向同性重采样（Spacingd）、非零区域强度归一化（NormalizeIntensityd）以及基于肿瘤类别的三通道转换（TC, WT, ET）。
*   **数据采样**：采用正负样本均衡裁剪策略（RandCropByPosNegLabeld），确保模型在有限的 Patch 内学习到充足的肿瘤体素。

### 3.2 训练策略
*   **损失函数 (Loss Function)**：采用 **Dice-CrossEntropy (DiceCELoss)**。Dice 损失优化区域重叠度，交叉熵损失优化像素级分类精度，有效应对类别不平衡问题。
*   **优化器 (Optimizer)**：AdamW (Learning Rate: 1e-4, Weight Decay: 1e-5)，配合混合精度训练 (AMP) 以平衡计算速度与内存效率。
*   **推理模式**：使用滑动窗口推理（Sliding Window Inference），设置重叠率（Overlap）为 0.25，确保大体积 MRI 的无缝分割。

## 4. 评估指标体系 (Evaluation Metrics)

实验采用五维指标全面衡量模型性能：
1.  **Dice Similarity Coefficient (DSC)**：核心指标，衡量预测掩码与真值的体积重合度。
2.  **AUC (Area Under ROC Curve)**：衡量模型在概率预测层面的分类能力。
3.  **Accuracy / Precision / Recall**：从像素分类的角度评估模型的准确性、精确率（减少假阳性）和召回率（减少漏诊）。

## 5. 预期贡献 (Expected Insights)

通过本次横向对比，本研究旨在探讨以下科学问题：
*   **迭代特征提取的价值**：R2U-Net 和 Conv-LSTM 的循环机制是否能显著提升小体积肿瘤（如 ET）的分割精度？
*   **Transformer 的优势界限**：在小样本（30例快速测试）环境下，Swin-UNETR 的全局建模能力是否优于参数量更小的精简版 UNet？
*   **医学专用结构的效率**：SegResNet 在 3D 卷积优化上的表现是否能提供最佳的性能-显存性价比？

---
**可视化产出说明：**
*   **柱状图 (Bar Chart)**：直观对比各模型在五大指标上的均值。
*   **热力图 (Heatmap)**：揭示各指标间的相关性及模型间的细微性能差距。
*   **Dice 排名图**：明确界定在脑肿瘤分割任务中的当前最佳架构（SOTA）。