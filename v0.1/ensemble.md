# 学术方法论：基于动态寻优的 Swin-ResLSTM 双骨干集成分割系统

## 1. 算法综述 (Architectural Synopsis)
本研究提出了一种创新的双骨干集成架构（Ensemble Framework），命名为 **Swin-ResLSTM Ensemble**。该系统旨在融合两种互补的深度学习范式：
1.  **Pro-Res-Conv-LSTM**：利用残差卷积捕捉局部几何细节，并通过卷积长短期记忆网络（ConvLSTM）进行瓶颈层特征的迭代精炼。
2.  **Swin-UNETR**：基于位移窗口视觉 Transformer，专门用于捕捉多模态 MRI 中的全局长程空间依赖。

通过这种集成，模型能够同时兼顾肿瘤边界的局部精确性（CNN 优势）与肿瘤整体形态的全局一致性（Transformer 优势）。

## 2. 核心技术组件 (Core Technical Components)

### 2.1 异构骨干网络设计 (Heterogeneous Backbone Design)
*   **Backbone A (Spatial-Temporal Refinement)**: 采用 ProResUNet 架构，其核心创新在于瓶颈层嵌入了 `ProConvLSTM3D` 单元。通过多步循环（LSTM Steps=3），该分支能对高层语义特征进行非线性演化，有效提升了复杂肿瘤区域（如水肿区域 WT 与核心区域 TC）的区分度。
*   **Backbone B (Global Context Modeling)**: 采用 Swin-UNETR 架构，利用层次化自注意力机制。在 $128 \times 128 \times 128$ 的超大感受野下，该分支为集成系统提供了宏观解剖结构的约束。

### 2.2 动态权重网格寻优引擎 (Dynamic Grid Search Optimization)
系统并未采用传统的简单算术平均，而是设计了一套**自适应概率加权引擎**。
*   **寻优算法**：在验证集子集上执行网格搜索（Grid Search），搜索空间为 $w \in [0.0, 1.0]$，步长为 0.05。
*   **优化目标**：以快速 Dice 系数为目标函数，动态计算模型 A 与模型 B 的最佳融合权重比例 $W_{opt}$。
*   **公式表达**：最终预测概率图 $P_{ensemble}$ 计算如下：
    $$P_{ensemble} = w_{opt} \cdot \sigma(O_A) + (1 - w_{opt}) \cdot \sigma(O_B)$$
    其中 $\sigma$ 为 Sigmoid 激活函数，$O_A, O_B$ 分别为两个骨干网络的 Logits 输出。

## 3. 推理与后处理流程 (Inference & Post-processing)

### 3.1 满血版大视野推理
为了保证特征的连贯性，系统在推理阶段统一采用了 **$128^3$ 超大感受野** 的滑动窗口推理（Sliding Window Inference），并设置重叠度（Overlap）为 0.5。这显著降低了切片边缘的伪影现象。

### 3.2 鲁棒性精炼 (Post-refinement)
为了模拟临床放射科专家的诊断一致性，系统集成了以下后处理步骤：
*   **连通域分析 (Connected Component Analysis)**：强制提取最大连通区域，有效过滤了由于场强不均或伪影导致的零星假阳性像素。
*   **病灶体积约束**：针对增强肿瘤（ET）设定了最小体积先验阈值（V < 50 像素归零），提高了模型在极端类别不平衡下的 Precision 指标。

## 4. 实验评估体系 (Evaluation Framework)

系统通过五大核心维度对集成模型进行全量扫描评估：
1.  **Dice Similarity Coefficient (DSC)**：衡量分割区域重合度。
2.  **Accuracy / Precision / Recall**：评估像素级分类的准确性、精确率与召回率。
3.  **AUC (Area Under ROC Curve)**：评估模型在概率层面上的分类辨别能力。

## 5. 技术参数汇总 (Technical Summary)

| 核心维度 | 技术配置 |
| :--- | :--- |
| **集成模式** | 动态加权概率融合 (Dynamic Probability Ensemble) |
| **骨干 A** | 3D-ResNet + 3D-ConvLSTM (init_filters=32) |
| **骨干 B** | Swin Transformer (feature_size=48) |
| **寻优策略** | 验证集离线网格扫描 (Grid Search on Val-set) |
| **感受野** | 128 x 128 x 128 (Full Patch) |
| **后处理** | Largest Connected Component + Volume Thresholding |
| **可视化** | Seaborn 驱动的指标分布分析与权重寻优曲线 |