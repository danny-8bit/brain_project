
---

# 🧠 Intelligent-Detection-of-Brain-Glioma-Based-on-MRI
**基于 BraTS 2021 数据集的 3D 脑肿瘤分割与多模型横向深度评测项目**

本项目使用 PyTorch 和 MONAI 框架，针对 **BraTS 2021** (脑肿瘤多模态 MRI) 数据集，构建了一套完整的 3D 医疗影像分割流水线。项目不仅包含了经典与前沿的 5 款 3D 分割模型横向对比，还针对 **Swin-UNETR** (视觉 Transformer) 和 **满血版 Pro-Res-Conv-LSTM** (带深层监督的 3D 循环残差网络) 编写了专属的高精度特训脚本。

---

## ✨ 项目特色

- 🏥 **多模态输入 & 多标签输出**：支持 4 模态 MRI (FLAIR, T1ce, T1, T2) 并行输入，自动解析并预测三大核心指标标签：`WT` (完整肿瘤)、`TC` (肿瘤核心)、`ET` (增强肿瘤)。
- 🚀 **极致的显存优化 (适配 16GB 显卡)**：利用 `torch.amp` 混合精度训练、`Gradient Accumulation` (梯度累加，等效大 Batch Size)、`use_checkpoint` 以及动态 `gc.collect()` 垃圾回收，让 RTX 5070 Ti / 4080 等 16G 显存设备也能顺畅运行 128x128x128 视野的 3D Transformer。
- 📊 **高度自动化的成果管理**：自带完善的目录路由，无需手动建文件，一键运行即可自动生成带有美观样式的训练曲线 (.png)、评估热力图 (.png) 和量化指标表格 (.csv)。
- 🔬 **前沿架构验证**：包含了自定义编写的 **Pro-Res-Conv-LSTM** 网络，融入了残差模块 (Residual)、3D 卷积长短时记忆单元 (Conv-LSTM) 以及**深层监督机制 (Deep Supervision)**，提升多尺度特征融合能力。

---

## 📁 目录结构

项目运行前，请确保数据集按照如下结构放置。代码首次运行时会**自动创建** `model/` 和 `result/` 文件夹，无需手动干预。

```text
├── data/
│   └── BraTS2021/
│       ├── BraTS2021_00000/
│       │   ├── BraTS2021_00000_flair.nii.gz
│       │   ├── BraTS2021_00000_t1ce.nii.gz
│       │   ├── BraTS2021_00000_t1.nii.gz
│       │   ├── BraTS2021_00000_t2.nii.gz
│       │   └── BraTS2021_00000_seg.nii.gz
│       ├── BraTS2021_00002/
│       └── ... (其他患者数据)
├── model/                     # 🚀 自动创建：存放所有训练好的 .pth 模型权重
├── result/                    # 📊 自动创建：存放图表 (.png) 和评估表格 (.csv)
├── compare_5_models.py        # 核心脚本 1：五大模型横向对比程序
├── train_swin_unetr.py        # 核心脚本 2：Swin-UNETR 专属特训程序
├── train_pro_lstm.py          # 核心脚本 3：Pro-Res-Conv-LSTM 专属特训程序
└── README.md
```

---

## 🛠️ 环境依赖

推荐使用 Python 3.8+ 及 PyTorch 2.0+。请安装以下核心依赖库：

```bash
pip install torch torchvision
pip install monai
pip install pandas numpy scikit-learn
pip install matplotlib seaborn tqdm
pip install nibabel # MONAI 加载 nii.gz 格式必备
```

---

## 🚀 核心脚本与使用指南

本项目包含三个独立的运行脚本，可以根据你的研究需求分别执行：

### 1. 五大模型横向对比评测 (`compare_5_models.py`)
该脚本会对 `UNet`, `Swin-Unet`, `SegResNet`, `R2U-Net`, `Conv-LSTM` 五个模型进行轮流训练和统一标准的验证，并在结束后生成直观的对比报告。
```bash
python compare_5_models.py
```
**✨ 产出结果：**
- `model/*.pth`：5 个模型的权重文件。
- `result/brats_ultimate_comparison.csv`：涵盖 Accuracy, Precision, Recall, AUC, Dice 指标的详细 CSV 成绩单。
- `result/brats_models_bar_chart.png`：五大模型的综合指标分组柱状图。
- `result/brats_models_heatmap.png`：性能指标热力图。
- `result/brats_models_dice_chart.png`：医学分割最关心的 Dice 得分独立排序图。

### 2. Swin-UNETR 专属特训 (`train_swin_unetr.py`)
针对 Vision Transformer 架构编写的高性能特训脚本，带有独立的数据增强增强 (Flip, Rotate, Scale, Shift) 以及 `OneCycleLR` 学习率调度策略，自带预测结果的**最大连通域 (Largest Connected Component)** 后处理去噪。
```bash
python train_swin_unetr.py
```
**✨ 产出结果：**
- `model/swin_unetr_standalone.pth`：Transformer 模型权重。
- `result/Swin_UNETR_Dynamics.png`：双轴绘制的 Training Loss 与 Learning Rate 动态曲线图。
- `result/swin_unetr_standalone_results.csv`：高精度评估最终成绩单。

### 3. 满血版 Pro-Res-Conv-LSTM 特训 (`train_pro_lstm.py`)
该脚本运行的是本项目自主设计的融合网络结构。结合了 4 层深度编码器、LSTM 循环记忆中枢，并在解码器阶段接入了 `out1, out2, out3` 的**多尺度深层监督探头 (Deep Supervision)**。
```bash
python train_pro_lstm.py
```
**✨ 产出结果：**
- `model/pro_res_conv_lstm_standalone.pth`：Pro-LSTM 终极模型权重。
- `result/Pro_LSTM_Dynamics.png`：深层监督损失与学习率动态曲线图。
- `result/pro_lstm_standalone_results.csv`：最终评估成绩单。

---

## ⚙️ 超参数调整建议 (配置调优)

在每个脚本的顶部，都预留了 `全局配置区`。如果你的硬件配置不同或想进行完整投产训练，可修改以下参数：
- `MAX_EPOCHS`：默认设为 `50` 用于极速测试。如需达到论文级别的 SOTA 精度，建议修改为 `150` 或 `300`。
- `TRAIN_BATCH_SIZE`：物理 Batch Size 默认为 `1`。
- `GRAD_ACCUM_STEPS`：梯度累加步数默认为 `4`（等效 Batch Size = 4）。如果显存达到 24GB (如 RTX 3090/4090)，可以尝试物理 Batch Size = 2，累加步数 = 2。

---

## 📝 评估指标说明

本项目在验证阶段统一计算以下 5 个指标。计算时针对每个体素 (Voxel) 进行统计：
1. **Dice Score**：医学图像分割核心评价指标，衡量预测与真实标签的体积重叠度。
2. **Accuracy**：全局像素准确率。
3. **Precision / Recall**：精准率与召回率。
4. **AUC (ROC-AUC)**：评估模型对正负样本的分类排序能力。

---

## 🤝 鸣谢与协议
- 数据集来源：[RSNA-ASNR-MICCAI BraTS 2021 Challenge](http://braintumorsegmentation.org/)
- 核心框架支持：[Project MONAI](https://monai.io/) (Medical Open Network for AI)
- 遵循 [MIT License](LICENSE) 开源协议。