# 模型说明（结构、输入输出、训练目标）

本文档聚焦模型本身的设计与实现细节，包括输入输出定义、结构配置、损失与监督方式，以及和模型相关的推理与后处理约束。

## 任务输出与标签编码

- **模型输出**：3 通道概率图（TC/WT/ET），每通道使用 Sigmoid。
- **BraTS 标签还原**：最终输出 `{0,1,2,4}` 的标签体积。

区域定义如下：

- **TC** = NCR/NET(1) ∪ ET(4)
- **WT** = NCR/NET(1) ∪ ED(2) ∪ ET(4)
- **ET** = ET(4)

标签三通道化由 [src/data/transforms.py](src/data/transforms.py) 中的 `ConvertToMultiChannelBratsd` 完成，确保训练监督和模型输出严格对齐。

## 模型输入与前处理约束

- **输入通道**：4 通道（T1、T1ce、T2、FLAIR）。
- **空间约束**：统一到 RAS 方向，spacing 固定为 1mm。
- **拼接方式**：4 个模态在通道维 concat 成 `image`（shape = `(4, D, H, W)`）。

上述处理由 [src/data/transforms.py](src/data/transforms.py) 统一实现，模型侧默认接收 4 通道输入。

## 模型构建与结构细节

模型入口为 [src/models/builder.py](src/models/builder.py)，通过 `model` 字段选择具体结构。

### SegResNet / SegResNetDS

- **优先使用 SegResNetDS**（若 MONAI 提供）。
- **结构配置**：
  - `init_filters`: 默认 32（见 [configs/segresnet.yaml](configs/segresnet.yaml)）。
  - `blocks_down`: `(1, 2, 2, 4, 4)`（SegResNetDS）。
  - `dsdepth`: 4（多尺度输出）。
  - 若没有 SegResNetDS，则退化为 SegResNet：
    - `blocks_down`: `(1, 2, 2, 4)`
    - `blocks_up`: `(1, 1, 1)`
    - `dropout_prob`: 默认 0.2

### SwinUNETR

- **结构**：Swin Transformer 编码器 + UNet 解码器（3D）。
- **关键参数**：
  - `feature_size`: 默认 48（见 [configs/swinunetr.yaml](configs/swinunetr.yaml)）。
  - `use_checkpoint`: 默认 true，降低显存。
  - `img_size`: 使用训练 ROI（默认 128^3）。
- **预训练**：支持 SSL 预训练权重（`ssl_pretrained`）。

### MedNeXt（可选）

- 通过 `nnunet_mednext.create_mednext_v1` 构建。
- 默认配置：`model_id="B"`, `kernel_size=3`, `deep_supervision=True`。
- 需额外安装 MedNeXt 依赖（见 [src/models/builder.py](src/models/builder.py)）。

## 深监督与输出形式

- **单输出模型**：直接输出 `(B, 3, D, H, W)`。
- **深监督模型**：输出多个尺度张量（列表或元组），第 0 个为全分辨率。
- 训练时由 [src/losses.py](src/losses.py) 中的 `DeepSupervisionLoss` 自动处理：
  - 标签会按尺度下采样。
  - 权重按 `1, 1/2, 1/4, ...` 归一化求和。

验证和推理阶段始终使用最高分辨率输出（见 [src/trainer.py](src/trainer.py)）。

## 损失函数与监督目标

损失为 Dice + BCE 的多标签组合：

$$
L = \lambda_{dice} L_{dice} + \lambda_{bce} L_{bce}
$$

- `L_dice` 与 `L_bce` 都按通道独立计算并加权。
- 通道权重 `loss_channel_weights` 默认为 `[1, 1, 1]`（可提高 ET 权重）。
- 实现位于 [src/losses.py](src/losses.py)。

## 关键训练超参（模型相关）

默认全局配置来自 [configs/base.yaml](configs/base.yaml)。不同模型在各自配置中覆盖超参。

- **ROI**：`128 x 128 x 128`
- **Batch size**：
  - SegResNet: 2
  - SwinUNETR: 2
  - MedNeXt: 1
- **学习率**：
  - SegResNet: `3e-4`
  - SwinUNETR: `1e-4`
  - MedNeXt: `1e-4`
- **AMP**：默认启用。
- **EMA**：默认启用（`ema_decay=0.999`）。
- **梯度裁剪**：`grad_clip=1.0`。

具体值以 [configs/segresnet.yaml](configs/segresnet.yaml)、[configs/swinunetr.yaml](configs/swinunetr.yaml)、[configs/mednext.yaml](configs/mednext.yaml) 为准。

## 推理输出与概率格式

推理由 [scripts/predict.py](scripts/predict.py) 完成，输出概率图为：

- `prob` 形状 `(3, D, H, W)`。
- `uint8` 存储，范围 `0~255`（对应 `0~1` 概率）。

若启用 TTA（见 [src/tta.py](src/tta.py)），会对 8 种翻转进行平均，以提高模型鲁棒性。

## 后处理与标签还原规则

后处理位于 [src/postprocess.py](src/postprocess.py)，用于将概率图稳定转成 BraTS 标签：

1. **阈值化**（支持 TC/WT/ET 独立阈值）。
2. **层级一致性**：ET 必须在 TC 内，TC 必须在 WT 内。
3. **ET 假阳性抑制**：若 ET 体素数过小且最大概率低，则清空 ET。
4. **小连通域过滤**：可按 region 设置 `min_sizes`。

这些规则直接约束模型输出的可用性，因此在模型评估时应保持与该后处理一致的参数。

## 模型配置入口速查

- 模型构建与结构分支：[src/models/builder.py](src/models/builder.py)
- 深监督与损失：[src/losses.py](src/losses.py)
- 训练/验证输出处理：[src/trainer.py](src/trainer.py)
- 数据与标签编码：[src/data/transforms.py](src/data/transforms.py)
- 关键超参：
  - [configs/base.yaml](configs/base.yaml)
  - [configs/segresnet.yaml](configs/segresnet.yaml)
  - [configs/swinunetr.yaml](configs/swinunetr.yaml)
  - [configs/mednext.yaml](configs/mednext.yaml)