# BraTS2021 脑胶质瘤分割训练与融合框架

本项目是一个面向 BraTS2021 脑胶质瘤分割任务的完整训练与推理框架，覆盖数据预处理、单模型训练、5-fold 交叉验证、TTA 推理、模型融合、后处理与论文评估。

## 项目定位与任务

- **数据**：BraTS2021 训练集，每个病例包含 4 个模态（T1、T1ce、T2、FLAIR）与分割标签。
- **目标**：采用 **region-based** 预测方式，输出 3 通道概率图：
  - TC = NCR/NET(1) ∪ ET(4)
  - WT = NCR/NET(1) ∪ ED(2) ∪ ET(4)
  - ET = ET(4)
- **标签输出**：最终输出 BraTS 原始标签体积 `{0,1,2,4}`。

## 核心特性

- **Region-based 预测**（3 通道 Sigmoid）
- **Deep Supervision**（多尺度监督）
- **EMA + AMP**（稳定训练，提升效率）
- **5-Fold 交叉验证** + **OOF 融合**
- **强数据增强**（空间 + 强度 + 模态随机 dropout）
- **滑窗推理 + TTA 翻转**（8 种翻转组合）
- **ET 假阳性后处理** 与层级一致性约束
- **融合权重 + 阈值网格搜索**

## 模型与损失

### 模型列表

- **SegResNet / SegResNetDS**（MONAI）
  - 如果环境支持 `SegResNetDS`，自动启用 deep supervision。
  - 默认 `init_filters=32`，可在 configs 中调整。
- **SwinUNETR**（MONAI）
  - 支持 `use_checkpoint` 减少显存。
  - 支持 SSL 预训练权重（`configs/swinunetr.yaml`）。
- **MedNeXt（可选）**
  - 需安装 `mednextv1` 或克隆官方仓库。
  - 当前通过 `nnunet_mednext.create_mednext_v1` 构建。

### 损失函数

- `DiceBCELoss = Dice + BCE`，按通道加权。
- `DeepSupervisionLoss` 对多尺度输出自动下采样标签并加权聚合。

## 数据与预处理

### 数据结构（BraTS2021 训练集）

```
BraTS2021_TrainingData/
  BraTS2021_00000/
    BraTS2021_00000_t1.nii.gz
    BraTS2021_00000_t1ce.nii.gz
    BraTS2021_00000_t2.nii.gz
    BraTS2021_00000_flair.nii.gz
    BraTS2021_00000_seg.nii.gz
```

### 预处理与增强

- **空间规范**：统一到 RAS 方向，spacing=1mm。
- **输入拼接**：4 模态 concat 成 4 通道 `image`。
- **增强策略**：随机裁剪、翻转、强度缩放/平移、噪声、模糊、对比度、模态 dropout。
- **标签转换**：`{0,1,2,4}` → 3 通道 `[TC, WT, ET]`。

## 训练与验证流程

- `scripts/train.py` 合并基础配置与模型配置，启动单 fold 训练。
- `src/trainer.py` 实现训练闭环：
  - `CacheDataset + DataLoader` 数据加载
  - Warmup + Cosine 学习率调度
  - AMP 混合精度
  - EMA 权重平均
  - 滑窗验证 + BraTS Dice 指标
  - 保存 `best.pt` 与 `last.pt`

训练日志与 TensorBoard 默认输出到：

```
work_dir/<model>/fold_<k>/{train.log,tb/}
```

## 推理与 TTA

- `scripts/predict.py` 在验证集或测试集上输出概率图 `.npz`：
  - `prob` 为 `uint8 (0~255)` 的 3 通道概率图
  - 可选 TTA 翻转（8 种组合）
  - 可选输出 per-case Dice CSV（验证集）

## 融合与后处理

- `scripts/ensemble.py` 对多个模型概率图进行加权平均并后处理。
- 核心后处理逻辑 `src/postprocess.py`：
  - 三通道阈值化（支持 per-region thresholds）
  - 层级一致性：`ET ⊆ TC ⊆ WT`
  - ET 小体积假阳性抑制（`et_threshold_voxels` + `et_min_prob`）
  - 小连通域过滤（`min_sizes` + `wt_component`）
- 输出为 BraTS label（`0/1/2/4`），并用参考 affine 还原 NIfTI。

## 网格搜索与评估

- `scripts/grid_search.py`：融合权重 + 阈值搜索，输出 `best_config.json`。
- `scripts/final_eval.py`：基于 `best_config.json` 的最终评估（Dice + HD95）。
- `scripts/evaluate.py`：对融合后的 `nii.gz` 直接评估。
- `scripts/eval_npz_dice_hd95.py`：评估单模型概率图。

## 一键流水线（推荐）

`scripts/run_all.sh` 提供完整流水线：

1. 生成 folds
2. 逐 fold 训练 + 推理
3. 合并 OOF 概率图
4. 融合 + 网格搜索

同时配套：

- `scripts/launch.sh`：tmux 一键启动
- `scripts/monitor.sh`：系统监控
- `scripts/live_train.sh`：训练关键日志实时流

## nnU-Net v2 数据转换（可选）

`prepare_nnunet_brats.py` 可将 BraTS 数据转成 nnU-Net v2 格式（标签 4 → 3）。

## 目录结构速览

```
configs/            # 基础 + 模型配置
scripts/            # 训练/推理/融合/评估/流水线脚本
src/                # 核心代码：数据、模型、训练、后处理
data/               # folds.json + 原始 BraTS 数据
work_dir/           # 训练权重与日志
logs/               # 流水线日志
run_state/          # 流水线任务状态
result/             # 评估 CSV / 表格输出
paper_data/         # 论文绘图与统计
pretrained/         # 预训练权重
```

## 常用命令示例

```bash
# 1) 生成 5-fold 划分
python scripts/make_folds.py --data_root ./data/BraTS2021_TrainingData --out ./data/folds.json

# 2) 单模型单 fold 训练
python scripts/train.py --base_config configs/base.yaml --model_config configs/segresnet.yaml --fold 0 --gpu 0

# 3) 单模型推理（验证集）
python scripts/predict.py --base_config configs/base.yaml --model_config configs/segresnet.yaml \
  --ckpt work_dir/segresnet/fold_0/best.pt --fold 0 --out_dir /dev/shm/brats_probs/segresnet/fold_0 \
  --use_ema --tta --save_csv --model_name segresnet

# 4) 融合与生成 NIfTI
python scripts/ensemble.py --prob_dirs /dev/shm/brats_probs/segresnet_oof /dev/shm/brats_probs/swinunetr_oof \
  --ref_dir data/BraTS2021_TrainingData --out_dir result/ensemble_nii

# 5) 融合权重与阈值搜索
python scripts/grid_search.py --prob_dirs /dev/shm/brats_probs/segresnet_oof /dev/shm/brats_probs/swinunetr_oof \
  --ref_dir data/BraTS2021_TrainingData --folds_json data/folds.json --out_dir work_dir/grid_search

# 6) 最终评估
python scripts/final_eval.py --prob_dirs /dev/shm/brats_probs/segresnet_oof /dev/shm/brats_probs/swinunetr_oof \
  --config work_dir/grid_search/best_config.json --out_csv result/final_ensemble.csv
```

如需快速全流程，可直接使用：

```bash
bash scripts/run_all.sh
```