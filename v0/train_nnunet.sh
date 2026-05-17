#!/bin/bash

# 1. 设置 nnU-Net v2 必需的环境变量
export nnUNet_raw="/home/cdy/workspace/brain_project/nnUNet_raw"
export nnUNet_preprocessed="/home/cdy/workspace/brain_project/nnUNet_preprocessed"
export nnUNet_results="/home/cdy/workspace/brain_project/nnUNet_results"

# 创建目录（如果尚未存在）
mkdir -p $nnUNet_raw
mkdir -p $nnUNet_preprocessed
mkdir -p $nnUNet_results

# 2. 执行数据转换 (将 BraTS 数据整理为 nnU-Net 所需格式，连带 label 重映射 4->3)
echo "========================================"
echo "开始转换 BraTS 数据集格式到 nnUNet_raw..."
python 8_create_nnunet_dataset.py
echo "========================================"

# 3. 数据集指纹提取与预处理
# -d 137 是我们在 py 脚本中指定的 Dataset ID
echo "========================================"
echo "开始数据预处理与计划生成 (Plan & Preprocess)..."
nnUNetv2_plan_and_preprocess -d 137 --verify_dataset_integrity
echo "========================================"

# 4. 模型训练
# 我们先默认训练 3D 全分辨率 (3d_fullres) 网络的 fold 0
echo "========================================"
echo "开始使用 nnU-Net v2 训练 3D U-Net 强基线模型 (Fold 0)..."
# 注意：标准的推断和训练通常是在有 GPU 的机器上进行。若需要后台运行可以加上 nohup 或 screen
nnUNetv2_train 137 3d_fullres 0
echo "========================================"

# 提示后续可用的命令 (推理)
echo ""
echo "训练完成后，您可以使用如下命令对验证集/测试集进行预测以供最终集成："
echo "nnUNetv2_predict -i /path/to/test_imagesTr -o ./result/nnunet_predictions -d 137 -c 3d_fullres -f 0 --save_probabilities"
