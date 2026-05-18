# 论文数据快照

**生成时间**: 2026-05-18 17:46:19
**源目录**:   /root/autodl-tmp/workspace/brain-project
**总大小**:   1.4G

## 目录结构

| 子目录 | 内容 | 大小 |
|---|---|---|
| `models/`    | 所有 fold 的 best.pt        | 1.3G |
| `curves/`    | TensorBoard 日志             | 20K |
| `logs/`      | 训练文本日志                 | 6.0M |
| `csv/`       | per-case Dice 合并表         | 0 |
| `probs/`     | 代表性 case 概率图           | 0 |
| `ensemble/`  | 融合 + 网格搜索结果          | 0 |
| `figures/`   | 论文图（生成后存这里）       | — |

## 论文图生成命令

```bash
# F4 训练曲线
python scripts/plot_training_curves.py \
    --logs_dir ./paper_data_test/logs \
    --models segresnet,swinunetr \
    --output ./paper_data_test/figures/F4_training_curves.png

# F5 定性分割结果（5 个代表性 case）
python scripts/viz_segmentation.py \
    --csv ./paper_data_test/csv/all_per_case_dice.csv \
    --auto 5 \
    --probs_dir ./paper_data_test/probs \
    --output ./paper_data_test/figures/F5_qualitative.png

# F6 Dice 箱线图（待写脚本）
# F10 失败案例（待写脚本）
```

## 关键发现

（推理未完成，无 CSV 摘要）
