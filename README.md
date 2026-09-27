# input_method

基于 PyTorch RNN 的「输入法下一个词预测」小项目：给一句话，预测最可能的下一个词。

## 功能

- 数据预处理：把对话语料用 jieba 切词、构建词表、生成 (上下文, 目标词) 训练样本
- 训练：RNN 语言模型，用交叉熵损失预测下一个词
- 推理：输入句子，输出 Top-5 候选词及概率

## 项目结构

```
src/                # 代码
  config.py         # 路径与超参数配置
  dataset.py        # 数据集与 DataLoader
  model.py          # RNN 模型定义
  train.py          # 训练脚本（训练完保存 models/model.pth）
  predict.py        # 预测脚本（交互式）
data/
  raw/              # 原始语料 synthesized_.jsonl
  processed/        # 处理后的 train/test 数据 + process.py 预处理脚本
models/
  vocab.txt         # 词表
  model.pth         # 训练好的模型权重
```

## 环境依赖

- Python（本项目使用 conda `pytorch` 环境）
- PyTorch（CPU / GPU 均可）
- jieba、pandas、scikit-learn

## 快速开始

在项目根目录下、已激活 `pytorch` 环境的终端里执行。

1. 数据预处理

```bash
python data/processed/process.py
```

2. 训练模型

```bash
python src/train.py
```

训练结束后模型权重保存到 `models/model.pth`。

3. 预测

```bash
python src/predict.py
```

运行后输入一句话回车，输出最可能的下一个词和 Top-5 候选；输入 `q` 退出。

## 说明

当前模型较小（RNN hidden 200、10 个 epoch、10% 采样数据），预测结果偏向高频词和标点符号，仅作演示。想提升效果可以从加大数据量、增加训练轮数、增大模型或改用 top-k 采样入手。
