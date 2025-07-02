# Minimalist EBM Library in PyTorch
# 极简 EBM 库

This repository provides a minimalist, educational implementation of Energy-Based Models (EBMs) in PyTorch. Its goal is to serve as a clear and simple framework for understanding and comparing different EBM training algorithms.

## Features
## 特性

- **Minimalist Code (极简代码)**: Stripped-down implementations focusing on core logic.
- **Modular Design (模块化设计)**: Clear separation of data, models, samplers, and losses.
- **Bilingual Comments (中英双语注释)**: Code is commented in both English and Chinese for better understanding.
- **Algorithm Comparison (算法比较)**: Easily switch between Contrastive Divergence (CD), Denoising Score Matching (DSM), and Noise Contrastive Estimation (NCE).
- **Visualization (可视化)**: Generates plots of the learned energy landscape during training.

## Setup
## 环境设置

- Install the required dependencies:
    安装所需的依赖包：

    ```bash
    pip install -r requirements.txt
    ```

## How to Run
## 如何运行

The project can be run using the `train.py` script. You can specify the dataset, loss function, and other hyperparameters via command-line arguments.

项目可以通过 `train.py` 脚本运行。您可以通过命令行参数指定数据集、损失函数和其他超参数。

Example shell scripts are provided in the `examples/` directory.

`examples/` 目录下提供了一些示例 shell 脚本。

**To train with Contrastive Divergence (CD):**
**使用对比散度（CD）进行训练：**
```bash
sh examples/train_cd_2d.sh
```

**To train with Denoising Score Matching (DSM)**
**使用去噪分数匹配（DSM）进行训练：**
```bash
sh examples/train_dsm_2d.sh
```

**To train with Noise Contrastive Estimation (NCE):**
**使用噪声对比估计（NCE）进行训练：**
```bash
sh examples/train_nce_2d.sh
```

Outputs, including the energy landscape plots, will be saved in the directory specified by the `--output_dir` argument.

输出（包括能量曲面图）将被保存在 `--output_dir` 参数指定的目录中。

## File Structure
## 文件结构

```
mini-ebm/
├── README.md               # Project overview / 项目概述
├── requirements.txt        # Project dependencies / 项目依赖
├── toy_data.py             # Functions for generating 2D toy data / 生成 2D 玩具数据的函数
├── models.py               # Definition of the EnergyNetwork (MLP) / EnergyNetwork (MLP) 的定义
├── samplers.py             # LangevinSampler class / LangevinSampler 类
├── losses/                 # Loss functions package / 损失函数包
│   ├── __init__.py         # Package interface / 包接口
│   ├── cd_variants.py      # Contrastive Divergence variants / 对比散度变体
│   ├── sm_variants.py      # Score Matching variants / 分数匹配变体
│   └── nce_variants.py     # Noise Contrastive Estimation variants / 噪声对比估计变体
├── visualize.py            # Plotting functions for energy landscapes / 能量曲面绘图函数
├── train.py                # Main training script / 主训练脚本
└── examples/               # Example shell scripts / 示例 shell 脚本
    ├── train_cd_2d.sh
    ├── train_dsm_2d.sh
    └── train_nce_2d.sh
```
