# Drifting 最小化演示

## 运行方式

```bash
JAX_PLATFORMS=cpu python demo.py \
    --classes 95,22,88,108,386,296 \
    --cfg-scale 1.0 \
    --output-dir demo_output
```

**无需** ImageNet 数据集、FID 参考统计文件或 TPU，模型权重从 HuggingFace 自动下载。

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `hf://pixel_B_sota` | 模型标识符，支持 `hf://<name>` 或本地路径 |
| `--classes` | `95,22,88,108,386,296` | 逗号分隔的 ImageNet 类别 ID（0–999） |
| `--cfg-scale` | `1.0` | Classifier-Free Guidance 缩放系数 |
| `--seed` | `42` | 随机种子 |
| `--output-dir` | `demo_output` | 图像保存目录 |

## 运行结果

| 项目 | 内容 |
|------|------|
| 模型 | `pixel_B_sota`（ViT-B/16，hidden=768，depth=12） |
| 输出分辨率 | 256×256 像素 |
| CFG scale | 1.0（论文报告 FID 1.73） |
| NFE | **1**（单次前向传播，无迭代） |
| CPU 推理速度 | JIT 编译后约 **0.7 秒/张** |
| 输出文件 | `demo_output/class_<id>_<name>.png` + `demo_output/grid.png` |

生成示例（6 张图）：

| class 95 | class 22 | class 88 | class 108 |
|----------|----------|----------|-----------|
| jacamar（拟啄木鸟） | bald eagle（白头鹰） | macaw（金刚鹦鹉） | sea anemone（海葵） |

| class 386 | class 296 |
|-----------|-----------|
| african elephant（非洲象） | ice bear（北极熊） |

## 模型选择说明

本演示使用 **pixel_B_sota** 而非 latent 模型，原因如下：

- **latent 模型**（如 `ablation`、`latent_L_sota`）的后处理需要 VAE 解码，仓库中 `dataset/vae.py` 的 `_put_tree_on_local_tpu` 函数强制要求 TPU 设备，CPU 环境下直接报错。
- **pixel 模型**直接在像素空间输出图像，无需 VAE，可在纯 CPU 环境运行，无需修改任何已有代码。

若在 TPU 环境下，可切换为 latent 模型以获得更高质量（FID 1.53）：

```bash
JAX_PLATFORMS=tpu,cpu python demo.py --model hf://latent_L_sota --cfg-scale 1.0
```

## 算法简介

Drifting 是一种**单步生成模型**，与扩散模型或 Flow Matching 的核心区别在于：

- 训练时维护一个 **memory bank**，存储来自真实数据分布的样本。
- 在预训练 **MAE 特征空间**中计算类 Wasserstein 的 **drift loss**，将生成分布"漂移"向真实数据分布（见 `drift_loss.py`）。
- 推理时只需**一次前向传播**，无需 ODE/SDE 求解器或多步迭代。
- latent-L 模型在 ImageNet 256×256 上达到 **FID 1.53**（CFG=1.0，1 NFE）。

详见论文：[arXiv 2602.04770](http://arxiv.org/abs/2602.04770)
