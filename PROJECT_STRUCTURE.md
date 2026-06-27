# Locle 项目说明文档

## 项目概述

**Locle** 是一个 SIGIR 2025 论文的开源实现，主题为"Leveraging Large Language Models for Effective Label-free Node Classification in Text-Attributed Graphs"（利用大语言模型对文本属性图进行有效的无标签节点分类）。

项目地址：[Hugging Face🤗](https://huggingface.co/papers/2412.11983)

---

## 目录结构

```
Locle/
├── config.yaml              # 配置文件（路径、API密钥等）
├── requirements.txt         # Python依赖包列表
├── README.md                # 项目说明文档
├── .gitignore               # Git忽略文件配置
├── data/                    # 📁 数据集存放目录（需要手动创建）
│   ├── annotations/         # 注释缓存数据
│   └── partitions/         # 数据划分文件
├── src/                     # 核心源代码目录
│   ├── main.py              # 主入口文件
│   ├── train.py             # 训练启动脚本
│   ├── precompute.py        # 预计算模块
│   ├── llm.py               # LLM（大语言模型）相关功能
│   ├── inspection.py        # 检查/分析工具
│   ├── case_s.py           # 案例研究（简单版）
│   ├── case_study.py        # 案例研究
│   ├── generate_data_file.py # 生成数据文件
│   ├── crawler/             # 爬虫相关代码
│   │   └── README.md
│   ├── helper/              # 辅助工具函数
│   │   ├── args.py          # 命令行参数处理
│   │   ├── data.py          # 数据处理相关
│   │   ├── utils.py         # 通用工具函数
│   │   ├── train_utils.py   # 训练工具函数
│   │   ├── active.py        # 主动学习相关
│   │   ├── noisy.py         # 噪声处理相关
│   │   ├── hoc.py           # 高阶相关性（High Order Correlation）
│   │   ├── partition.py     # 数据划分相关
│   │   ├── my_utils.py      # 杂项工具
│   │   ├── hyper_search.py  # 超参数搜索
│   │   ├── Rewire_GNN.py    # GNN重连算法
│   │   └── __init__.py
│   ├── models/              # 模型定义
│   │   ├── nn.py            # 神经网络模型
│   │   ├── gcn_lib/         # GCN（图卷积网络）库
│   │   │   ├── dense/       # 密集版本
│   │   │   └── sparse/      # 稀疏版本
│   │   └── rev/             # 可逆层/RevNet相关
│   │       ├── gcn_revop.py
│   │       ├── memgcn.py
│   │       └── rev_layer.py
│   └── openail/             # OpenAI API封装
│       ├── asyncapi.py
│       ├── config.py
│       └── utils.py
└── train_bash/              # 训练脚本集合
    └── best_cache.sh        # 最佳缓存配置的训练脚本
```

---

## 数据集存放位置

**重要：数据集应放在 `data/` 目录下**

根据 README.md 的说明：
1. 从 [Google Drive](https://drive.google.com/file/d/1c5yqvUu3PBUhQHvUUlhzs_37IRWcHA2E/view?usp=sharing) 下载数据集
2. 将下载的数据放入 `data/` 文件夹
3. 注释缓存已准备好放在 `data/annotations/` 下

**当前状态：** `data/` 目录尚不存在，需要手动创建并放入数据。

```bash
# 创建目录结构（如果不存在）
mkdir -p data/annotations
mkdir -p data/partitions
```

---

## 核心文件说明

| 文件/目录 | 作用 |
|---------|------|
| `config.yaml` | 全局配置：数据路径(`DATA_PATH: data`)、API密钥、日志路径等 |
| `src/main.py` | 项目主入口，负责协调各个模块 |
| `src/train.py` | 训练启动脚本 |
| `src/precompute.py` | 预计算特征嵌入等工作 |
| `src/llm.py` | 与大语言模型交互的核心模块 |
| `src/helper/data.py` | 数据加载和处理 |
| `src/helper/active.py` | 主动学习策略实现 |
| `src/helper/hoc.py` | 高阶相关性计算 |
| `src/models/nn.py` | 核心神经网络模型 |
| `train_bash/best_cache.sh` | 使用最佳配置运行训练的脚本 |

---

## 环境配置

主要依赖（见 `requirements.txt`）：
- `python==3.10.0`
- `torch==2.3.0`
- `torch_geometric==2.5.3`
- `openai==1.59.4`
- `dgl==1.1.1`

安装命令：
```bash
pip install -r requirements.txt
```

---

## 快速开始

1. **安装环境**
   ```bash
   pip install -r requirements.txt
   ```

2. **准备数据集**
   - 下载数据集并放入 `data/` 目录

3. **配置**
   - 修改 `config.yaml` 中的 `OPENAI_KEY` 为你的API密钥

4. **运行训练**
   ```bash
   bash train_bash/best_cache.sh
   ```

---

