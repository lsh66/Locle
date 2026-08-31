# ANS-OCD 使用说明文档

## 1. 项目简介

**ANS-OCD（Adaptive Network Sampling for Overlapping Community Detection）** 是一个用于发现社交网络中**重叠社区**的算法框架。该框架结合了以下技术：

- **子空间聚类（Subspace Clustering）**：基于节点属性/嵌入估计社区数量 `K_init`。
- **DivRank 代表节点选择**：在每个潜在社区中选择最具代表性的节点。

DivRank 代表节点选择的流程是：

分簇：根据子空间聚类结果将图分成若干簇；
子图提取：对每个簇构建诱导子图；
DivRank 评分：在子图上运行带自环的 PageRank 变体，得到每个节点的影响力分数；
Top-k 选择：每个簇取分数最高的 reps_per_cluster 个节点；
Prompt 构建：将代表节点的文本组织成 LLM 可理解的社区描述，用于后续主题校验。


| 方式               | 优点                                   | 缺点                                   |
| ------------------ | -------------------------------------- | -------------------------------------- |
| 全局 Top DivRank   | 选出的节点全局影响力大                 | 大社区垄断、小社区被忽视、主题覆盖不均 |
| 每簇内 Top DivRank | 每个社区都有代表、主题更局部、避免偏见 | 需要先做聚类                           |






- **图神经网络（GNN）**：学习节点的社区隶属概率。
- **标签熵（Label Entropy）**：识别高不确定性节点。
- **社区扩展（Community Expansion）**：基于 GNN 概率启发式扩展社区边界。
- **冲突触发 LLM 校验（Conflict-Triggered Labeling）**：在规则与模型预测冲突时引入大语言模型进行裁决。
- **多轮自训练（Self-Training）**：迭代优化 GNN 的伪标签，提高重叠社区检测质量。

---

## 2. 目录结构

```text
Locle/
├── ANS_OCD/                          # 主实现目录
│   ├── main.py                       # 六阶段主流程入口
│   ├── config.py                     # 全局超参数与 API Key 配置
│   ├── utils/
│   │   ├── data_loader.py            # 图/属性/文本/标签加载 + SBERT 编码
│   │   └── metrics.py                # Omega Index / Jaccard / F1 评估指标
│   ├── module1_representative/
│   │   ├── subspace_cluster.py       # 子空间聚类 + Eigengap K 估计
│   │   └── divrank_selector.py       # 簇内 DivRank 代表节点选择
│   ├── module2_gnn_entropy/
│   │   ├── gnn_model.py              # 多标签 GCN 模型
│   │   └── label_entropy.py          # 多标签二元标签熵计算
│   ├── module3_expansion/
│   │   └── community_expander.py     # GNN 概率引导的社区扩展
│   └── module4_ctl_llm/
│       ├── prompt_engine.py          # 主题校验 / 冲突裁决 Prompt 生成
│       ├── ctl_mechanism.py          # LLM 调用封装（支持智谱 / OpenAI / Mock）
│       └── conflict_detector.py      # 规则 vs GNN 冲突检测
├── ANS-OCD/                          # 说明文档所在目录
│   └── README.md                     # 本说明文档
├── config.yaml                       # 项目根配置（含 API Key）
├── requirements.txt                  # Python 依赖
└── data/                             # 数据目录（需自行准备）
```

---

## 3. 环境安装

当前代码依赖 `torch_geometric`、`sentence-transformers` 等库，且对 Python 版本敏感。推荐在 **Python 3.10** 环境下安装：

```bash
conda create -n ansocd python=3.10
conda activate ansocd
cd c:/Users/23739/Desktop/study/work2/Locle
pip install -r requirements.txt
```

**注意**：
- Python 3.12 + `numpy>=2.0` 与 `torch==2.2.0/2.3.0` 存在兼容性问题，不建议使用。
- 若使用智谱 API，需确保 `requests` 已安装（`requirements.txt` 中通常已包含）。

---

## 4. 运行方式

### 4.1 合成数据冒烟测试（无需准备数据）

```bash
cd c:/Users/23739/Desktop/study/work2/Locle
python -m ANS_OCD.main
```

默认会生成 40 个节点、若干社区的合成图进行全流程测试。

### 4.2 使用真实数据

在项目根目录创建 `data/` 文件夹，并准备以下 CSV 文件：

| 文件 | 是否必需 | 格式说明 |
|---|---|---|
| `data/edges.csv` | 是 | 表头：`source,target`；每行一条无向边 |
| `data/attributes.csv` | 否 | 无表头，每行一个节点的数值属性向量 |
| `data/labels.csv` | 否 | 表头：`label`；单标签写 `1`，多标签写 `0;1` |
| `data/texts.csv` | 否 | 表头：`text`；文本会调用 Sentence-BERT 编码 |

示例：

```csv
# data/edges.csv
source,target
0,1
1,2
2,0
```

```csv
# data/labels.csv
label
0;1
1
0
```

```csv
# data/texts.csv
text
graph neural networks for community detection
overlapping community discovery
```

---

## 5. 输出结果解释

以下是一段典型的合成数据运行日志及其含义：

```text
[ANS-OCD] Running on synthetic data for smoke testing.
[Stage 2] Subspace clustering and K estimation ...
  K_init estimated by Eigengap: 1
  LLM validated K: 1
  Themes: {1: 'Cluster 0'}
  Merge suggestions: []
```

### 阶段 2：子空间聚类与 K 估计

- **`K_init estimated by Eigengap: 1`**：
  通过归一化拉普拉斯矩阵的特征值间隙（Eigengap）自动估计出的初始社区数量。本例中估计为 1 个社区。

- **`LLM validated K: 1`**：
  大语言模型（LLM）对初始聚类结果进行主题校验后，确认最终社区数为 1。

- **`Themes: {1: 'Cluster 0'}`**：
  LLM 为每个社区提炼出的主题描述。本例中社区 1 的主题被概括为 "Cluster 0"。

- **`Merge suggestions: []`**：
  LLM 认为没有需要合并的社区。

---

```text
[Stage 6] Self-training round 1/3 ...
  High-entropy nodes selected: 4
  Expanded nodes per community: {0: 3}
  Conflicts detected: 0

[Stage 6] Self-training round 2/3 ...
  High-entropy nodes selected: 4
  Expanded nodes per community: {0: 3}
  Conflicts detected: 0
  Mean probability change: 0.109466

[Stage 6] Self-training round 3/3 ...
  High-entropy nodes selected: 4
  Expanded nodes per community: {0: 3}
  Conflicts detected: 0
  Mean probability change: 0.022382
```

### 阶段 6：多轮自训练

- **`Self-training round X/3`**：
  自训练共进行 3 轮，每轮都会重新选择高熵节点、扩展社区、检测冲突并更新 GNN 伪标签。

- **`High-entropy nodes selected: 4`**：
  每轮选出 4 个标签熵最高的节点。这些节点是模型最不确定的节点，需要通过社区扩展或 LLM 校验来优化。

- **`Expanded nodes per community: {0: 3}`**：
  社区 0 本轮扩展了 3 个新节点。

- **`Conflicts detected: 0`**：
  规则扩展结果与 GNN 预测之间没有发生冲突，因此没有触发 LLM 裁决。

- **`Mean probability change: 0.109466`**：
  本轮 GNN 预测概率相比上轮的平均变化量。数值逐渐减小（`0.109` → `0.022`）说明模型正在收敛。

---

```text
[ANS-OCD] Final overlapping communities:
  Community 0: 9 nodes

Smoke test completed. Final membership shape: (40, 1)
```

### 最终结果

- **`Community 0: 9 nodes`**：
  最终检测出的社区 0 包含 9 个节点。

- **`Final membership shape: (40, 1)`**：
  最终的社区隶属矩阵形状为 40 个节点 × 1 个社区。由于本次合成数据只估计出 1 个社区，因此是 `(40, 1)`。每个元素的值在 0~1 之间，表示该节点属于对应社区的**软概率**。

---

## 6. 配置参数

主要参数位于 `ANS_OCD/config.py`：

```python
LLM_CONFIG = {
    "use_mock": False,        # True=离线 Mock，False=调用真实 LLM
    "provider": "zhipu",      # "zhipu" 或 "openai"
    "model": "glm-4-flash",   # 智谱模型；OpenAI 可改为 "gpt-3.5-turbo"
    "temperature": 0.0,
    "max_tokens": 512,
}
```

- `use_mock=True`：不调用任何真实 LLM，返回预设 Mock 结果，适合快速测试。
- `provider="zhipu"`：使用智谱 AI API。
- `provider="openai"`：使用 OpenAI API（需安装 `openai==0.28`）。

API Key 读取优先级：
1. 环境变量：`ZHIPU_API_KEY` / `OPENAI_API_KEY`
2. 项目根目录 `config.yaml` 中的 `ZHIPU_KEY` / `OPENAI_KEY`

---

## 7. 评估指标

如果提供了真实标签 `data/labels.csv`，程序会输出以下指标：

| 指标 | 含义 |
|---|---|
| `omega_index` | Omega Index，衡量预测社区与真实社区集合的相似度 |
| `avg_jaccard` | 平均 Jaccard 相似度 |
| `f1_micro` | 微观 F1 分数，用于多标签社区隶属评估 |

---

## 8. 常见问题

### Q1：为什么 LLM 调用失败并回退到 Mock 模式？

A：请检查：
- `LLM_CONFIG["use_mock"]` 是否为 `False`。
- `provider` 是否设置正确（`zhipu` 或 `openai`）。
- API Key 是否已配置在 `config.yaml` 或环境变量中。
- 若使用 OpenAI，请确保安装 `openai==0.28`；若使用智谱，确保 `requests` 可用。

### Q2：K_init 估计为 1 是否正常？

A：合成数据规模较小或社区结构不明显时，Eigengap 可能估计为 1。真实数据集上通常会有更合理的估计。可通过调整 `SUBSPACE_CONFIG` 中的 `alpha` 和 `affinity_threshold` 来影响估计结果。

### Q3：如何处理 NumPy / Torch 版本冲突？

A：在 Python 3.10 环境中安装 `requirements.txt` 指定版本即可避免。

---

## 9. 安全提醒

`config.yaml` 中保存了真实的 API Key，**请勿将其提交到 Git**。建议将 `config.yaml` 加入 `.gitignore`：

```gitignore
config.yaml
``

---

## 10. 引用

本项目基于自适应网络采样与图神经网络的思想，结合大语言模型进行社区发现与校验。具体实现参考 `ANS_OCD/` 目录下的各模块源码及注释。
