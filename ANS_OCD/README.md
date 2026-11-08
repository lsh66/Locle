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

### 方法定位：与 ANS-CD 的边界（重要修正）

需要注意：ANS-CD 与 ANS-OCD **共享上半段流程**，以下两项**不属于**本项目的创新：

| 环节 | ANS-CD | ANS-OCD | 判定 |
| --- | --- | --- | --- |
| 代表节点 | DivRank → `top_indices` | DivRank → 代表节点 | **共有** |
| 不确定节点 | NEW-ONE → 低确定性节点 | 标签熵 → 高熵节点 | **共有** |

**真正的分水岭在于「找到不确定节点之后做什么」**：

```python
262:271:src/run_zu/advanced_util.py
    for node in top_indices:
        label=query_label(node,dataset,mini_dataset)     # ← ANS-CD：查真实标签
        ...
        for neighbor in adj_list[node]:
            ...
            queried_nodes_to_community[neighbor] = label  # 邻居也直接赋标签
```

- **ANS-CD**：把不确定节点当作「**该问谁**」的索引 → 逐个 `query_label` **消耗标注预算**。
- **ANS-OCD**：把它们送进**扩展豁免队列**与 **LLM 裁决**，当作「**该推理谁**」的对象 → **不消耗标注**。

#### ANS-OCD 的独有优势（排除共有组件后）

| # | 优势 | 说明 |
| --- | --- | --- |
| 1 | **不确定节点用于推理而非问标签** | 范式差异，标签效率的核心来源 |
| 2 | **重叠社区输出** | ANS-CD 用 `redistribute_duplicates_to_max_AS_per` 强制单归属（从其他社区 `remove`）；ANS-OCD 用 Sigmoid 多标签 + 独立社区集合 + 无去重 + 软分配 |
| 3 | **LLM 冲突裁决（CTL）** | ANS-CD 无语义仲裁；实测条件采纳后 `avg_jaccard` 0.2611 → 0.2775、`ari_flat` +77% |
| 4 | **在线 GNN + 自训练闭环** | vs ANS-CD 离线 ONE（硬编码 `result_dir` 加载预计算嵌入），一次性、无反馈、换机器即失效 |
| 5 | **全图覆盖** | ANS-CD 受 `raw_budget` 限制（citeseer 覆盖 1305、**遗漏 2007**，靠「就近塞」兜底）；ANS-OCD 用 GNN argmax 覆盖全部 3186 节点 |
| 6 | **重叠感知评估** | Omega Index / Jaccard / overlap_rate；ANS-CD 的单标签指标无法评价重叠质量 |

#### 关于「自适应 K」的现状与改进方向

当前实现中 `num_communities` 取自真实标签类别数 `true_k`（见 `main.py`），LLM 校验得到的 `k_final`
仅在无标签场景下生效，因此**「自适应 K」尚未真正兑现**。

原因是 Eigengap 在自表达亲和矩阵上估出 K=40（真实 6），偏差 6.7 倍——它度量的是「**拓扑子团数**」，
而真实标签是「**语义主题数**」，一个语义主题内部常含多个紧密子团。

改用**轮廓系数（silhouette）在属性空间扫描选 K** 可在无标签下得到接近真实的粒度：

| K | silhouette | NMI（仅作参照） |
| --- | --- | --- |
| **5** | **0.0646** ★ | **0.4739** ★ |
| 6（真实值） | 0.0623 | 0.4413 |
| 40（Eigengap） | — | 极差 |

silhouette 峰值与 NMI 峰值**重合**，说明内部准则能捕捉语义粒度，自适应 K **可救活**；
且 K=5 的 NMI 反而高于 K=6。

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

API Key **只从环境变量读取**，`config.yaml` 中不存放任何密钥（该文件已纳入版本控制）：

```powershell
# Windows PowerShell（当前会话）
$env:ZHIPU_API_KEY  = "<your-zhipu-key>"   # ANS-OCD，provider="zhipu"
$env:OPENAI_API_KEY = "<your-openai-key>"  # ANS-CD 或 provider="openai"

# 永久生效（新开终端后仍有效）
setx ZHIPU_API_KEY  "<your-zhipu-key>"
setx OPENAI_API_KEY "<your-openai-key>"
```

```bash
# Linux / macOS
export ZHIPU_API_KEY="<your-zhipu-key>"
export OPENAI_API_KEY="<your-openai-key>"
```

未设置时 `ctl_mechanism.py` 会打印提示并回退到 Mock 模式，不会报错中断。

---

## 7. 评估指标

如果提供了真实标签 `data/labels.csv`，程序会输出以下指标：

**集合重叠度**

| 指标 | 含义 |
| --- | --- |
| `omega_index` | Omega Index（Fuzzy Rand Index），统计节点对「是否同属至少一个社区」的判别一致率 |
| `avg_jaccard` | 逐节点预测社区集与真实社区集的交并比，再对全部节点取平均 |

**多标签 F1**

| 指标 | 含义 |
| --- | --- |
| `f1_micro` | 微观 F1：隶属矩阵展平为所有 (节点, 社区) 对的二分类后计算 |
| `f1_macro` | 宏观 F1：各类别（社区）等权平均 |
| `avg_precision` / `avg_recall` / `avg_f1` | 逐节点 P/R/F1 后取节点平均（对「空预测」敏感，可直接暴露覆盖不足） |

**展平聚类指标**

| 指标 | 含义 |
| --- | --- |
| `nmi_flat` | 隶属矩阵展平后的归一化互信息 |
| `ari_flat` | 展平后的调整兰德指数（排除随机吻合） |

**重叠统计**

| 指标 | 含义 |
| --- | --- |
| `avg_pred_communities` | 预测的每个节点平均隶属社区数 |
| `avg_true_communities` | 真实每个节点平均隶属社区数 |
| `overlap_rate` | 隶属 >1 个社区的节点占比（**重叠程度的核心判据**） |

> **读指标的注意事项**：`omega_index` 与 `f1_micro` 在「大量节点被判为无社区」时会虚高
> （因为真实标签多为单社区，「都判空」在配对层面反而一致）。
> 判断效果应优先看 `avg_jaccard`、`f1_macro`、`avg_pred_communities` 与 `overlap_rate`。

> **输入对齐**：真实标签需为二维多标签二进制矩阵 `[N, K]`；若为一维单标签会自动转 one-hot。
> 预测 K 与真实 K 不一致时按列截断/补零对齐（见 `main.py: compute_overlapping_metrics`）。

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

### Q2.1：最终使用的 K 到底来自哪里？（重要）

A：**不是** `K_init`，也通常**不是** LLM 校验的 `k_final`，而是 `target_k`，其取值规则为（见 `main.py`）：

```python
if true_k is not None:      # true_k = 真实标签的类别数
    target_k = true_k       # ← 有标签时，直接采用真实类别数
else:
    target_k = k_final      # ← 仅在无标签场景才用 LLM 校验结果
target_k = min(target_k, k_init)
```

即：
- **有真实标签时**：K = 真实类别数（Citeseer 为 6）。LLM 返回的 `k_final` **不会覆盖**它，只打印不生效。
  原因是 Mock 模式下 LLM 会返回 36 个主题（`k_init` 量级），若让它覆盖会把 K 撑爆并严重劣化指标。
- **无真实标签时**：K = `k_final`（LLM 校验结果）。
- 可用 `--k <int>` 手动强制覆盖（消融用），例如 `--k 6`。

> 因此严格来说，当前「自适应 K」**尚未真正兑现**，详见第 1 节「方法定位」中的「关于自适应 K 的现状与改进方向」。

### Q2.2：Eigengap 估出的 K_init 为什么远大于真实类别数？

A：Eigengap 是在**自表达亲和矩阵**上估计的，度量的是「**拓扑子团数**」（Citeseer 上为 40）；
而真实标签是「**语义主题数**」（6）。一个语义主题内部常包含多个紧密子团，二者本就不同——
这不是「估不准」，而是准则回答了另一个问题。改用轮廓系数等语义导向准则可显著改善，见第 1 节。

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
