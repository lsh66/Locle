# Locle

[Hugging Face🤗](https://huggingface.co/papers/2412.11983)

This repository contains the code of the paper ["Leveraging Large Language Models for Effective Label-free Node
Classification in Text-Attributed Graphs"](https://arxiv.org/abs/2412.11983) accepted to SIGIR 2025 Full Paper. To reproduce the results in the paper, please follow the steps:

1. Prepare the environment with
```
pip install -r requirements.txt
```
2. The above environment may contain some unrelated packages. You can also just install it yourself with some key requirements:

```
python==3.10.0
torch==2.3.0
torch_cluster==1.6.3
torch_geometric==2.5.3
torch_scatter==2.1.2
torch_sparse==0.6.18
openai==1.59.4
dgl==1.1.1
```

3. Prepare the required datasets. Please download the dataset from [google_drive_link](https://drive.google.com/file/d/1c5yqvUu3PBUhQHvUUlhzs_37IRWcHA2E/view?usp=sharing), and put them under the data/ folder. We have also prepared our annotation cache under the data/annotations/ folder. In most cases, this should be able to satisfy the requirement. If there are still omissions, please change the api url and key and generate the annotations by yourself.

4. Replace the configurations in config.yaml with your own config, and reproduce all the results with 
```
bash train_bash/best_cache.sh
```



为什么这么多 .pt 文件：
项目在每个计算步骤后都调用 torch.save() 缓存中间结果，避免重复计算。具体分布：

目录	          存什么	                 例子
data/*.pt	数据集主文件	cora_fixed_sbert.pt
data/aax/	主动学习预计算	aax_2708.pt、density_aax_2708_70.pt
data/active/	LLM 伪标签结果	cora^cache^few_shot_all.pt
data/cache/	API 请求 / 响应缓存	.json 文件
data/annotations/	LLM 标注结果	.json 文件
data/partitions/	图分区结果	给 gpart_query 用
data/subspace_selection/	子空间选择缓存	729 个文件

本质是这个项目的实验流程很长（数据预处理 → LLM标注 → 主动学习 → 训练），
每个环节的中间结果都落盘成 .pt，方便断点续跑和复用。