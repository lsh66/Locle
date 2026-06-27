import torch
import os.path as osp
import ipdb 
from helper.args import get_command_line_args
from helper.utils import load_yaml, replace_tensor_values, noise_transition_matrix
from helper.utils import get_one_hop_neighbors, get_two_hop_neighbors_no_multiplication, get_sampled_nodes
from helper.train_utils import graph_consistency
from helper.data import get_dataset
from helper.active import density_query
import numpy as np
import matplotlib.pyplot as plt

# 用于评估伪标签（pseudo labels）的质量
# 1. 伪标签匹配率（第 18-42 行）
# 加载伪标签文件（_fixed_pl.pt）和 SBERT 特征文件（_fixed_sbert.pt）
# 对 Cora 数据集做标签映射对齐（mapping 字典）
# 计算两个指标：
# Full match ratio：Top-1 伪标签与真实标签完全一致的比例
# Partial match ratio：真实标签出现在伪标签候选列表（Top-K）中的比例
# 2. 基于密度的准确率分布（第 48-81 行）
# 调用 density_query 计算每个节点的密度分数
# 按密度从高到低排序，过滤掉无效预测（-1）
# 将样本等分为 20 个区间（按密度从高到低）
# 计算每个区间的 Top-1 准确率，绘制柱状图
# 目的：验证"高密度区域的伪标签是否更准确"这一假设，为主动学习/样本选择策略提供依据


def case_study(data_path, datasets = ['cora', 'pubmed']):
    ## 1. 真实标签出现在伪标签 Top-K 候选列表中的比例是多少？

    ## 加载伪标签和 SBERT 特征数据
    for dataset in datasets:
        # 加载伪标签文件（由 LLM 生成的 Top-K 预测标签）
        data = torch.load(osp.join(data_path, f"{dataset}_fixed_pl.pt"))
        # 加载 SBERT 特征文件（包含真实标签 y）
        s_data = torch.load(osp.join(data_path, f"{dataset}_fixed_sbert.pt"))
        # 伪标签矩阵，每行是一个节点的 Top-K 候选标签
        pseudo_labels = data.x
        # 标签值减 1，从 1-indexed 转为 0-indexed
        pseudo_labels -= 1
        pl_list = pseudo_labels.tolist()
        # Cora 数据集的 LLM 输出标签与真实标签存在映射关系，需要手动对齐
        if dataset == 'cora':
            mapping = {0: 2, 1:3, 2:1, 3:6, 4:5, 5:0, 6:4}
            pseudo_labels = replace_tensor_values(pseudo_labels, mapping)
        # import ipdb; ipdb.set_trace()
        # 真实标签（ground truth）
        gt = s_data.y 
        # 取 Top-1 预测标签（每行第一列）
        top1_pred = pseudo_labels[:, 0]
        # 计算 Top-1 完全匹配的样本数
        full_match = (top1_pred == gt).sum().item()

        # 将真实标签扩展为与伪标签相同的二维形状，便于逐行比较
        ground_truth = gt.view(-1, 1)

        # 检查真实标签是否出现在每个节点的 Top-K 候选列表中
        mask = pseudo_labels.eq(ground_truth)

        # 统计真实标签命中 Top-K 的样本数量（任一行中匹配即可）
        count = mask.any(dim=1).sum().item()

        # 输出 Top-1 精确匹配率
        print(f" {dataset} Full match ratio: ", full_match / len(gt))
        # 输出 Top-K 部分匹配率（真实标签在候选列表中出现的比例）
        print(f" {dataset} Partial match ratio: ", count / len(gt))

        # 噪声转移矩阵的保存路径（当前已注释，未启用）
        plot_name = f"{dataset}_noise_transition_matrix.png"

        # noise_transition_matrix(top1_pred, gt, plot_name)

        ## 2. 基于密度的准确率分布分析
        ## 目的：验证高密度区域的伪标签是否更准确，为主动学习样本选择提供依据
        # 使用伪标签文件中的特征作为嵌入（注意：此处可能应使用 s_data.x 即 SBERT 特征）
        x_embed = data.x
        # 聚类数量 = 标签最大值 + 1
        n_cluster = data.y.max().item() + 1
        # 将所有节点标记为训练节点（全量参与密度计算）
        train_mask = torch.ones(len(data.y), dtype = torch.bool)
        seed = 0
        device = 'cuda'
        # 调用密度查询函数，计算每个节点的局部密度分数
        _, density = density_query(x_embed.shape[0], x_embed, n_cluster, train_mask, seed, device)

        # 按密度从高到低排序，获取排序后的节点索引
        sorted_indices = torch.argsort(density, descending=True)

        # 过滤掉无效预测（标签为 -1 表示 LLM 未给出有效预测）
        non_empty_column = (top1_pred[sorted_indices] != -1)

        # 按密度排序并过滤后的预测标签
        sorted_pred = top1_pred[sorted_indices][non_empty_column]
        # 按密度排序并过滤后的真实标签（注意：此处使用 data.y 而非 s_data.y）
        sorted_y = data.y[sorted_indices][non_empty_column]
        # sorted_conf = conf[sorted_indices][non_empty_column]

        # 将样本等分为 20 个密度区间（从高密度到低密度）
        num_regions = 20
        samples_per_region = len(sorted_y) // num_regions
        regions = range(num_regions)
        accuracies = []

        # 逐区间计算 Top-1 准确率
        for i in regions:
            start_idx = i * samples_per_region
            end_idx = start_idx + samples_per_region
            accuracy = (sorted_pred[start_idx:end_idx] == sorted_y[start_idx:end_idx]).float().mean()
            accuracies.append(accuracy.item())
        
        # 绘制柱状图：横轴为密度区间索引（0=最高密度），纵轴为准确率
        plt.bar(regions, accuracies, alpha=0.8, color='blue')
        plt.xlabel("Region Index")
        plt.ylabel("Average Accuracy")
        plt.title("Accuracy by Region")
        plt.xticks(regions)
        plt.ylim([0, 1])
        # 注意：show() 在 savefig() 之前调用，可能导致保存的图片为空白
        plt.show()
        plt.savefig("{}_active.png".format(dataset))

        # if dataset == 'cora':
        #     _, density = density_query(b, x_embed, n_cluster, train_mask, seed, device):


            






    ## load dataset

    
    ##2. What's the accuracy of only 1 predictions
            # Find where tensor is not -1 (i.e., non-empty), this gives a tensor of the same shape with True where it's not -1
        # non_empty = pseudo_labels.ne(-1)
        # # Count the number of non-empty entries along each row
        # count = non_empty.sum(dim=1)
        # # Find where the count is exactly 1
        # rows = (count == 1).nonzero(as_tuple=True)[0]
        # count = (gt[rows] == top1_pred[rows]).sum().item()
        # total_count = len(rows)
        # print(f"{dataset} only 1 label acc: ", count / total_count)
        # print(f"{dataset} has {total_count} samples with only 1 label")
    
        # ## Consistency based on 1 hop neighbors
        # test_node_idxs, train_node_idxs = get_sampled_nodes(s_data)
        # one_hop_neighbor_dict = get_one_hop_neighbors(s_data, test_node_idxs)
        # two_hop_neighbor_dict = get_two_hop_neighbors_no_multiplication(s_data, test_node_idxs)

        # sorted_nodes = graph_consistency(one_hop_neighbor_dict, pseudo_labels, gt, top1_pred)
        # sorted_nodes = sorted_nodes[::-1]
        # for i in range(5):
        #     this_sorted_nodes = sorted_nodes[:len(sorted_nodes) // 5 * (i + 1)]
        #     this_part_pred = torch.tensor([top1_pred[key] for key in this_sorted_nodes])
        #     this_part_gt = torch.tensor([gt[key] for key in this_sorted_nodes])
        #     acc = (this_part_pred == this_part_gt).sum().item() / len(this_part_gt)
        #     print("This sorted acc:", acc)

        
        



        

if __name__ == '__main__':
    command_line_args = get_command_line_args()
    params_dict = load_yaml(command_line_args.yaml_path)
    data_path = params_dict['DATA_PATH']
    case_study(data_path)
        


