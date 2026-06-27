"""Load Raw Communities Data"""
import pickle
import random

import pandas as pd

import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from mkl_random.mklrand import permutation


# 加载数据集
def load_data(dataset,mini_dataset=None):
    mainpath = '../run_zu/data'
    if dataset=='webkbs':

        filepath = mainpath + "/" + dataset + "/"
        structfile = f'adjacency_matrix_{mini_dataset}.csv'
        contfile = f'content_file_{mini_dataset}.csv'
        fileLabels = f'label_file_{mini_dataset}.csv'

    else:

        filepath = mainpath +"/"+ dataset+"/"
        structfile = 'adjacency_matrix.csv'
        contfile = 'content_file.csv'
        fileLabels = 'label_file.csv'

    A = pd.read_csv(filepath + structfile, header=None)
    A = A.values

    C = pd.read_csv(filepath + contfile, header=None)
    C = C.values

    # 2. 提取第一列数据并转换为列表
    true_labels = pd.read_csv(filepath + fileLabels, header=None)
    true_labels = true_labels[0].tolist()

    # NoComm代表数据集中不同类别的数量，也就是真实标签集合里不同元素的数量。
    NoComm = len(set(true_labels))

    print("A的形状",A.shape)
    print("C的形状",C.shape)
    # assert (A.shape[0] == C.shape[0] & A.shape[0] == len(true_labels))

    # is_symmetric = True
    # for i in range(A.shape[0]):
    #     for j in range(A.shape[0]):
    #         if A[i][j] != A[j][i]:
    #             print("不对称")
    #
    #
    # print("对称")
    undirected_A=directed_to_undirected_binary(A)

    return undirected_A, C, true_labels, NoComm


def directed_to_undirected_binary(adj_matrix):
    """
    将有向图的二进制邻接矩阵转换为无向图的邻接矩阵
    :param adj_matrix: 有向图的邻接矩阵（numpy 数组）
    :return: 无向图的邻接矩阵（numpy 数组）
    """
    # 获取邻接矩阵的行数和列数
    rows, columns = adj_matrix.shape
    # 确保邻接矩阵是方阵
    if rows != columns:
        raise ValueError("邻接矩阵必须是方阵")
    # 创建一个与原矩阵相同大小的零矩阵
    undirected_matrix = np.zeros((rows, columns))
    # 遍历矩阵的每个元素
    for i in range(rows):
        for j in range(columns):
            # 取 A[i][j] 和 A[j][i] 的逻辑或
            undirected_matrix[i][j] = adj_matrix[i][j] or adj_matrix[j][i]
    return undirected_matrix



def draw_graph(G, title="Graph"):
    # 使用 spring 布局，更自然分布
    pos = nx.spring_layout(G, seed=42)  # 固定随机种子以保持图形稳定

    # 节点度数影响节点大小
    degrees = dict(G.degree())


    # 自定义颜色：可以替换为按社区或类型着色
    node_colors = 'skyblue'

    plt.figure(figsize=(10, 7))
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors,
                           node_size=5,
                           edgecolors='black', linewidths=0.5)



    plt.title(title, fontsize=14)
    plt.axis('off')  # 关闭坐标轴
    plt.tight_layout()
    plt.show()
    print("图绘制完成")



