import argparse
import os
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import torch
from networkx.algorithms.bipartite.basic import density
from orderedset import OrderedSet
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score, completeness_score, \
    v_measure_score, f1_score
from functools import lru_cache

from sklearn.metrics.pairwise import cosine_similarity

from run_zu.divrank import divrank, find_minimal_k, load_data_from_2file

from run_zu.utils.get_centrality import centralissimo
from run_zu.utils.load_dataset import load_data
from run_zu.utils.metric import evaluate_metrics

# 缓存内容相似度
_sim_cache = {}
def parse_args():
    args = argparse.ArgumentParser(description='arguments.')
    args.add_argument('--dataset', type=str, default='cora')


    args = args.parse_args()
    return args

def cosine_similarity_vector(a, b, idx_a=None, idx_b=None):
    """
    a, b: np.ndarray (或 list)
    idx_a, idx_b: 节点索引，用于稳定缓存
    """
    if idx_a is not None and idx_b is not None:
        key = (min(idx_a, idx_b), max(idx_a, idx_b))
        if key in _sim_cache:
            return _sim_cache[key]

    a = np.array(a)
    b = np.array(b)
    numerator = np.dot(a, b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    result = 0.0 if denominator == 0 else numerator / denominator

    if idx_a is not None and idx_b is not None:
        _sim_cache[key] = result
    return result



import scipy.sparse as sp

def cal_AC(c, content_np):
    """
    优化版：先归一化后按稀疏矩阵乘法，减少显存消耗和加速
    """
    if len(c) == 1:
        return 1.0

    X_c_sp = sp.csr_matrix(content_np[list(c)])
    X_c_sp = X_c_sp.multiply(1.0 / (np.sqrt(X_c_sp.power(2).sum(axis=1)).A1[:, None] + 1e-8))
    S = X_c_sp @ X_c_sp.T
    total_sum = S.sum() - len(c)
    ac = 2 * total_sum / (len(c) * (len(c) - 1))
    return ac



def cal_del_AC_add_node(AC_before,c,n,content):
    c_len = len(c)
    if c_len == 1:
        return 1  # c中只有一个节点，属性内聚为1

    D_before = (c_len - 1) * c_len / 2
    sum_added = 0
    content_n = content[n]
    for u in c:
        content_u = content[u]

        sum_added += cosine_similarity_vector(content_u, content_n,u,n)
    sum_after = AC_before*D_before + sum_added*2

    D_after = (c_len + 1) * c_len / 2
    del_AC=sum_after/D_after-AC_before
    return del_AC
def cal_after_AC_add_node(AC_before,c,n,content):
    c_len = len(c)
    if c_len == 1:
        return 1  # c中只有一个节点，属性内聚为1

    D_before = (c_len - 1) * c_len / 2
    sum_added = 0
    content_n = content[n]
    for u in c:
        content_u = content[u]

        sum_added += cosine_similarity_vector(content_u, content_n, u, n)
    sum_after = AC_before * D_before + sum_added * 2

    D_after = (c_len + 1) * c_len / 2
    AC = sum_after / D_after
    return AC


def cal_SC_incremental(c_len, ein, eout):
    """
       在增量计算中保证结构内聚性 SC 严格在 [0,1] 内
       """
    if c_len == 1:
        return 1.0

    density = 2 * ein / (c_len * (c_len - 1))
    mean_degree = 2 * ein / c_len
    M = ein / (eout + 1e-8)

    mean_degree_norm = mean_degree / (mean_degree + 1)  # 映射到 [0,1)
    M_norm = M / (M + 1)  # 映射到 [0,1)

    SC_norm = (density + mean_degree_norm + M_norm) / 3
    return SC_norm

def prepare_SC_incremental_data(c, G):
    """
    对当前社区 c 预先统计 c_len, ein, eout 用于增量计算
    """
    c_len = len(c)
    c_set = set(c)

    ein = sum(1 for u, v in G.edges(c) if u in c_set and v in c_set)
    eout = sum(1 for u, v in G.edges(c) if (u in c_set) != (v in c_set))

    return c_len, ein, eout

def cal_delta_SC_add_node_incremental(c_len, ein, eout, c_set, n, G):
    """
    在已知 c_len, ein, eout, c_set 时，快速计算加入 n 后 SC 的增量
    """
    c_len_new = c_len + 1
    added_internal_edges = sum(1 for neighbor in G.neighbors(n) if neighbor in c_set)
    added_external_edges = sum(1 for neighbor in G.neighbors(n) if neighbor not in c_set)

    ein_new = ein + added_internal_edges
    eout_new = eout - added_internal_edges + added_external_edges

    SC_before = cal_SC_incremental(c_len, ein, eout)
    SC_after = cal_SC_incremental(c_len_new, ein_new, eout_new)

    return SC_after - SC_before
def cal_SC_after_add_node_incremental(c_len, ein, eout, c_set, n, G):
    """
    在已知 c_len, ein, eout, c_set 时，快速计算加入 n 后 SC 的增量
    """
    c_len_new = c_len + 1
    added_internal_edges = sum(1 for neighbor in G.neighbors(n) if neighbor in c_set)
    added_external_edges = sum(1 for neighbor in G.neighbors(n) if neighbor not in c_set)

    ein_new = ein + added_internal_edges
    eout_new = eout - added_internal_edges + added_external_edges

    # SC_before = cal_SC_incremental(c_len, ein, eout)
    SC_after = cal_SC_incremental(c_len_new, ein_new, eout_new)

    return SC_after




def query_label(node1,dataset,mini_dataset=None):
    '''

    :
    '''
    if mini_dataset is not None:
        fileLabels = f'label_file_{mini_dataset}.csv'
    else:
        fileLabels = 'label_file.csv'
    mainpath = './data'
    filepath = mainpath + "/" + dataset + "/"

    true_labels = pd.read_csv(filepath + fileLabels, header=None)
    true_labels = true_labels[0].tolist()
    return true_labels[int(node1)]


def get_divrank_nodes(dataset,mini_dataset=None):

    # divrank
    dataset = dataset.lower()
    # dataset_dir = f"./data/{dataset}/"
    # if dataset in ["cora", "citeseer"]:
    #     content_file = os.path.join(dataset_dir, f"{dataset}.content")
    #     cites_file = os.path.join(dataset_dir, f"{dataset}.cites")
    #     G, labels = load_data_from_2file(content_file, cites_file)
    #     idx_features_labels = np.genfromtxt(content_file, dtype=str)
    # else:
    #     undirected_A, C, labels, _=load_data(dataset,mini_dataset)
    #     G = nx.from_numpy_array(undirected_A)
    #     idx_features_labels = result_time_08281704_LOSS_397356.76515277824
    undirected_A, C, labels, _ = load_data(dataset, mini_dataset)
    G = nx.from_numpy_array(undirected_A)
    # idx_features_labels = result_time_08281704_LOSS_397356.76515277824

    # node_id -> label 映射
    # idx_features_labels = np.genfromtxt(content_file, dtype=str)

    # if idx_features_labels is not result_time_08281704_LOSS_397356.76515277824:
    #     # node_raw_ids = idx_features_labels[:, 0]
    #     node_labels = idx_features_labels[:, -1]
    #
    # else:
        # node_raw_ids=range(len(labels))
    node_labels = labels
    id2label = dict(enumerate(node_labels))
    all_labels = set(node_labels)
    print(all_labels)
    print("准备开始divrank")
    # DivRank 计算

    scores_dr = divrank(G)
    print("准备找最小的k")
    # 找最小K
    min_k, covered_labels = find_minimal_k(scores_dr, id2label, all_labels)

    print(f"覆盖所有 {len(all_labels)} 类所需的最小K值: {min_k}")

    # 打印前min_k个节点
    sorted_nodes = sorted(scores_dr, key=lambda n: scores_dr[n], reverse=True)
    print(f"\n# DivRank 前 {min_k} 节点")
    top_indices=sorted_nodes[:min_k]
    for i, n in enumerate(top_indices):
        print(f"{i+1}. 节点 {n}, 分数 {scores_dr[n]:.6f}, 标签 {id2label[n]}")
    return top_indices,min_k


# def distribute_no_allocate_nodes(communitys)
def process_selected_nodes(expan_budget,graph, dataset, mini_dataset,top_indices, selected_nodes, G, content,
                           max_iter=50, tol=1e-2, diff_tol=0.5):
    adj_list = {v: set(G.neighbors(v)) for v in G.nodes}
    dic_label_nodes = defaultdict(set)
    communitys = []
    c_labels = []
    queried_nodes_to_community={}


    queried_nodes = set()

    # 去重节点清理并初始化查询
    selected_nodes_clean = [int(n) if isinstance(n, np.ndarray) else int(n) for n in selected_nodes]
    # print("selected_ndoes_clean",selected_nodes_clean)
    print("top_indices",top_indices)
    # selected_nodes_clean=None
    nodes_selected_and_top = list(set(selected_nodes_clean).union(set(top_indices)))
    # nodes_selected_and_top = list(set(top_indices))
    query_count = 0

       # 加入一跳邻居（只加入未 query 过的邻居）
    for node in top_indices:
        label=query_label(node,dataset,mini_dataset)
        queried_nodes.add(node)
        queried_nodes_to_community[node]=label    #topk 采用的预算已经考虑了，所以不用query_count+1
        dic_label_nodes[label].add(node)
        for neighbor in adj_list[node]:
            if neighbor not in queried_nodes:

                dic_label_nodes[label].add(neighbor)
                queried_nodes.add(neighbor)
                queried_nodes_to_community[neighbor] = label

                # ========== 查询顶点及其邻居 ==========
    for node in nodes_selected_and_top:   #selected_nodes与topk都不会消耗expan_budget
        if node not in queried_nodes:
            label = query_label(node, dataset, mini_dataset)
            dic_label_nodes[label].add(node)
            queried_nodes.add(node)
            queried_nodes_to_community[node]=label
    # ========== 社区扩展 ==========
    num_communities = len(dic_label_nodes)
    communities_expanded = 0  #此刻已经扩展的社区

    for l, community in dic_label_nodes.items():
        # if query_count >= expan_budget:
        #     for l, community in dic_label_nodes.items():
        #         if l not in c_labels:
        #             c_labels.append(l)
        #             communitys.append(list(community))
        #
        #     queried_nodes_len = len(queried_nodes)
        #     return communitys, c_labels, queried_nodes_len, queried_nodes, queried_nodes_to_community
        s = community.copy()
        c_len, ein, eout = prepare_SC_incremental_data(s, G)
        Ch = []
        SC_first = cal_SC_incremental(c_len, ein, eout)
        AC_first = cal_AC(s, content)

        prev_MSD = -np.inf

        # 计算可用预算并分配，communities_remaining此时还未扩展的社区
        communities_remaining = num_communities - communities_expanded
        query_budget_per_community = max((expan_budget - query_count) // communities_remaining, 0)
        community_query_count = 0

        for _ in range(max_iter):
            candidates_AS = []
            neighbors_to_consider = set()
            #考虑二跳，弱链接
            for node in s:
                neighbors_to_consider.update(adj_list[node])
            neighbors_to_consider -= s
            neighbors_two_hot_to_consider = neighbors_to_consider.copy()
            for node in neighbors_two_hot_to_consider:
                neighbors_to_consider.update(adj_list[node])
            neighbors_to_consider -= s
            # neighbors_two_hot_to_consider = neighbors_to_consider.copy()
            # for node in neighbors_two_hot_to_consider:
            #     neighbors_to_consider.update(adj_list[node])
            # neighbors_to_consider -= s
            for neighbor in neighbors_to_consider:
                if neighbor in queried_nodes:
                    continue

                del_AC = cal_del_AC_add_node(AC_first, s, neighbor, content)
                del_SC = cal_delta_SC_add_node_incremental(c_len, ein, eout, s, neighbor, G)
                del_AS = del_AC + del_SC
                AC_after = AC_first + del_AC
                SC_after = SC_first + del_SC

                if del_AS > 0:
                    candidates_AS.append((neighbor, del_AS))

                # 不一致性触发标注（考虑 diff_tol）
                if query_count < expan_budget:
                    if (del_AS < 0 or abs(AC_after - SC_after) > diff_tol):
                        if community_query_count < query_budget_per_community and query_count < expan_budget and neighbor not in queried_nodes:
                            label = query_label(neighbor, dataset, mini_dataset)
                            dic_label_nodes[label].add(neighbor)
                            queried_nodes.add(neighbor)
                            queried_nodes_to_community[neighbor] = label
                            query_count += 1
                            community_query_count += 1

            if not candidates_AS:
                break

            # 排序并按收益最大扩展
            candidates_AS_sorted = sorted(candidates_AS, key=lambda x: x[1], reverse=True)
            MSD = -np.inf
            Copt = s.copy()

            for i, _ in candidates_AS_sorted:
                s_new = s | {i}
                c_len, ein, eout = prepare_SC_incremental_data(s_new, G)
                SC_after = cal_SC_incremental(c_len, ein, eout)
                AC_after = cal_AC(s_new, content)
                AS = SC_after + AC_after

                if AS > MSD:
                    MSD = AS
                    Copt = s_new.copy()
            # print("DEBUG: MSD =", MSD, "prev_MSD =", prev_MSD)
            if np.isnan(prev_MSD):
                prev_MSD = np.nan_to_num(MSD, nan=1e-5)

            if abs(MSD - prev_MSD) / (abs(prev_MSD) + 1e-8) < tol:
                break

            s = Copt.copy()
            Ch.append(s.copy())
            prev_MSD = MSD

        print(f"第 {l} 个社区扩展完成，节点数: {len(s)}")
        communitys.append(list(s))
        if l not in c_labels:
            c_labels.append(l)

        communities_expanded += 1



    return communitys, c_labels, query_count, queried_nodes,queried_nodes_to_community


def redistribute_duplicates_to_max_AS_per(c_labels, queried_nodes_to_community, queried_nodes, communitys, content, G):
    from collections import defaultdict

    element_to_indices = defaultdict(set)

    # 缓存社区指标
    cached_AC = {}
    cached_SC_pre = {}
    cached_SC = {}

    for idx, community in enumerate(communitys):
        s = set(community)
        cached_AC[idx] = cal_AC(s, content)
        c_len, ein, eout = prepare_SC_incremental_data(s, G)
        cached_SC_pre[idx] = (c_len, ein, eout)
        cached_SC[idx] = cal_SC_incremental(c_len, ein, eout)

    # 建立元素在哪些社区中出现
    for idx, community in enumerate(communitys):
        for element in community:
            element_to_indices[element].add(idx)

    dup_num = 0

    for element, indices in element_to_indices.items():
        if len(indices) > 1:
            dup_num += 1

            # 查询节点则直接用标签指派
            if element in queried_nodes_to_community.keys():
                label = queried_nodes_to_community[element]
                keep_idx = c_labels.index(label)
                for idx in indices:
                    if idx != keep_idx:
                        try:
                            communitys[idx].remove(element)
                        except ValueError:
                            pass
                continue

            # 否则基于 AS_per 计算最优社区
            max_AS_per = -float('inf')
            max_idx = -1
            for idx in indices:
                c_exclude = communitys[idx].copy()
                c_exclude.remove(element)

                AC_before = cached_AC[idx]
                AC_after = cal_after_AC_add_node(AC_before, c_exclude, element, content)
                c_len, ein, eout = cached_SC_pre[idx]
                SC_before = cached_SC[idx]
                SC_after = cal_SC_after_add_node_incremental(c_len, ein, eout, c_exclude, element, G)

                AS_before = AC_before + SC_before
                AS_after = AC_after + SC_after
                AS_per = (AS_after - AS_before) / (AS_before + 1e-8)

                if AS_per > max_AS_per:
                    max_AS_per = AS_per
                    max_idx = idx

            # 如果找到了最优社区，则保留，其他删
            if max_idx != -1:
                for idx in indices:
                    if idx != max_idx:
                        try:
                            communitys[idx].remove(element)
                        except ValueError:
                            pass

    # 再次确保 queried_nodes 正确加入对应社区，避免重复
    # 最后确保 queried_nodes 只出现在它应在的社区中（彻底唯一化）
    for n, c in queried_nodes_to_community.items():
        target_idx = c_labels.index(c)
        for idx, community in enumerate(communitys):
            if n in community:
                if idx != target_idx:
                    community.remove(n)
        if n not in communitys[target_idx]:
            communitys[target_idx].append(n)

    return communitys, dup_num


import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

def missing_nodes_to_community(communitys, covered_nodes_set, missing_nodes, G, content):
    """
    将遗漏节点分配到最合适的社区：
    1. 优先基于 modularity 增益 (ΔQ 增量公式 + 社区度缓存)
    2. 如果 ΔQ 均无提升，再用内容相似度分配
    """

    # ----------------------
    # 初始化缓存
    # ----------------------
    m = G.number_of_edges()  # 总边数
    degrees = dict(G.degree())  # 节点度缓存

    # 保证 communitys 内部是 set（提高查找效率，避免重复节点）
    communitys = [set(comm) for comm in communitys]

    # 每个社区的总度（sum_tot）
    community_degree_sum = {i: sum(degrees[n] for n in c) for i,c in enumerate(communitys)}
    community_neighbors = {}
    community_content_avg = {}

    for i,comm in enumerate(communitys):
        neighbors = set()
        vectors = []
        for u in comm:
            neighbors.update(G.neighbors(u))
            vectors.append(content[u])
        community_neighbors[i]=neighbors
        avg_vec = np.mean(vectors, axis=0) if vectors else np.zeros_like(content[0])
        community_content_avg[i]=avg_vec

    # ----------------------
    # 处理每个遗漏节点
    # ----------------------
    for m_node in missing_nodes:
        content_m = content[m_node]
        k_node = degrees.get(m_node, 0)
        m_neighbors = set(G.neighbors(m_node))
        # 情况 1：孤立节点（完全没有边）
        if k_node == 0:

            print(f" 孤立节点 {m_node}，使用内容相似度分配")
            sims = cosine_similarity([content_m],  list(community_content_avg.values())).flatten()
            best_idx = int(np.argmax(sims))
            communitys[best_idx].add(m_node)
            # 增量更新社区均值向量

            n_old = len(communitys[best_idx]) - 1  # 之前的节点数
            community_content_avg[best_idx] = (community_content_avg[best_idx] * n_old + content[m_node]) / (n_old + 1)
            community_neighbors[best_idx].update(m_neighbors)
            continue

        # 情况 2：基于 ΔQ 公式计算最优社区
        #保留与节点m有边的社区的id
        relevant_communities = [idx for idx, comm in enumerate(communitys) if len(m_neighbors & comm) > 0]

        best_idx = None
        best_dq = -1e9
        for idx in relevant_communities:

            sum_tot = community_degree_sum[idx]

            # 节点与该社区的内部边数 k_in
            # k_in = sum(1 for nbr in G.neighbors(m_node) if nbr in community)
            k_in = len(m_neighbors & communitys[idx])

            # ΔQ 公式
            dq = (k_in / (2*m)) - (k_node * sum_tot) / (4 * m * m)

            if dq > best_dq:
                best_dq = dq
                best_idx = idx

        # 如果 ΔQ 有提升，则加入该社区
        if best_idx is not None and best_dq > 1e-6:
            communitys[best_idx].add(m_node)
            community_degree_sum[best_idx] += k_node
            # 增量更新均值向量
            n_old = len(communitys[best_idx]) - 1
            community_content_avg[best_idx] = (community_content_avg[best_idx] * n_old + content_m) / (n_old + 1)
            community_neighbors[best_idx].update(m_neighbors)
            print(f"节点 {m_node} 分配到社区 {best_idx} (ΔQ={best_dq:.4f})")

        else:
            # ΔQ 无提升 → 用内容相似度
            sims = cosine_similarity([content_m], list(community_content_avg.values())).flatten()
            best_idx = int(np.argmax(sims))
            communitys[best_idx].add(m_node)
            # 增量更新均值向量
            n_old = len(communitys[best_idx]) - 1
            community_content_avg[best_idx] = (community_content_avg[best_idx] * n_old + content_m) / (n_old + 1)
            community_neighbors[best_idx].update(m_neighbors)
            print(f"节点 {m_node} 通过内容相似度分配到社区 {best_idx}")
    #         集合转为list
    communitys = [list(comm) for comm in communitys]
    return communitys






import matplotlib.pyplot as plt

def evaluate_on_test_nodes(graph,
                           queried_nodes,
                           predicted_labels,
                           true_labels,
                           c_labels,
                           final_communitys,
                           metric_record=None,
                           query_record=None):

    # 1️构造 id -> community label 映射
    id_to_label = {}
    for idx, community in enumerate(final_communitys):
        for nid in community:
            id_to_label[nid] = c_labels[idx]

    # 2️生成预测标签列表并筛选出测试集
    predicted_labels_all = []
    test_nodes = []
    test_predicted_labels = []
    test_true_labels = []

    for node in graph.nodes():
        pred_label = id_to_label.get(node, -1)  # 未分配社区标记为 -1
        predicted_labels_all.append(pred_label)

        if node not in queried_nodes:
            test_nodes.append(node)
            test_predicted_labels.append(pred_label)
            test_true_labels.append(true_labels[node])


    # 调用已有 evaluate_metrics，仅在测试集上评估
    metrics = evaluate_metrics(
        test_predicted_labels,
        test_true_labels,
        graph=None
    )

    return metrics
