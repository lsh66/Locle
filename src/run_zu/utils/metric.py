from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score,f1_score,jaccard_score
import numpy as np
import networkx as nx

def evaluate_metrics(predicted_labels, ground_truth_labels, graph=None):
    """
    评价社区检测结果：
    - 准确率（Accuracy）
    - NMI (Normalized Mutual Information)
    - ARI (Adjusted Rand Index)
    - 模块度（Modularity，可选，需要提供 graph）

    参数：
    ----------
    predicted_labels : list or np.ndarray
        社区预测标签（长度为节点数，表示每个节点属于的社区 id）
    ground_truth_labels : list or np.ndarray
        真实标签（同上）
    graph : nx.Graph, optional
        若需要计算模块度时提供对应的无向图

    返回：
    ----------
    metrics : dict
        包含 accuracy, NMI, ARI, modularity (可选)
    """

    metrics = {}

    # === 1️⃣ Accuracy (基于最大匹配) ===
    from scipy.optimize import linear_sum_assignment

    def compute_accuracy(pred, true):
        pred = np.array(pred)
        true = np.array(true)
        accuracy = np.mean(pred == true)
        return accuracy

    accuracy = compute_accuracy(np.array(predicted_labels), np.array(ground_truth_labels))
    metrics['Accuracy'] = accuracy

    # === 2️⃣ NMI ===
    nmi = normalized_mutual_info_score(ground_truth_labels, predicted_labels)
    metrics['NMI'] = nmi

    # === 3️⃣ ARI ===
    ari = adjusted_rand_score(ground_truth_labels, predicted_labels)
    metrics['ARI'] = ari

    # F1 Score (macro)
    f1 = f1_score(ground_truth_labels, predicted_labels, average='macro')
    metrics["F1_macro"] = f1

    # Jaccard Similarity (macro)
    # # scikit-learn 的 jaccard_score 仅支持二分类/多标签问题，需要设置 average='macro'
    # jaccard = jaccard_score(ground_truth_labels, predicted_labels, average='macro')
    # metrics["Jaccard_macro"] = jaccard


    # # === 4️⃣ Modularity (若提供 graph) ===
    # if graph is not None:
    #     # 转换 predicted_labels => 社区划分
    #     communities = {}
    #     for idx, label in enumerate(predicted_labels):
    #         communities.setdefault(label, set()).add(idx)
    #     partition = list(communities.values())
    #     modularity = nx.algorithms.community.quality.modularity(graph, partition)
    #     metrics['Modularity'] = modularity

    return metrics


# import networkx as nx
#
# 
# predicted_labels=[1,2,3,1,2]
# ground_truth_labels=[1,2,3,1,3]
# G=nx.Graph()
# G.add_edges_from([(1,2),(2,3),(1,3),(0,4)])
# # 假设 predicted_labels 和 ground_truth_labels 为长度为 n 的数组
# metrics = evaluate_metrics(predicted_labels, ground_truth_labels, graph=G)
#
# for k, v in metrics.items():
#     print(f"{k}: {v:.4f}")
