#https://github.com/recruit-tech/summpy/blob/master/summpy/misc/divrank.py
#两个版本
#divrank：纯 Python + NetworkX 实现（适合小图）。
#
# divrank_scipy：稀疏矩阵 + SciPy 实现（适合大图）。
#k取10个节点，cora的divrank的效果比pagerank好


import networkx as nx
import numpy as np
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for

from run_zu.utils.load_dataset import load_data


@not_implemented_for('multigraph')
def divrank(G, alpha=0.25, d=0.85, personalization=None,
            max_iter=100, tol=1.0e-6, nstart=None, weight='weight',
            dangling=None):
    """
    Returns the DivRank (Diverse Rank) of the nodes in the graph.
    This code is based on networkx.pagerank.

    Args:
      alpha: controls strength of self-link [0.0-1.0]
      d: the damping factor

    Reference:
      Qiaozhu Mei, Jian Guo, Dragomir Radev,
      DivRank: the Interplay of Prestige and Diversity in Information Networks,
      http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.174.7982
    """

    if len(G) == 0:
        return {}

    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G

    # Create a copy in (right) stochastic form
    W = nx.stochastic_graph(D, weight=weight)
    N = W.number_of_nodes()

    # self-link (DivRank)
    for n in W.nodes():
        for n_ in W.nodes():
            if n != n_:
                if n_ in W[n]:
                    W[n][n_][weight] *= alpha
            else:
                if n_ not in W[n]:
                    W.add_edge(n, n_)
                W[n][n_][weight] = 1.0 - alpha

    # Choose fixed starting vector if not given
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        s = float(sum(nstart.values()))
        x = {k: v / s for k, v in nstart.items()}

    if personalization is None:
        p = dict.fromkeys(W, 1.0 / N)
    else:
        missing = set(G) - set(personalization)
        if missing:
            raise NetworkXError(f'Personalization dictionary missing nodes: {missing}')
        s = float(sum(personalization.values()))
        p = {k: v / s for k, v in personalization.items()}

    if dangling is None:
        dangling_weights = p
    else:
        missing = set(G) - set(dangling)
        if missing:
            raise NetworkXError(f'Dangling node dictionary missing nodes: {missing}')
        s = float(sum(dangling.values()))
        dangling_weights = {k: v / s for k, v in dangling.items()}

    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

    # power iteration
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = d * sum(xlast[n] for n in dangling_nodes)
        for n in x:
            D_t = sum(W[n][nbr][weight] * xlast[nbr] for nbr in W[n])
            for nbr in W[n]:
                x[nbr] += (
                    d * (W[n][nbr][weight] * xlast[nbr] / D_t) * xlast[n]
                )
            x[n] += danglesum * dangling_weights[n] + (1.0 - d) * p[n]

        err = sum(abs(x[n] - xlast[n]) for n in x)
        if err < N * tol:
            return x
    raise NetworkXError(f'divrank: power iteration failed to converge in {max_iter} iterations.')

def plot_scores(G, scores, title):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 8))
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=[3000 * scores[n] for n in G.nodes()],
        node_color=[scores[n] for n in G.nodes()],
        cmap=plt.cm.viridis,
        alpha=0.8
    )
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=6, font_color="white")
    plt.colorbar(nodes)
    plt.title(title)
    plt.axis("off")
    plt.show()
import matplotlib.pyplot as plt

def find_minimal_k(scores_dr, id2label, all_labels):
    """找到最小K使得覆盖率=100%"""
    sorted_nodes = sorted(scores_dr, key=lambda n: scores_dr[n], reverse=True)
    K = len(all_labels)  # 初始化为类别数
    while K <= len(sorted_nodes):
        top_nodes = sorted_nodes[:K]
        covered = set(id2label[n] for n in top_nodes)
        if len(covered) == len(all_labels):
            return K, covered
        K += 1
    return None, set()
def load_data_from_2file(content_file, cites_file):
    idx_features_labels = np.genfromtxt(content_file, dtype=str)
    node_ids = idx_features_labels[:, 0]
    labels = idx_features_labels[:, -1]

    # node id -> index
    node_id_map = {j: i for i, j in enumerate(node_ids)}

    G = nx.Graph()
    # 加节点（用连续编号）
    G.add_nodes_from(range(len(node_ids)))

    # 加边（转成连续编号）
    with open(cites_file, "r") as f:
        for line in f:
            src, dst = line.strip().split()
            if src in node_id_map and dst in node_id_map:
                G.add_edge(node_id_map[src], node_id_map[dst])

    # 构造 id->label 映射
    id2label = {i: labels[i] for i in range(len(labels))}

    return G, labels

if __name__ == "__main__":
    # 数据路径
    content_file = "./data/cora/cora.content"
    cites_file = "./data/cora/cora.cites"
    dataset="cora"
    G, labels = load_data_from_2file(content_file, cites_file)
    # undirected_A, C, true_labels, NoComm = load_data(dataset, mini_dataset=result_time_08281704_LOSS_397356.76515277824)
    # G=nx.from_numpy_array(undirected_A)   #k为34

    # node_id -> label 映射
    idx_features_labels = np.genfromtxt(content_file, dtype=str)
    node_raw_ids = idx_features_labels[:, 0]
    node_id_map = {raw_id: i for i, raw_id in enumerate(node_raw_ids)}
    node_labels = idx_features_labels[:, -1]
    # id2label = {i: node_labels[i] for i in range(len(node_labels))}
    id2label = dict(enumerate(node_labels))
    # id2label = {nid: lab for nid, lab in zip(node_ids, node_labels)}
    all_labels = set(node_labels)

    # DivRank 计算
    scores_dr = divrank(G)

    # 找最小K
    min_k, covered_labels = find_minimal_k(scores_dr, id2label, all_labels)

    print(f"覆盖所有 {len(all_labels)} 类所需的最小K值: {min_k}")
    print("覆盖到的类别:", covered_labels)

    # 打印前min_k个节点
    sorted_nodes = sorted(scores_dr, key=lambda n: scores_dr[n], reverse=True)
    print(f"\n# DivRank 前 {min_k} 节点")
    for i, n in enumerate(sorted_nodes[:min_k]):
        print(f"{i+1}. 节点 {n}, 分数 {scores_dr[n]:.6f}, 标签 {id2label[n]}")
