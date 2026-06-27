"""
DivRank-based representative node selection for overlapping community detection.

For each initial cluster (from subspace clustering), this module runs DivRank on
the induced subgraph to select high-influence and diverse nodes as community seeds.

Author: ANS-OCD project
"""
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple

from ANS_OCD.config import DIVRANK_CONFIG


def _divrank_scores(G: nx.Graph, alpha: float = 0.25, d: float = 0.85,
                    max_iter: int = 1000, tol: float = 1e-6) -> Dict[int, float]:
    """
    Compute DivRank scores for nodes in a NetworkX graph.

    DivRank is a variant of PageRank that adds a self-link term (controlled by
    alpha) to encourage diversity among top-ranked nodes. The implementation
    follows the algorithm in src/run_zu/divrank.py.

    Parameters
    ----------
    G : nx.Graph
        Undirected graph (or will be treated as such).
    alpha : float
        Self-link strength in [0, 1]. Lower values penalize similar nodes more.
    d : float
        Damping factor (standard PageRank teleport probability).
    max_iter : int
        Maximum power iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    scores : Dict[int, float]
        Mapping from node id to DivRank score.
    """
    if len(G) == 0:
        return {}

    D = G.to_directed() if not G.is_directed() else G
    W = nx.stochastic_graph(D, weight="weight")
    N = W.number_of_nodes()
    weight = "weight"

    # Add self-link term.
    for n in W.nodes():
        for n_ in W.nodes():
            if n != n_:
                if n_ in W[n]:
                    W[n][n_][weight] *= alpha
            else:
                if n_ not in W[n]:
                    W.add_edge(n, n_)
                W[n][n_][weight] = 1.0 - alpha

    x = dict.fromkeys(W, 1.0 / N)
    p = dict.fromkeys(W, 1.0 / N)
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0.0)
        danglesum = d * sum(xlast[n] for n in dangling_nodes)
        for n in x:
            D_t = sum(W[n][nbr][weight] * xlast[nbr] for nbr in W[n])
            for nbr in W[n]:
                x[nbr] += d * (W[n][nbr][weight] * xlast[nbr] / D_t) * xlast[n]
            x[n] += danglesum * p[n] + (1.0 - d) * p[n]

        err = sum(abs(x[n] - xlast[n]) for n in x)
        if err < N * tol:
            return x
    raise RuntimeError(f"DivRank failed to converge in {max_iter} iterations.")


def select_representatives_per_cluster(adj_matrix: np.ndarray,
                                       cluster_labels: np.ndarray,
                                       reps_per_cluster: int = DIVRANK_CONFIG["reps_per_cluster"],
                                       alpha: float = DIVRANK_CONFIG["lambda_param"],
                                       d: float = 0.85,
                                       max_iter: int = DIVRANK_CONFIG["max_iters"],
                                       tol: float = DIVRANK_CONFIG["tol"]) -> Tuple[Dict[int, List[int]], Dict[int, Dict[int, float]]]:
    """
    Select representative nodes for each cluster using per-cluster DivRank.

    Parameters
    ----------
    adj_matrix : np.ndarray, shape [N, N]
        Full adjacency matrix of the graph.
    cluster_labels : np.ndarray, shape [N]
        Integer cluster labels from subspace clustering.
    reps_per_cluster : int
        Number of representative nodes to select from each cluster.
    alpha : float
        DivRank self-link parameter.
    d : float
        PageRank damping factor.
    max_iter : int
        Maximum iterations for DivRank.
    tol : float
        Convergence tolerance.

    Returns
    -------
    representatives : Dict[int, List[int]]
        Mapping from cluster id to sorted list of representative node ids.
    scores : Dict[int, Dict[int, float]]
        Mapping from cluster id to {node_id: DivRank score}.
    """
    N = adj_matrix.shape[0]
    clusters = np.unique(cluster_labels)
    representatives = {}
    all_scores = {}

    for cid in clusters:
        members = np.where(cluster_labels == cid)[0]
        if len(members) == 0:
            continue

        # Induced sub-matrix for this cluster.
        sub_adj = adj_matrix[np.ix_(members, members)]
        G = nx.from_numpy_array(sub_adj)

        # Run DivRank on the induced subgraph.
        sub_scores = _divrank_scores(G, alpha=alpha, d=d, max_iter=max_iter, tol=tol)
        # Map back to original node ids.
        original_scores = {int(members[n]): sub_scores[n] for n in sub_scores}

        # Pick the top-k nodes (ties broken by lower index for reproducibility).
        top = sorted(original_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        k = min(reps_per_cluster, len(top))
        representatives[int(cid)] = [int(node) for node, _ in top[:k]]
        all_scores[int(cid)] = original_scores

    return representatives, all_scores


def get_representative_texts(representatives: Dict[int, List[int]],
                             texts: List[str],
                             top_neighbors: Optional[Dict[int, List[int]]] = None) -> Dict[int, str]:
    """
    Format representative node texts into a compact prompt string per cluster.

    Parameters
    ----------
    representatives : Dict[int, List[int]]
        Mapping from cluster id to representative node ids.
    texts : List[str]
        Raw text for every node.
    top_neighbors : Dict[int, List[int]], optional
        For each representative node, a list of neighbor ids to append as context.

    Returns
    -------
    prompt_blocks : Dict[int, str]
        Mapping from cluster id to a formatted text block.
    """
    blocks = {}
    for cid, nodes in representatives.items():
        lines = [f"Cluster {cid} representative nodes:"]
        for i, node in enumerate(nodes, start=1):
            line = f"  [{i}] Node {node}: {texts[node][:200]}"
            if top_neighbors and node in top_neighbors:
                neighbor_ids = top_neighbors[node][:3]
                line += f" (neighbors: {neighbor_ids})"
            lines.append(line)
        blocks[cid] = "\n".join(lines)
    return blocks


if __name__ == "__main__":
    # Self-test: 3-block graph with cluster labels.
    np.random.seed(0)
    N = 30
    sizes = [10, 10, 10]
    labels = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
    adj = np.zeros((N, N))
    for i in range(3):
        start = sum(sizes[:i])
        end = start + sizes[i]
        block = np.random.rand(sizes[i], sizes[i]) < 0.5
        block = (block + block.T) > 0
        adj[start:end, start:end] = block.astype(float)
    np.fill_diagonal(adj, 0.0)

    reps, scores = select_representatives_per_cluster(adj, labels, reps_per_cluster=3)
    for cid, nodes in reps.items():
        print(f"Cluster {cid} representatives: {nodes}")
        print(f"  scores: { {n: round(scores[cid][n], 4) for n in nodes} }")
