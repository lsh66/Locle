"""
Evaluation metrics for overlapping community detection.

Author: ANS-OCD project
"""
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score
from sklearn.metrics.cluster import contingency_matrix
from typing import List, Dict


def membership_matrix_from_sets(communities: List[List[int]], num_nodes: int) -> np.ndarray:
    """
    Convert a list of communities (sets of node ids) to a binary membership matrix.

    Parameters
    ----------
    communities : List[List[int]]
        Each inner list contains node ids belonging to one community.
    num_nodes : int
        Total number of nodes.

    Returns
    -------
    membership : np.ndarray, shape [num_nodes, num_communities]
        Binary membership matrix.
    """
    membership = np.zeros((num_nodes, len(communities)), dtype=np.float32)
    for cid, nodes in enumerate(communities):
        for n in nodes:
            if 0 <= n < num_nodes:
                membership[n, cid] = 1.0
    return membership


def omega_index(pred_membership: np.ndarray, true_membership: np.ndarray) -> float:
    """
    Compute the Omega Index (also known as the Fuzzy Rand Index) for overlapping
    community detection. It measures the agreement of co-membership pairs.

    Parameters
    ----------
    pred_membership : np.ndarray, shape [N, K1]
        Predicted binary membership matrix.
    true_membership : np.ndarray, shape [N, K2]
        Ground-truth binary membership matrix.

    Returns
    -------
    omega : float
        Omega index in [0, 1].
    """
    N = pred_membership.shape[0]
    # Co-membership counts: how many communities two nodes share.
    # Vectorized: the previous implementation used a double Python loop over all
    # node pairs, which is O(N^2) interpreted iterations -- on pubmed (N=19717)
    # that is ~1.9e8 iterations and took >10 minutes (and 6+ GB for two float64
    # [N,N] matrices). Computing the comparison with numpy keeps the same result
    # but runs in seconds and uses far less memory (float32 matmul + boolean).
    pred = np.asarray(pred_membership, dtype=np.float32)
    true = np.asarray(true_membership, dtype=np.float32)
    pred_co = (pred @ pred.T) > 0
    true_co = (true @ true.T) > 0

    # Compare only the strict upper triangle (unordered node pairs).
    agree = np.triu((pred_co == true_co), k=1)
    agreements = int(np.count_nonzero(agree))
    total_pairs = N * (N - 1) // 2
    return float(agreements / total_pairs) if total_pairs > 0 else 1.0


def average_jaccard(pred_membership: np.ndarray, true_membership: np.ndarray) -> float:
    """
    Compute average Jaccard similarity between predicted and true community sets
    per node.

    Parameters
    ----------
    pred_membership : np.ndarray, shape [N, K1]
    true_membership : np.ndarray, shape [N, K2]

    Returns
    -------
    avg_jaccard : float
        Average Jaccard index across nodes.
    """
    scores = []
    for i in range(pred_membership.shape[0]):
        p = pred_membership[i].astype(bool)
        t = true_membership[i].astype(bool)
        inter = (p & t).sum()
        union = (p | t).sum()
        if union == 0:
            scores.append(1.0 if inter == 0 else 0.0)
        else:
            scores.append(inter / union)
    return float(np.mean(scores))


def overlapping_f1(pred_membership: np.ndarray, true_membership: np.ndarray) -> Dict[str, float]:
    """
    Compute micro/macro F1 for overlapping community membership.

    Parameters
    ----------
    pred_membership : np.ndarray, shape [N, K]
    true_membership : np.ndarray, shape [N, K]

    Returns
    -------
    f1_scores : Dict[str, float]
        Micro and macro F1 scores.
    """
    pred_flat = pred_membership.flatten()
    true_flat = true_membership.flatten()
    return {
        "f1_micro": f1_score(true_flat, pred_flat, average="micro", zero_division=0),
        "f1_macro": f1_score(true_flat, pred_flat, average="macro", zero_division=0),
    }


def relative_membership(probs: np.ndarray, alpha: float = 0.7,
                        m_max: int = 3, ensure_top1: bool = True) -> np.ndarray:
    """Overlapping membership: keep community c when p[n,c] >= alpha*max_j p[n,j].

    A relative rule is needed because a fixed 0.5 threshold almost never fires
    on the non-top-1 columns of a K-community sigmoid output, which is exactly
    why the legacy argmax decision collapses to one community per node.
    """
    probs = np.asarray(probs, dtype=np.float32)
    n, k = probs.shape
    if k == 0:
        return np.zeros((n, 0), dtype=np.float32)
    m = max(1, min(int(m_max), k))
    order = np.argsort(-probs, axis=1)[:, :m]
    top = np.take_along_axis(probs, order, axis=1)
    pmax = top[:, :1]
    keep = (top >= alpha * pmax) & (pmax > 0) & (top > 0)
    vals = np.where(keep, np.clip(top, 0.0, 1.0), 0.0).astype(np.float32)
    out = np.zeros((n, k), dtype=np.float32)
    rows = np.arange(n)[:, None]
    out[rows, order] = vals
    if ensure_top1:
        out[np.arange(n), order[:, 0]] = 1.0
    return (out > 0).astype(np.float32)


def hungarian_align_membership(pred_membership: np.ndarray,
                               true_membership: np.ndarray):
    """Align predicted communities to GT communities by best Jaccard.

    Returns (aligned_pred, mean_matched_jaccard); aligned_pred keeps the FULL
    ground-truth column count so metrics stay valid when K_pred != K_true.
    """
    from scipy.optimize import linear_sum_assignment

    pred = np.asarray(pred_membership, dtype=np.float32)
    true = np.asarray(true_membership, dtype=np.float32)
    n, kp = pred.shape
    kt = true.shape[1]
    if kp == 0 or kt == 0:
        return np.zeros((n, kt), dtype=np.float32), 0.0
    score = np.zeros((kp, kt), dtype=np.float64)
    for i in range(kp):
        a = pred[:, i] > 0
        for j in range(kt):
            b = true[:, j] > 0
            union = np.count_nonzero(a | b)
            score[i, j] = np.count_nonzero(a & b) / union if union else 0.0
    ri, ci = linear_sum_assignment(-score)
    aligned = np.zeros((n, kt), dtype=np.float32)
    for i, j in zip(ri, ci):
        aligned[:, j] = pred[:, i]
    return aligned, (float(score[ri, ci].mean()) if len(ri) else 0.0)


def evaluate_overlapping(pred_communities: List[List[int]],
                         true_communities: List[List[int]],
                         num_nodes: int) -> Dict[str, float]:
    """
    High-level evaluation wrapper for overlapping community detection.

    Parameters
    ----------
    pred_communities : List[List[int]]
        Predicted communities.
    true_communities : List[List[int]]
        Ground-truth communities.
    num_nodes : int
        Total number of nodes.

    Returns
    -------
    metrics : Dict[str, float]
        Dictionary containing Omega Index, average Jaccard, and micro/macro F1.
    """
    pred_membership = membership_matrix_from_sets(pred_communities, num_nodes)
    true_membership = membership_matrix_from_sets(true_communities, num_nodes)

    f1_scores = overlapping_f1(pred_membership, true_membership)
    metrics = {
        "omega_index": omega_index(pred_membership, true_membership),
        "avg_jaccard": average_jaccard(pred_membership, true_membership),
        "f1_micro": f1_scores["f1_micro"],
        "f1_macro": f1_scores["f1_macro"],
    }
    return metrics


if __name__ == "__main__":
    # Self-test: two overlapping communities with known ground truth.
    N = 6
    pred = [[0, 1, 2, 3], [2, 3, 4, 5]]
    true = [[0, 1, 2], [2, 3, 4, 5]]
    metrics = evaluate_overlapping(pred, true, N)
    print("Evaluation metrics:", metrics)
