"""
Multi-label label entropy for overlapping community detection.

Unlike the traditional softmax-based Shannon entropy, the implementation below uses
binary cross-entropy per community independently. This allows a node to be
uncertain about membership in multiple communities simultaneously, which is the
core requirement for overlapping community detection.

Author: ANS-OCD project
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Union

from ANS_OCD.config import MULTILABEL_CONFIG, DEVICE


def label_entropy_overlap(probs: Union[np.ndarray, torch.Tensor],
                          sigma: float = MULTILABEL_CONFIG["entropy_sigma"]) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute the overlapping label entropy for each node.

    Math:
        LE_overlap(v_i) = - sum_{j=1}^k [
            y_{ij} * log(y_{ij} + sigma) +
            (1 - y_{ij}) * log(1 - y_{ij} + sigma)
        ]
    where y_{ij} is the Sigmoid probability of node i belonging to community j.

    Parameters
    ----------
    probs : np.ndarray or torch.Tensor, shape [num_nodes, num_communities]
        Independent membership probabilities after Sigmoid, values in [0, 1].
    sigma : float
        Small constant for numerical stability (default 1e-9).

    Returns
    -------
    entropy : np.ndarray or torch.Tensor, shape [num_nodes]
        Per-node overlapping label entropy. Higher means more uncertain.
    """
    if isinstance(probs, torch.Tensor):
        eps = sigma
        probs = torch.clamp(probs, min=eps, max=1.0 - eps)
        entropy = -torch.sum(
            probs * torch.log(probs + eps) +
            (1.0 - probs) * torch.log(1.0 - probs + eps),
            dim=1
        )
    else:
        eps = sigma
        probs = np.clip(probs, eps, 1.0 - eps)
        entropy = -np.sum(
            probs * np.log(probs + eps) +
            (1.0 - probs) * np.log(1.0 - probs + eps),
            axis=1
        )
    return entropy


def label_entropy_from_logits(logits: Union[np.ndarray, torch.Tensor],
                              sigma: float = MULTILABEL_CONFIG["entropy_sigma"]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convenience wrapper: apply Sigmoid first, then compute overlapping label entropy.

    Parameters
    ----------
    logits : np.ndarray or torch.Tensor, shape [num_nodes, num_communities]
        GNN output logits.

    Returns
    -------
    entropy : np.ndarray or torch.Tensor, shape [num_nodes]
    """
    if isinstance(logits, torch.Tensor):
        probs = torch.sigmoid(logits)
    else:
        probs = 1.0 / (1.0 + np.exp(-logits))
    return label_entropy_overlap(probs, sigma=sigma)


def select_high_entropy_nodes(probs: Union[np.ndarray, torch.Tensor],
                                top_ratio: float = 0.1,
                                min_nodes: int = 1,
                                return_scores: bool = False) -> Union[np.ndarray, tuple]:
    """
    Select the nodes with the highest overlapping label entropy.

    Parameters
    ----------
    probs : np.ndarray or torch.Tensor, shape [num_nodes, num_communities]
        Membership probabilities after Sigmoid.
    top_ratio : float
        Ratio of top-entropy nodes to return (e.g., 0.1 for top 10%).
    min_nodes : int
        Minimum number of nodes to return regardless of ratio.
    return_scores : bool
        If True, also return entropy scores for all nodes.

    Returns
    -------
    selected_nodes : np.ndarray
        Indices of selected high-entropy nodes, sorted by entropy descending.
    scores : np.ndarray, optional
        Entropy scores for all nodes, returned if return_scores=True.
    """
    scores = label_entropy_overlap(probs)
    if isinstance(scores, torch.Tensor):
        # Use torch.topk to avoid requiring numpy at runtime.
        k = max(min_nodes, int(torch.ceil(torch.tensor(len(scores) * top_ratio)).item()))
        k = min(k, len(scores))
        _, selected = torch.topk(scores, k, largest=True)
        try:
            selected = selected.detach().cpu().numpy()
        except RuntimeError:
            selected = np.asarray(selected.tolist())
        selected = np.asarray(selected)
    else:
        k = max(min_nodes, int(np.ceil(len(scores) * top_ratio)))
        k = min(k, len(scores))
        selected = np.argsort(scores)[::-1][:k]
    if return_scores:
        return selected, scores
    return selected


if __name__ == "__main__":
    # Self-test: verify that overlapping and certain nodes get distinct entropy values.
    import torch.nn.functional as F

    np.random.seed(42)
    num_nodes = 5
    num_communities = 4

    # Case A: uniform probabilities (maximally uncertain).
    uniform = torch.full((num_nodes, num_communities), 0.5)
    ent_uniform = label_entropy_overlap(uniform)
    print("Entropy for uniform (0.5) probabilities:", ent_uniform.tolist() if isinstance(ent_uniform, torch.Tensor) else ent_uniform)

    # Case B: one-hot-like probabilities (certain single membership).
    onehot = torch.zeros(num_nodes, num_communities)
    for i in range(num_nodes):
        onehot[i, i % num_communities] = 1.0
    ent_onehot = label_entropy_overlap(onehot)
    print("Entropy for one-hot probabilities:", ent_onehot.tolist() if isinstance(ent_onehot, torch.Tensor) else ent_onehot)

    # Case C: overlapping but high-confidence.
    overlap = torch.zeros(num_nodes, num_communities)
    for i in range(num_nodes):
        overlap[i, [0, 2]] = 0.9
    ent_overlap = label_entropy_overlap(overlap)
    print("Entropy for high-confidence overlap:", ent_overlap.tolist() if isinstance(ent_overlap, torch.Tensor) else ent_overlap)

    # High-entropy selection
    probs = torch.rand(num_nodes, num_communities)
    selected, scores = select_high_entropy_nodes(probs, top_ratio=0.4, return_scores=True)
    print("Selected high-entropy nodes:", selected.tolist() if isinstance(selected, torch.Tensor) else selected)
    print("All entropy scores:", scores.tolist() if isinstance(scores, torch.Tensor) else scores)
