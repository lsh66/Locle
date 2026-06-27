"""
Conflict detection for the upgraded CTL mechanism.

Detects boundary nodes where the GNN prediction and the heuristic expansion rule
disagree significantly. These nodes are sent to an LLM for final multi-label
arbitration.

Author: ANS-OCD project
"""
import numpy as np
from typing import Dict, List, Set, Tuple

from ANS_OCD.config import MULTILABEL_CONFIG
from ANS_OCD.module3_expansion.community_expander import CommunityExpander


def detect_conflicts(expander: CommunityExpander,
                     high_prob: float = MULTILABEL_CONFIG["ctl_conflict_high"],
                     low_prob: float = MULTILABEL_CONFIG["ctl_conflict_low"],
                     delta_high: float = 0.5) -> List[Tuple[int, int, str]]:
    """
    Scan boundary nodes of all communities and detect severe rule-vs-GNN conflicts.

    Two conflict types are captured:
    1. "rule_favor_gnn_against": heuristic strongly favors adding the node (ΔCo is
       large and positive) but the GNN probability for this community is very low.
    2. "gnn_favor_rule_against": GNN strongly favors membership (probability > 0.9)
       but the heuristic ΔCo is negative (the node would hurt community cohesion).

    Parameters
    ----------
    expander : CommunityExpander
        The expander holding current communities and GNN probabilities.
    high_prob : float
        High-probability threshold for GNN-favor conflict.
    low_prob : float
        Low-probability threshold for rule-favor conflict.
    delta_high : float
        ΔCo threshold for "strong heuristic favor".

    Returns
    -------
    conflicts : List[Tuple[int, int, str]]
        List of (node_id, community_id, conflict_type).
    """
    conflicts = []
    for cid in range(expander.num_communities):
        if not expander.communities[cid]:
            continue
        candidates = expander._boundary_candidates(cid)
        for node in candidates:
            delta_co = expander._delta_Co(cid, node)
            prob = float(expander.probs[node, cid])
            if delta_co > delta_high and prob < low_prob:
                conflicts.append((int(node), cid, "rule_favor_gnn_against"))
            elif delta_co < 0 and prob > high_prob:
                conflicts.append((int(node), cid, "gnn_favor_rule_against"))
    return conflicts


def deduplicate_conflicts(conflicts: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """
    Keep only one conflict record per (node, community) pair. If both conflict
    types exist for the same pair, the one detected first is retained.
    """
    seen = set()
    unique = []
    for node, cid, ctype in conflicts:
        key = (node, cid)
        if key not in seen:
            seen.add(key)
            unique.append((node, cid, ctype))
    return unique


if __name__ == "__main__":
    # Self-test: reuse the synthetic graph from community_expander.py.
    from ANS_OCD.module3_expansion.community_expander import CommunityExpander

    np.random.seed(0)
    N = 20
    adj = np.zeros((N, N))
    for i in range(2):
        start = i * 10
        end = start + 10
        block = np.random.rand(10, 10) < 0.4
        block = (block + block.T) > 0
        adj[start:end, start:end] = block.astype(float)
    adj[9, 10] = adj[10, 9] = 1.0
    np.fill_diagonal(adj, 0.0)

    attr = np.random.randn(N, 8)
    attr[:10, 0] += 2.0
    attr[10:, 1] += 2.0

    # Artificial conflict: node 9 has high GNN prob for community 1 but negative ΔCo.
    probs = np.zeros((N, 2))
    probs[:10, 0] = 0.9
    probs[10:, 1] = 0.9
    probs[9, 1] = 0.95  # GNN strongly favors community 1
    probs[10, 0] = 0.05

    expander = CommunityExpander(adj, attr, probs, pruning_threshold=0.05)
    expander.initialize({0: [0, 1], 1: [10, 11]})

    conflicts = detect_conflicts(expander, high_prob=0.9, low_prob=0.05, delta_high=0.5)
    conflicts = deduplicate_conflicts(conflicts)
    print(f"Detected {len(conflicts)} conflicts:")
    for node, cid, ctype in conflicts:
        print(f"  Node {node} vs Community {cid}: {ctype}")
