"""
GNN-guided heuristic community expansion for overlapping community detection.

This module extends the non-overlapping greedy expansion in
src/run_zu/advanced_util.py to the overlapping setting:
1. GNN membership probabilities prune low-confidence candidates before
   computing the objective.
2. A candidate can be added to multiple communities (no hard assignment).
3. The objective C_o combines structural cohesion (SC) and attribute cohesion (AC).

Author: ANS-OCD project
"""
import numpy as np
import networkx as nx
from scipy import sparse as sp
from typing import Dict, List, Set, Tuple, Optional

from ANS_OCD.config import MULTILABEL_CONFIG


# --- Attribute cohesion (AC) helpers, adapted from src/run_zu/advanced_util.py ---

def _cosine_similarity_vector(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)


def cal_AC(c: List[int], content: np.ndarray) -> float:
    """
    Compute attribute cohesion (AC) of community c as the average cosine similarity
    between all pairs of member nodes.

    Parameters
    ----------
    c : List[int]
        Member node indices.
    content : np.ndarray, shape [N, D]
        Node attribute matrix.

    Returns
    -------
    ac : float
        Attribute cohesion in [0, 1].
    """
    if len(c) <= 1:
        return 1.0
    X_c_sp = sp.csr_matrix(content[np.array(c, dtype=np.int32)])
    norms = np.sqrt(X_c_sp.power(2).sum(axis=1)).A1
    X_c_sp = X_c_sp.multiply(1.0 / (norms[:, None] + 1e-8))
    S = X_c_sp @ X_c_sp.T
    total_sum = S.sum() - len(c)
    ac = 2 * total_sum / (len(c) * (len(c) - 1))
    return float(np.clip(ac, -1.0, 1.0))


def cal_delta_AC_add_node(AC_before: float, c: List[int], n: int, content: np.ndarray) -> float:
    """
    Compute the change in AC when adding node n to community c.

    Parameters
    ----------
    AC_before : float
        Current AC of community c.
    c : List[int]
        Current member nodes.
    n : int
        Candidate node to add.
    content : np.ndarray, shape [N, D]
        Node attribute matrix.

    Returns
    -------
    delta_ac : float
        Change in AC after adding n.
    """
    c_len = len(c)
    if c_len <= 1:
        return 1.0 - AC_before
    D_before = (c_len - 1) * c_len / 2.0
    sum_added = sum(_cosine_similarity_vector(content[u], content[n]) for u in c)
    sum_after = AC_before * D_before + sum_added * 2.0
    D_after = (c_len + 1) * c_len / 2.0
    delta_ac = sum_after / D_after - AC_before
    return float(delta_ac)


# --- Structural cohesion (SC) helpers, adapted from src/run_zu/advanced_util.py ---

def cal_SC_incremental(c_len: int, ein: int, eout: int) -> float:
    """
    Compute structural cohesion (SC) of a community using density, mean degree
    and internal/external edge ratio.

    Parameters
    ----------
    c_len : int
        Number of community members.
    ein : int
        Number of internal edges.
    eout : int
        Number of boundary edges.

    Returns
    -------
    sc : float
        Structural cohesion in [0, 1].
    """
    if c_len <= 1:
        return 1.0
    density = 2.0 * ein / (c_len * (c_len - 1))
    mean_degree = 2.0 * ein / c_len
    M = ein / (eout + 1e-8)
    mean_degree_norm = mean_degree / (mean_degree + 1.0)
    M_norm = M / (M + 1.0)
    sc = (density + mean_degree_norm + M_norm) / 3.0
    return float(np.clip(sc, 0.0, 1.0))


def prepare_SC_incremental_data(c: List[int], G: nx.Graph) -> Tuple[int, int, int]:
    """Precompute c_len, ein, eout for incremental SC updates."""
    c_len = len(c)
    c_set = set(c)
    edges = list(G.edges(c))
    ein = sum(1 for u, v in edges if u in c_set and v in c_set)
    eout = sum(1 for u, v in edges if (u in c_set) != (v in c_set))
    return c_len, ein, eout


def cal_delta_SC_add_node_incremental(c_len: int, ein: int, eout: int,
                                    c_set: Set[int], n: int, G: nx.Graph) -> float:
    """Compute the SC change when adding node n to community c."""
    c_len_new = c_len + 1
    neighbors = set(G.neighbors(n))
    added_internal = len(neighbors & c_set)
    added_external = len(neighbors - c_set)
    ein_new = ein + added_internal
    eout_new = eout - added_internal + added_external
    SC_before = cal_SC_incremental(c_len, ein, eout)
    SC_after = cal_SC_incremental(c_len_new, ein_new, eout_new)
    return SC_after - SC_before


# --- GNN-guided overlapping expansion ---

class CommunityExpander:
    """
    GNN-guided greedy expansion of multiple overlapping communities.

    Parameters
    ----------
    adj_matrix : np.ndarray, shape [N, N]
        Binary/weighted adjacency matrix.
    attr_matrix : np.ndarray, shape [N, D]
        Node attribute matrix.
    gnn_probs : np.ndarray, shape [N, K]
        Sigmoid membership probabilities from the multi-label GNN.
    pruning_threshold : float
        Candidates with membership probability below this value are discarded.
    """

    def __init__(self, adj_matrix: np.ndarray, attr_matrix: np.ndarray,
                 gnn_probs: np.ndarray, pruning_threshold: float = MULTILABEL_CONFIG["pruning_threshold"]):
        self.adj = np.asarray(adj_matrix, dtype=np.float32)
        self.attr = np.asarray(attr_matrix, dtype=np.float32)
        self.probs = np.asarray(gnn_probs, dtype=np.float32)
        self.num_nodes, self.num_communities = self.probs.shape
        self.pruning_threshold = pruning_threshold
        self.G = nx.from_numpy_array(self.adj)
        self.communities: List[Set[int]] = [set() for _ in range(self.num_communities)]

    def initialize(self, seeds: Dict[int, List[int]]):
        """
        Initialize each community with seed nodes.

        Parameters
        ----------
        seeds : Dict[int, List[int]]
            Mapping from community id to seed node list.
        """
        for cid, nodes in seeds.items():
            if 0 <= cid < self.num_communities:
                self.communities[cid] = set(int(n) for n in nodes)

    def _boundary_candidates(self, cid: int) -> Set[int]:
        """Return the set of neighbor nodes that are not yet in community cid."""
        community = self.communities[cid]
        candidates = set()
        for node in community:
            for neighbor in self.G.neighbors(node):
                if neighbor not in community:
                    candidates.add(neighbor)
        return candidates

    def _delta_Co(self, cid: int, node: int) -> float:
        """
        Compute the combined structural + attribute cohesion increment when adding
        node to community cid.
        """
        c = list(self.communities[cid])
        c_len, ein, eout = prepare_SC_incremental_data(c, self.G)
        delta_sc = cal_delta_SC_add_node_incremental(c_len, ein, eout, set(c), node, self.G)
        ac_before = cal_AC(c, self.attr) if len(c) > 1 else 1.0
        delta_ac = cal_delta_AC_add_node(ac_before, c, node, self.attr)
        # Equal weighting of structural and attribute gains. Can be tuned.
        return delta_sc + delta_ac

    def expand_community(self, cid: int, budget: int) -> List[int]:
        """
        Greedily expand a single community within the given budget.

        Parameters
        ----------
        cid : int
            Community id.
        budget : int
            Maximum number of new nodes to add.

        Returns
        -------
        added_nodes : List[int]
            Nodes added in this expansion round.
        """
        added_nodes = []
        for _ in range(budget):
            candidates = self._boundary_candidates(cid)
            if not candidates:
                break
            # GNN-guided pruning: keep only candidates with high membership probability.
            candidates = [n for n in candidates if self.probs[n, cid] >= self.pruning_threshold]
            if not candidates:
                break

            best_node, best_score = None, -np.inf
            for n in candidates:
                score = self._delta_Co(cid, n)
                if score > best_score:
                    best_score = score
                    best_node = n

            # Only accept strictly positive cohesion improvements.
            if best_node is None or best_score <= 0:
                break
            self.communities[cid].add(best_node)
            added_nodes.append(best_node)
        return added_nodes

    def expand_all(self, budgets: Optional[Dict[int, int]] = None,
                   default_budget: int = 50) -> Dict[int, List[int]]:
        """
        Expand all communities. Each community can have its own budget.

        Parameters
        ----------
        budgets : Dict[int, int], optional
            Mapping from community id to expansion budget. Defaults to default_budget.
        default_budget : int
            Default budget per community if budgets is not provided.

        Returns
        -------
        added_by_community : Dict[int, List[int]]
            Mapping from community id to list of added nodes.
        """
        added_by_community = {}
        for cid in range(self.num_communities):
            if not self.communities[cid]:
                continue
            budget = budgets.get(cid, default_budget) if budgets else default_budget
            added = self.expand_community(cid, budget)
            added_by_community[cid] = added
        return added_by_community

    def get_membership_matrix(self, threshold: float = MULTILABEL_CONFIG["overlap_threshold"]) -> np.ndarray:
        """
        Return a binary membership matrix indicating node-community assignments.

        Parameters
        ----------
        threshold : float
            Probability threshold for final hard membership (in addition to the
            expanded community sets).

        Returns
        -------
        membership : np.ndarray, shape [N, K]
            Binary membership matrix.
        """
        membership = np.zeros((self.num_nodes, self.num_communities), dtype=np.float32)
        for cid, nodes in enumerate(self.communities):
            for n in nodes:
                membership[n, cid] = 1.0
        # Also include GNN high-confidence predictions even if not expanded.
        membership = np.maximum(membership, (self.probs >= threshold).astype(np.float32))
        return membership


if __name__ == "__main__":
    # Self-test: synthetic graph with two planted communities sharing a bridge node.
    np.random.seed(42)
    N = 20
    # Community 0: nodes 0-9, Community 1: nodes 10-19, overlap node 9-10 edge.
    adj = np.zeros((N, N))
    for i in range(2):
        start = i * 10
        end = start + 10
        block = np.random.rand(10, 10) < 0.4
        block = (block + block.T) > 0
        adj[start:end, start:end] = block.astype(float)
    adj[9, 10] = adj[10, 9] = 1.0
    np.fill_diagonal(adj, 0.0)

    # Attributes: two group-specific signals.
    attr = np.random.randn(N, 8)
    attr[:10, 0] += 2.0
    attr[10:, 1] += 2.0

    # GNN probabilities: high for in-group, moderate for overlap node.
    probs = np.zeros((N, 2))
    probs[:10, 0] = 0.9
    probs[10:, 1] = 0.9
    probs[9, 1] = 0.3
    probs[10, 0] = 0.3

    expander = CommunityExpander(adj, attr, probs, pruning_threshold=0.05)
    expander.initialize({0: [0, 1], 1: [10, 11]})
    added = expander.expand_all(default_budget=10)
    print("Added nodes per community:", added)
    membership = expander.get_membership_matrix(threshold=0.4)
    for cid in range(2):
        print(f"Community {cid}: {np.where(membership[:, cid] == 1.0)[0].tolist()}")
