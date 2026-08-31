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
    # S.sum() includes the diagonal (len(c) self-similarities of 1). Subtracting it
    # leaves the sum over ORDERED pairs (i!=j), i.e. 2 * sum_{i<j} cos.
    # AC is the mean over UNORDERED pairs: sum_{i<j} cos / (|c|(|c|-1)/2).
    # NOTE: the previous formula multiplied by an extra factor of 2, which
    # double-counted and then got clipped to 1.0 -- making AC a constant 1.0 with
    # no discriminative power for guiding expansion.
    total_sum = S.sum() - len(c)
    ac = total_sum / (len(c) * (len(c) - 1))
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

        # --- Incremental bookkeeping for fast Delta C_o evaluation ---
        # Without these caches, each Delta C_o call rebuilt the whole community's
        # structural counts and its |c| x |c| attribute-similarity matrix, giving
        # O(budget * candidates * |c|^2 * D) total cost -- which made expansion
        # hang for hours on graphs like citeseer (3186 nodes).
        # With unit-norm features, AC can be maintained from the running feature
        # SUM alone:  sum_{i<j} cos(x_i,x_j) = (|sum x|^2 - |c|) / 2.
        norms = np.linalg.norm(self.attr, axis=1, keepdims=True)
        self.attr_norm = (self.attr / (norms + 1e-12)).astype(np.float32)
        # Per community: {"size", "ein", "eout", "sum_feat"}
        self._stats: List[Dict] = [self._empty_stats() for _ in range(self.num_communities)]
        self._boundary: List[Set[int]] = [set() for _ in range(self.num_communities)]

    def _empty_stats(self) -> Dict:
        """Return a zeroed incremental-stats dict for one community."""
        return {
            "size": 0,
            "ein": 0,
            "eout": 0,
            "sum_feat": np.zeros(self.attr.shape[1], dtype=np.float64),
        }

    def _build_stats(self, cid: int) -> Dict:
        """Compute incremental stats for community cid from scratch (O(|c|*deg))."""
        members = list(self.communities[cid])
        st = self._empty_stats()
        if members:
            _, ein, eout = prepare_SC_incremental_data(members, self.G)
            st["size"] = len(members)
            st["ein"] = int(ein)
            st["eout"] = int(eout)
            st["sum_feat"] = self.attr_norm[members].sum(axis=0).astype(np.float64)
        return st

    def _build_boundary(self, cid: int) -> Set[int]:
        """Compute the boundary (neighbors outside the community) from scratch."""
        community = self.communities[cid]
        b = set()
        for node in community:
            for nb in self.G.neighbors(node):
                if nb not in community:
                    b.add(nb)
        return b

    def _rebuild_caches(self):
        """Rebuild stats and boundary caches for all communities."""
        for cid in range(self.num_communities):
            self._stats[cid] = self._build_stats(cid)
            self._boundary[cid] = self._build_boundary(cid)

    def _ac_from_stats(self, st: Dict) -> float:
        """Attribute cohesion from cached stats: AC = (|sum x|^2 - |c|)/(|c|(|c|-1))."""
        size = st["size"]
        if size <= 1:
            return 1.0
        s = st["sum_feat"]
        pair_sum = (float(s @ s) - size) / 2.0
        return float(np.clip(2.0 * pair_sum / (size * (size - 1)), -1.0, 1.0))

    def _delta_Co_fast(self, cid: int, node: int, st: Dict) -> float:
        """Delta C_o in O(D + deg(node)) using cached stats (vs O(|c|^2 * D) before)."""
        c_set = self.communities[cid]
        neighbors = set(self.G.neighbors(node))
        added_internal = len(neighbors & c_set)
        added_external = len(neighbors - c_set)
        size_new = st["size"] + 1
        ein_new = st["ein"] + added_internal
        eout_new = st["eout"] - added_internal + added_external
        delta_sc = (cal_SC_incremental(size_new, ein_new, eout_new)
                    - cal_SC_incremental(st["size"], st["ein"], st["eout"]))
        ac_before = self._ac_from_stats(st)
        sum_after = st["sum_feat"] + self.attr_norm[node]
        ac_after = self._ac_from_stats({"size": size_new, "sum_feat": sum_after})
        return delta_sc + (ac_after - ac_before)

    def _commit_node(self, cid: int, node: int):
        """Add node to community cid and update all incremental caches."""
        self.communities[cid].add(node)
        st = self._stats[cid]
        neighbors = set(self.G.neighbors(node))
        members = self.communities[cid]
        added_internal = len(neighbors & (members - {node}))
        added_external = len(neighbors - members)
        st["ein"] += added_internal
        st["eout"] += added_external - added_internal
        st["size"] += 1
        st["sum_feat"] = st["sum_feat"] + self.attr_norm[node]
        # Boundary update: node is no longer a candidate; its outside neighbors are.
        self._boundary[cid].discard(node)
        for nb in neighbors:
            if nb not in members:
                self._boundary[cid].add(nb)

    def initialize(self, seeds: Dict[int, List[int]]):
        """
        Initialize each community with seed nodes and rebuild incremental caches.

        Parameters
        ----------
        seeds : Dict[int, List[int]]
            Mapping from community id to seed node list.
        """
        for cid, nodes in seeds.items():
            if 0 <= cid < self.num_communities:
                self.communities[cid] = set(int(n) for n in nodes)
        self._rebuild_caches()

    def _boundary_candidates(self, cid: int) -> Set[int]:
        """Return cached boundary (neighbors not yet in community cid)."""
        return self._boundary[cid]

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

    def expand_community(self, cid: int, budget: int, prune: bool = True,
                          max_negative_gain: float = 0.0,
                          max_candidates: int = 200) -> List[int]:
        """
        Greedily expand a single community within the given budget.

        Parameters
        ----------
        cid : int
            Community id.
        budget : int
            Maximum number of new nodes to add.
        prune : bool, default True
            If True, drop candidate nodes whose GNN membership probability is
            below ``self.pruning_threshold`` before scoring. If False (used in
            early self-training rounds when the GNN is still weak), all boundary
            neighbors are kept so the community can grow via the heuristic
            cohesion objective alone, preventing the expansion from stalling.
        max_negative_gain : float, default 0.0
            Allow a candidate with slightly negative cohesion improvement to be
            accepted (top-1 among candidates), as long as its score is above
            this floor. On sparse graphs (low avg degree) most boundary nodes
            yield Delta C_o <= 0, so the old "strictly positive" rule stopped
            expansion after only the seed nodes. Permitting a small negative
            gain lets the community spread across structural boundaries instead
            of stalling.

        Returns
        -------
        added_nodes : List[int]
            Nodes added in this expansion round.
        """
        added_nodes = []
        st = self._stats[cid]
        for _ in range(budget):
            candidates = self._boundary_candidates(cid)
            if not candidates:
                break
            # GNN-guided pruning: keep only candidates with high membership
            # probability. Skipped when prune=False (early rounds).
            if prune:
                candidates = [n for n in candidates if self.probs[n, cid] >= self.pruning_threshold]
                if not candidates:
                    break

            # Cap the number of scored candidates so a huge boundary cannot make a
            # single expansion step quadratic in the graph size.
            if max_candidates and len(candidates) > max_candidates:
                candidates = sorted(
                    candidates, key=lambda n: float(self.probs[n, cid]), reverse=True
                )[:max_candidates]

            best_node, best_score = None, -np.inf
            for n in candidates:
                score = self._delta_Co_fast(cid, n, st)
                if score > best_score:
                    best_score = score
                    best_node = n

            # Accept the best candidate unless its cohesion improvement is worse
            # than the allowed negative floor. This lets communities grow even
            # when the top candidate has a slightly negative Delta C_o.
            if best_node is None or best_score <= max_negative_gain:
                break
            self._commit_node(cid, best_node)
            added_nodes.append(best_node)
        return added_nodes

    def expand_all(self, budgets: Optional[Dict[int, int]] = None,
                   default_budget: int = 50, prune: bool = True,
                   max_negative_gain: float = 0.0,
                   max_candidates: int = 200) -> Dict[int, List[int]]:
        """
        Expand all communities. Each community can have its own budget.

        Parameters
        ----------
        budgets : Dict[int, int], optional
            Mapping from community id to expansion budget. Defaults to default_budget.
        default_budget : int
            Default budget per community if budgets is not provided.
        prune : bool, default True
            Whether to apply GNN probability pruning (see ``expand_community``).
        max_negative_gain : float, default 0.0
            Negative cohesion floor passed to ``expand_community``.
        max_candidates : int, default 200
            Max candidates scored per expansion step (performance guard).

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
            added = self.expand_community(cid, budget, prune=prune,
                                          max_negative_gain=max_negative_gain,
                                          max_candidates=max_candidates)
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
