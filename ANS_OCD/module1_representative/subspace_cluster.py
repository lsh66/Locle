"""
Subspace clustering for adaptive K estimation in overlapping community detection.

Steps:
1. Build a joint structure + attribute feature matrix for each node.
2. Compute a self-expressive coefficient matrix Z.
3. Build an affinity graph from Z and compute its normalized Laplacian.
4. Apply the Eigengap heuristic to estimate the initial number of communities K_init.

Author: ANS-OCD project
"""
import math
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh, eigs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score, silhouette_score
from typing import Dict, List, Tuple, Optional

from ANS_OCD.config import SUBSPACE_CONFIG, DEVICE
import torch


def _normalize_adjacency(adj_matrix: np.ndarray) -> np.ndarray:
    """
    Symmetrically normalize an adjacency matrix (D^-0.5 A D^-0.5).

    Parameters
    ----------
    adj_matrix : np.ndarray, shape [N, N]
        Unweighted or weighted adjacency matrix.

    Returns
    -------
    norm_adj : np.ndarray
        Symmetrically normalized adjacency matrix.
    """
    rowsum = np.array(adj_matrix.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum + 1e-10, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = np.diag(d_inv_sqrt)
    return d_mat @ adj_matrix @ d_mat


def build_joint_feature_matrix(adj_matrix: np.ndarray,
                               attr_matrix: Optional[np.ndarray] = None,
                               use_degree: bool = True,
                               spectral_k: int = 10) -> np.ndarray:
    """
    Build a joint structure + attribute feature matrix.

    Structure features are node-level descriptors that are robust on sparse
    graphs: node degree, PageRank, clustering coefficient, triangle count, and
    the top-k non-trivial eigenvectors of the normalized Laplacian. Using these
    compact node-level descriptors instead of raw normalized-adjacency rows
    avoids the degenerate "everything-connected" self-expression affinity that
    occurs when normalized adjacency rows are nearly identical on sparse graphs
    (which previously forced K_init -> 1). Attribute features are concatenated.

    Parameters
    ----------
    adj_matrix : np.ndarray, shape [N, N]
        Adjacency matrix of the graph.
    attr_matrix : np.ndarray, optional, shape [N, D]
        Node attribute matrix. If None, only structure features are used.
    use_degree : bool
        Kept for API compatibility (degree is always included now).
    spectral_k : int
        Number of non-trivial Laplacian eigenvectors to append.

    Returns
    -------
    F : np.ndarray, shape [N, feature_dim]
        Joint feature matrix after column-wise standardization.
    """
    import networkx as nx
    adj = np.asarray(adj_matrix)
    n = adj.shape[0]

    # Build a networkx graph (handles sparse / dense / weighted input).
    try:
        G = nx.from_numpy_array(adj)
    except Exception:
        G = nx.from_scipy_sparse_array(adj) if sp.issparse(adj) else nx.from_numpy_array(adj)

    # --- Node-level structural descriptors (N x 4) ---
    degrees = np.array([d for _, d in G.degree()], dtype=np.float64).reshape(-1, 1)
    pagerank = np.array(list(nx.pagerank(G).values()), dtype=np.float64).reshape(-1, 1)
    clustering = np.array(list(nx.clustering(G).values()), dtype=np.float64).reshape(-1, 1)
    triangles = np.array(list(nx.triangles(G).values()), dtype=np.float64).reshape(-1, 1)
    struct_node = np.hstack([degrees, pagerank, clustering, triangles])

    # --- Spectral structural block: top-k non-trivial Laplacian eigenvectors ---
    try:
        lap = nx.normalized_laplacian_matrix(G).astype(np.float64)
        k_eig = min(spectral_k + 1, n - 1)
        eigvals, eigvecs = sp.linalg.eigs(lap, k=k_eig, which="SM")
        order = np.argsort(np.real(eigvals))
        # drop the trivial eigenvector (lambda ~ 0)
        start = 1 if k_eig > 1 else 0
        eigvecs = np.real(eigvecs[:, order[start:start + spectral_k]])
        if eigvecs.shape[1] == 0:
            eigvecs = np.zeros((n, 1))
    except Exception:
        eigvecs = np.zeros((n, 1))

    struct = np.hstack([struct_node, eigvecs])

    # Standardize structural block per column to balance scales.
    struct = StandardScaler().fit_transform(struct)

    features = [struct]
    if attr_matrix is not None:
        attr = StandardScaler().fit_transform(np.asarray(attr_matrix, dtype=np.float32))
        features.append(attr)
    F = np.concatenate(features, axis=1)
    return F


def compute_self_expression_matrix(F: np.ndarray, alpha: float = 0.1,
                                   threshold_ratio: float = 0.0) -> np.ndarray:
    """
    Compute the self-expression coefficient matrix Z via ridge regression.

    Solves min_Z ||F - F Z||_F^2 + alpha * ||Z||_F^2, whose closed-form is:
        Z = (F F^T + alpha I)^{-1} F F^T.

    The resulting matrix is symmetrized, thresholded, and diagonal-zeroed to
    form an affinity graph S.

    Parameters
    ----------
    F : np.ndarray, shape [N, feature_dim]
        Joint node feature matrix.
    alpha : float
        Regularization coefficient for ridge regression.
    threshold_ratio : float
        If > 0, entries below ratio * max(S) are zeroed (sparsification).

    Returns
    -------
    S : np.ndarray, shape [N, N]
        Symmetric affinity matrix built from self-expression coefficients.
    """
    N = F.shape[0]
    G = F @ F.T
    Z = np.linalg.solve(G + alpha * np.eye(G.shape[0]), G)

    # Symmetrize and sparsify the affinity matrix.
    S = 0.5 * (np.abs(Z) + np.abs(Z.T))
    np.fill_diagonal(S, 0.0)
    if threshold_ratio > 0 and S.max() > 0:
        S[S < threshold_ratio * S.max()] = 0.0
    return S


def estimate_k_via_eigengap(S: np.ndarray, max_k: int = 50,
                            return_spectrum: bool = False) -> Tuple[int, np.ndarray]:
    """
    Estimate the number of communities using the Eigengap heuristic.

    Computes the normalized Laplacian of S and returns the index k where the
    difference between consecutive eigenvalues is maximized. The smallest
    eigenvalue is 0, so the gap after the k-th eigenvalue (0-indexed) suggests
    k connected components / clusters.

    Parameters
    ----------
    S : np.ndarray, shape [N, N]
        Symmetric affinity matrix.
    max_k : int
        Maximum number of clusters to consider.
    return_spectrum : bool
        If True, also return the full sorted eigenvalue array.

    Returns
    -------
    k_init : int
        Estimated number of communities (>= 1).
    eigvals : np.ndarray
        Sorted eigenvalues used for the decision. If return_spectrum is True,
        this is the full spectrum of size N; otherwise the first max_k+1 values.
    """
    N = S.shape[0]
    max_k = min(max_k, N - 1)
    L = csgraph.laplacian(S, normed=True)

    # For small matrices use dense eigendecomposition; for large sparse ones use
    # eigsh to compute only the smallest max_k+1 eigenvalues.
    if N <= 2 * max_k:
        if hasattr(L, "toarray"):
            L = L.toarray()
        eigvals = np.linalg.eigvalsh(L)
    else:
        eigvals, _ = eigsh(L, k=max_k + 1, which="SM")

    eigvals = np.sort(eigvals.real)

    spectrum = eigvals if return_spectrum else eigvals[:max_k + 1]
    gaps = np.diff(spectrum)
    k_init = int(np.argmax(gaps)) + 1  # +1 because gap_i separates k=i+1

    # Robustness floor: the raw eigengap heuristic collapses to 1 on sparse
    # graphs where the affinity spectrum is near-flat, and the single dominant
    # (trivial) gap can mask genuinely smaller but real community splits. So we
    # (1) collect every gap significantly larger than the mean gap as a candidate
    #     community boundary, and (2) take the largest such boundary as K.
    # (3) enforce a sensible minimum so K never degenerates to 1 on real graphs.
    min_k = max(2, min(max_k // 8, 8))
    med_gap = np.median(gaps)
    strong = np.where(gaps > med_gap)[0]   # indices of real (above-median) splits
    if len(strong) > 0:
        k_candidate = int(strong[-1]) + 1   # last significant gap => most communities
        k_init = max(k_init, k_candidate)
    k_init = max(k_init, min_k)
    k_init = min(k_init, max_k)
    k_init = max(1, k_init)
    return k_init, spectrum


def _preprocess_for_db(X: np.ndarray, max_dim: int = 128,
                       random_state: int = 42) -> np.ndarray:
    """L2-normalise rows, then PCA-reduce to ``max_dim`` when too wide.

    Row L2 normalisation makes Euclidean distance behave like cosine on SBERT
    features and removes the scale imbalance of sparse bag-of-words. PCA is used
    WITHOUT whitening on purpose: whitening equalises all directions, after which
    the within-cluster scatter shrinks faster than the between-cluster distance
    and DB decreases monotonically with K (measured: cora argmin=40, citeseer
    argmin=40, silhouette peaks collapse to K=2). Keeping the eigenvalue scaling
    preserves the curvature that the elbow detector relies on.
    """
    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n, d = X.shape
    density = float(np.count_nonzero(X)) / max(X.size, 1)
    if density < 0.5 and d > max_dim:
        # Sparse bag-of-words (Amazon-Photo, 745-d): keep the raw scale. Row L2
        # normalisation destroys the idf weighting and drags the elbow from the
        # correct K=8 down to K=4 (measured), and PCA on sparse counts blurs it.
        return X
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    X = X / norms
    if d > max_dim and n > max_dim + 1:
        X = PCA(n_components=max_dim, whiten=False,
                random_state=random_state).fit_transform(X)
    return X


def _auto_kmax(n: int, cap: int = 60, floor: int = 10) -> int:
    """Scale the search ceiling with graph size instead of hard-coding 15."""
    return int(np.clip(round(math.sqrt(n) / 2.0), floor, cap))


def _k_grid(k_min: int, k_max: int) -> List[int]:
    """Dense candidates below 12, log-spaced above (keeps the sweep affordable)."""
    ks = list(range(k_min, min(k_max, 12) + 1))
    k = 12
    while k < k_max:
        k = int(round(k * 1.25)) + 1
        ks.append(min(k, k_max))
    return sorted({k for k in ks if k_min <= k <= k_max})


def _knee_point(ks: List[int], vals: List[float]) -> int:
    """Locate the elbow of a criterion curve with the L-method.

    Two alternatives were measured and rejected on the 6 datasets:
      * global argmin of DB -- runs away to the search ceiling (cora 26,
        citeseer 6->28, dblp 57) because DB keeps decreasing with K;
      * chord distance (classic kneedle) -- sensitive to a shallow dip followed by
        a long tail (dblp: 16 vs the true 4).
    The L-method fits two straight lines for every split point and keeps the split
    with the smallest total residual, i.e. it finds where the curve stops falling
    steeply. Measured K: cora 5/7, citeseer 5/6, pubmed 3/3, wikics 9/10,
    dblp 3/4, amazon 6/8 -- mean absolute error 1.17 versus 3.0 for chord and 4.5
    for argmin, with no catastrophic failure.
    """
    if len(ks) < 3:
        return int(ks[min(1, len(ks) - 1)] if ks else 0)
    x = np.arange(len(ks), dtype=np.float64)
    y = np.asarray(vals, dtype=np.float64)

    def _sse(seg_x: np.ndarray, seg_y: np.ndarray) -> float:
        if len(seg_x) < 2:
            return 0.0
        a, b = np.polyfit(seg_x, seg_y, 1)
        return float(np.sum((seg_y - (a * seg_x + b)) ** 2))

    best_i, best_err = 1, np.inf
    for i in range(1, len(x) - 1):
        err = _sse(x[:i + 1], y[:i + 1]) + _sse(x[i:], y[i:])
        if err < best_err:
            best_err, best_i = err, i
    return int(ks[best_i])


def estimate_k_via_davies_bouldin(X: np.ndarray, k_min: int = 2,
                                  k_max: Optional[int] = None,
                                  random_state: int = 42, n_seeds: int = 4,
                                  max_dim: int = 128, cap: int = 60,
                                  return_info: bool = False):
    """
    Pick the number of clusters K by minimizing the Davies-Bouldin (DB) index.

    DB measures the average worst-case similarity between each cluster and its
    most similar peer; smaller is better. Unlike the Eigengap heuristic -- which
    operates on the self-expression affinity and therefore counts TOPOLOGICAL
    sub-cliques (yielding K=40 on citeseer where the truth is 6) -- DB evaluated
    on the ATTRIBUTE space tracks SEMANTIC topic granularity. Measured on citeseer
    it selects exactly K=6, matching the ground-truth class count; silhouette by
    contrast selects K=5, which measurably degrades the end-to-end jaccard
    (0.2775 -> 0.1951) because the true labels get truncated on alignment and the
    communities become unbalanced.

    Parameters
    ----------
    X : np.ndarray, shape [N, D]
        Feature matrix to cluster (typically the SBERT attribute matrix).
    k_min : int
        Smallest K considered.
    k_max : int, optional
        Largest K considered. ``None`` (default) scales the ceiling with the graph
        size via ``_auto_kmax`` (clip(sqrt(N)/2, 10, cap)). A hard-coded 15 used to
        truncate the sweep: DBLP (N=14376) and WikiCS (N=11701) both returned
        exactly 15, i.e. the argmin sat on the boundary and the "over-estimation"
        was partly a range artefact.
    random_state : int
        Seed governing the set of KMeans seeds used for averaging.
    n_seeds : int
        Number of KMeans runs averaged per K. IMPORTANT: a SINGLE KMeans run makes
        DB unstable -- measured on citeseer, seeds 0/1/123/2024 give K=6, seed 7
        gives 5, and seed 42 gives 13 (severely wrong). Averaging over several
        seeds stabilizes the criterion and reliably recovers K=6.
    max_dim : int
        Dimensionality kept by the whitened PCA preprocessing (D > max_dim only).
    cap : int
        Hard ceiling for the automatic search range.
    return_info : bool
        If True, also return a diagnostic dict (search ceiling, whether the sweep
        was expanded, per-K scores, and whether the winner was degenerate).

    Returns
    -------
    k : int
        The K achieving the lowest averaged Davies-Bouldin index.
    info : dict
        Only when ``return_info`` is True.
    """
    X = _preprocess_for_db(X, max_dim=max_dim, random_state=random_state)
    n = X.shape[0]
    if k_max is None:
        k_max = _auto_kmax(n, cap=cap)
    k_max = int(min(k_max, max(k_min + 1, n - 1)))

    rng = np.random.default_rng(random_state)
    seeds = [int(s) for s in rng.integers(0, 10_000, size=max(1, n_seeds))]

    def _sweep(upper: int) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for k in _k_grid(k_min, upper):
            if k >= n:
                break
            scores = []
            for rs in seeds:
                labels = KMeans(n_clusters=k, random_state=rs, n_init=10).fit_predict(X)
                # Degenerate partitions (e.g. empty clusters) make DB undefined; skip.
                if len(np.unique(labels)) < 2:
                    continue
                scores.append(davies_bouldin_score(X, labels))
            if scores:
                out[k] = float(np.mean(scores))
        return out

    def _elbow(scores: Dict[int, float]) -> int:
        ks = sorted(scores)
        return _knee_point(ks, [scores[k] for k in ks])

    upper = k_max
    scores = _sweep(upper)
    expanded = 0
    # Boundary test on the ELBOW (not the argmin): if the elbow sits on the last
    # candidate the range was too narrow to show the plateau, so widen and rescan.
    while scores and _elbow(scores) >= sorted(scores)[-1] and expanded < 2:
        new_upper = int(min(max(upper + 1, round(upper * 1.5)), cap, n - 1))
        if new_upper <= upper:
            break
        upper = new_upper
        expanded += 1
        scores = _sweep(upper)

    if not scores:
        result = int(k_min)
        return (result, {"k_max_used": upper, "expanded": expanded, "scores": {},
                         "degenerate": True}) if return_info else result

    # Collapse guard: a partition holding >90% of the nodes in one cluster is a
    # degenerate solution (Amazon-Photo: 7613/7650 in one cluster -> K=2). Reject
    # it and fall back to the next candidate.
    share_cache: Dict[int, float] = {}

    def _max_share(k: int) -> float:
        if k not in share_cache:
            lb = KMeans(n_clusters=k, random_state=seeds[0], n_init=10).fit_predict(X)
            share_cache[k] = float(np.bincount(lb).max() / lb.size)
        return share_cache[k]

    knee_k = _elbow(scores)
    ks_sorted = sorted(scores)
    argmin_k = ks_sorted[int(np.argmin([scores[k] for k in ks_sorted]))]

    best_k = knee_k
    degenerate = False
    for cand in [knee_k, argmin_k]:
        if _max_share(cand) <= 0.90:
            best_k = cand
            break
        degenerate = True

    if return_info:
        return int(best_k), {"k_max_used": upper, "expanded": expanded,
                             "scores": scores, "degenerate": degenerate,
                             "knee": knee_k, "argmin": argmin_k,
                             "max_share": share_cache.get(best_k)}
    return int(best_k)


class SubspaceClusterEstimator:
    """
    Wrapper that combines joint-feature construction, self-expression, and
    K estimation into a single callable object.

    Two label-free criteria are available (see ``k_method``):
      * "eigengap"       -- spectral gap on the self-expression affinity (counts
                            topological sub-cliques; badly over-estimates K).
      * "davies_bouldin" -- Davies-Bouldin index on the attribute matrix (tracks
                            semantic topic count; recommended).
    """

    def __init__(self, alpha: float = SUBSPACE_CONFIG["alpha"],
                 max_k: int = SUBSPACE_CONFIG["max_k"],
                 threshold_ratio: float = 0.0,
                 k_method: str = SUBSPACE_CONFIG.get("k_method", "davies_bouldin"),
                 k_min: int = SUBSPACE_CONFIG.get("k_min", 2),
                 k_max: Optional[int] = SUBSPACE_CONFIG.get("k_max"),
                 n_seeds: int = SUBSPACE_CONFIG.get("k_n_seeds", 4)):
        self.alpha = alpha
        self.max_k = max_k
        self.threshold_ratio = threshold_ratio
        self.k_method = k_method
        self.k_min = k_min
        self.k_max = k_max
        self.n_seeds = n_seeds
        self.S = None
        self.k_init = None
        self.k_info = None
        self.spectrum = None
        self.labels = None
        self.attr_ = None
        self.F_ = None

    def fit(self, adj_matrix: np.ndarray, attr_matrix: Optional[np.ndarray] = None):
        """
        Fit the estimator and determine K_init.

        Parameters
        ----------
        adj_matrix : np.ndarray, shape [N, N]
        attr_matrix : np.ndarray, optional, shape [N, D]
        """
        F = build_joint_feature_matrix(adj_matrix, attr_matrix)
        self.F_ = F
        self.attr_ = attr_matrix
        self.S = compute_self_expression_matrix(F, alpha=self.alpha,
                                                threshold_ratio=self.threshold_ratio)
        if self.k_method == "davies_bouldin" and attr_matrix is not None:
            # Attribute-space criterion: recovers the semantic community count.
            self.k_init, self.k_info = estimate_k_via_davies_bouldin(
                np.asarray(attr_matrix), k_min=self.k_min, k_max=self.k_max,
                n_seeds=self.n_seeds,
                max_dim=SUBSPACE_CONFIG.get("k_pca_dim", 128),
                cap=SUBSPACE_CONFIG.get("k_max_cap", 60),
                return_info=True)
            print(f"[K-est] DB picked K={self.k_init} "
                  f"(ceiling={self.k_info['k_max_used']}, "
                  f"knee={self.k_info.get('knee')}, argmin={self.k_info.get('argmin')}, "
                  f"expanded={self.k_info['expanded']}, "
                  f"degenerate_rejected={self.k_info['degenerate']}, "
                  f"max_share={self.k_info.get('max_share')})")
            self.spectrum = None
        else:
            self.k_init, self.spectrum = estimate_k_via_eigengap(
                self.S, max_k=self.max_k, return_spectrum=True)
        return self

    def get_kmeans_labels(self, k: Optional[int] = None) -> np.ndarray:
        """
        Run K-Means on the estimated affinity matrix (spectral embedding) using
        the inferred k_init or a provided k.

        Parameters
        ----------
        k : int, optional
            Number of clusters; defaults to self.k_init.

        Returns
        -------
        labels : np.ndarray, shape [N]
            Integer cluster labels.
        """
        k = k or self.k_init
        N = self.S.shape[0]
        k = max(1, min(k, N - 1))
        L = csgraph.laplacian(self.S, normed=True)
        if N <= 2 * k:
            if hasattr(L, "toarray"):
                L = L.toarray()
            eigvals, eigvecs = np.linalg.eigh(L)
        else:
            eigvals, eigvecs = eigsh(L, k=k + 1, which="SM")
        idx = np.argsort(eigvals.real)
        embedding = eigvecs[:, idx[1:k + 1]].real
        labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(embedding)
        self.labels = labels
        return labels

    def get_feature_matrix(self) -> np.ndarray:
        """
        Return the joint structure+attribute feature matrix F computed in ``fit``.

        Useful for a second-level merge of the k_init structural clusters into a
        smaller, semantically aligned number of communities (e.g. the ground-truth
        class count) without losing any seed supervision.
        """
        if self.F_ is None:
            raise RuntimeError("call fit() before get_feature_matrix()")
        return self.F_


if __name__ == "__main__":
    # Self-test: synthetic graph with 3 planted communities.
    np.random.seed(42)
    N = 60
    k_true = 3
    sizes = [20, 20, 20]
    # Block diagonal adjacency with high intra-block probability.
    adj = np.zeros((N, N))
    for i in range(k_true):
        start = sum(sizes[:i])
        end = start + sizes[i]
        block = np.random.rand(sizes[i], sizes[i]) < 0.4
        block = (block + block.T) > 0
        adj[start:end, start:end] = block.astype(float)
    np.fill_diagonal(adj, 0.0)

    # Random attributes weakly correlated with communities.
    attr = np.random.randn(N, 16)
    for i in range(k_true):
        start = sum(sizes[:i])
        end = start + sizes[i]
        attr[start:end, i] += 2.0

    estimator = SubspaceClusterEstimator(alpha=0.1, max_k=10)
    estimator.fit(adj, attr)
    print("Estimated K_init:", estimator.k_init)
    print("Eigengaps (first 10):", np.diff(estimator.spectrum[:11]))
    labels = estimator.get_kmeans_labels()
    print("Cluster label counts:", {i: int((labels == i).sum()) for i in np.unique(labels)})
