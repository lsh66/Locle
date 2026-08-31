"""
Subspace clustering for adaptive K estimation in overlapping community detection.

Steps:
1. Build a joint structure + attribute feature matrix for each node.
2. Compute a self-expressive coefficient matrix Z.
3. Build an affinity graph from Z and compute its normalized Laplacian.
4. Apply the Eigengap heuristic to estimate the initial number of communities K_init.

Author: ANS-OCD project
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh, eigs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

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


class SubspaceClusterEstimator:
    """
    Wrapper that combines joint-feature construction, self-expression, and
    Eigengap-based K estimation into a single callable object.
    """

    def __init__(self, alpha: float = SUBSPACE_CONFIG["alpha"],
                 max_k: int = SUBSPACE_CONFIG["max_k"],
                 threshold_ratio: float = 0.0):
        self.alpha = alpha
        self.max_k = max_k
        self.threshold_ratio = threshold_ratio
        self.S = None
        self.k_init = None
        self.spectrum = None
        self.labels = None

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
        self.S = compute_self_expression_matrix(F, alpha=self.alpha,
                                                threshold_ratio=self.threshold_ratio)
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
