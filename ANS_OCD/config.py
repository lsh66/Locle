import torch
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
ANS_OCD_ROOT = Path(__file__).parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RUNZU_DATA_DIR = PROJECT_ROOT / "src" / "run_zu" / "data"

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sentence-BERT model for text feature extraction
SBERT_MODEL = "all-MiniLM-L6-v2"

# GNN hyperparameters
GNNConfig = {
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.5,
    "learning_rate": 0.01,
    "weight_decay": 5e-4,
    "epochs": 200,
    "early_stopping": 20,
}

# Multi-label training settings
MULTILABEL_CONFIG = {
    "prob_threshold": 0.5,      # Probability threshold for hard assignment during training
    "overlap_threshold": 0.4,   # Threshold for overlapping community membership
    "prefer_adaptive_k": True,  # if True, use the label-free adaptive K instead of the ground-truth class count
    "entropy_sigma": 1e-9,     # Smoothing term for label entropy
    "pruning_threshold": 0.05, # GNN pruning threshold for expansion candidates
    "max_negative_gain": -0.05, # Allow slightly negative cohesion gain during expansion (sparse graphs)
    "expansion_budget_ratio": 0.35, # Per-community budget = ceil(N/K * ratio); full coverage now comes from GNN argmax, not brute-force expansion
    "ctl_conflict_high": 0.7,   # High GNN probability but low heuristic score (was 0.9, too strict)
    "ctl_conflict_low": 0.05,   # Low GNN probability but high heuristic score
    "ctl_delta_high": 0.2,      # ΔCo threshold for "heuristic strongly favors" (was 0.5 -> 0 conflicts)
    "ctl_max_resolve": 200,     # Max conflicts sent to the LLM per round (replaces the old [:5] cap)
    "ctl_accept_threshold": 0.3, # For an arbitration that DISAGREES with the GNN top-1, require the
                                 # GNN probability of that community to be at least this before
                                 # accepting it. Prevents low-confidence LLM calls from polluting
                                 # the communities (measured: unconditional acceptance cost ~0.024 jaccard).

    # --- Overlap-aware assignment (opt-in; defaults reproduce legacy behaviour) ---
    # "argmax"  : legacy single-label decision (one community per node).
    # "relative": keep community c when p[n,c] >= overlap_alpha * max_j p[n,j].
    "overlap_mode": "argmax",
    "overlap_alpha": 0.7,
    "overlap_m_max": 3,
    # Soft multi-label seed targets instead of one-hot, from attribute-to-prototype
    # cosine similarity, propagated to 1-hop neighbours.
    "soft_seed": False,
    "soft_seed_topk": 3,
    "soft_seed_neighbor_weight": 0.4,
    "soft_seed_pos_clip": (0.2, 0.6),
    # BCE pos_weight for sparse positive columns; 0 disables it.
    "pos_weight_clip": 0.0,
    # Measurement-only: align pred/GT communities before scoring (no GT truncation).
    # This is NOT part of the method; it only makes metrics valid when K_pred != K_true.
    "eval_hungarian_align": False,
}

# Subspace clustering settings
SUBSPACE_CONFIG = {
    "alpha": 0.1,              # Self-expression regularization
    "max_k": 50,               # Maximum number of initial clusters
    "eigengap_ratio": 0.1,     # Ratio for eigengap heuristic
    # Label-free criterion used to pick K:
    #   "davies_bouldin" -- Davies-Bouldin index on the attribute matrix (recommended;
    #                       selects K=6 on citeseer, matching the true class count)
    #   "eigengap"       -- spectral gap on the self-expression affinity (counts
    #                       topological sub-cliques; over-estimates K, e.g. 40 vs 6)
    "k_method": "davies_bouldin",
    "k_min": 2,                # smallest K considered by the criterion
    "k_max": None,             # None => adaptive ceiling clip(sqrt(N)/2, 10, 60).
                               # A hard-coded 15 truncated the sweep (DBLP/WikiCS
                               # both returned exactly 15, i.e. the argmin sat on
                               # the boundary), which looked like over-estimation.
    "k_max_cap": 60,           # absolute ceiling for the adaptive range
    "k_pca_dim": 128,          # whitened-PCA width used before the DB criterion
    "k_n_seeds": 4,            # KMeans runs averaged per K (DB is seed-unstable otherwise)
}

# DivRank settings
DIVRANK_CONFIG = {
    "lambda_param": 0.95,      # Trade-off between relevance and diversity
    "max_iters": 1000,
    "tol": 1e-6,
    "reps_per_cluster": 5,     # Number of representative nodes per cluster
}

# LLM provider settings (mock by default, set use_mock=False to use real API)
LLM_CONFIG = {
    "use_mock": False,
    "provider": "zhipu",
    "model": "glm-4-flash",
    "temperature": 0.0,
    "max_tokens": 512,
}

ZHIPU_CONFIG = {
    "api_base": "https://open.bigmodel.cn/api/paas/v4/",
}


def get_zhipu_key():
    """Retrieve the Zhipu AI API key from the ``ZHIPU_API_KEY`` environment variable.

    API keys are injected via the environment only; they must never be stored
    in ``config.yaml`` or any other version-controlled file.
    """
    return os.environ.get("ZHIPU_API_KEY", "").strip()


def get_openai_key():
    """Retrieve the OpenAI API key from the ``OPENAI_API_KEY`` environment variable.

    API keys are injected via the environment only; they must never be stored
    in ``config.yaml`` or any other version-controlled file.
    """
    return os.environ.get("OPENAI_API_KEY", "").strip()


if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"ANS-OCD root: {ANS_OCD_ROOT}")
    print(f"Device: {DEVICE}")
    print(f"OpenAI key available: {bool(get_openai_key())}")
