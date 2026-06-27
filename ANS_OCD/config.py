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
    "entropy_sigma": 1e-9,     # Smoothing term for label entropy
    "pruning_threshold": 0.05, # GNN pruning threshold for expansion candidates
    "ctl_conflict_high": 0.9,   # High GNN probability but low heuristic score
    "ctl_conflict_low": 0.05,   # Low GNN probability but high heuristic score
}

# Subspace clustering settings
SUBSPACE_CONFIG = {
    "alpha": 0.1,              # Self-expression regularization
    "max_k": 50,               # Maximum number of initial clusters
    "eigengap_ratio": 0.1,     # Ratio for eigengap heuristic
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
    """Retrieve Zhipu AI API key from environment or project config.yaml."""
    key = os.environ.get("ZHIPU_API_KEY")
    if key:
        return key
    try:
        import yaml
        config_path = PROJECT_ROOT / "config.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            return cfg.get("ZHIPU_KEY", "")
    except Exception:
        pass
    return ""


def get_openai_key():
    """Retrieve OpenAI API key from environment or project config.yaml."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    try:
        import yaml
        config_path = PROJECT_ROOT / "config.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            return cfg.get("OPENAI_KEY", "")
    except Exception:
        pass
    return ""


if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"ANS-OCD root: {ANS_OCD_ROOT}")
    print(f"Device: {DEVICE}")
    print(f"OpenAI key available: {bool(get_openai_key())}")
