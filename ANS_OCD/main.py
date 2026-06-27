"""
ANS-OCD main pipeline.

This script orchestrates the six stages of the overlapping community detection
algorithm:
  1. Data loading + multi-label GNN initialization.
  2. Subspace clustering + DivRank representative selection + LLM theme validation.
  3. Multi-label label entropy for uncertain-node selection.
  4. GNN-guided heuristic community expansion.
  5. Conflict-triggered LLM arbitration.
  6. Multi-round self-training + soft overlapping assignment.

Author: ANS-OCD project
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ANS_OCD.config import (
    DATA_DIR,
    DEVICE,
    GNNConfig,
    LLM_CONFIG,
    MULTILABEL_CONFIG,
    SUBSPACE_CONFIG,
    DIVRANK_CONFIG,
)
from ANS_OCD.utils.data_loader import load_graph_data, read_texts, load_cora_pt
from ANS_OCD.utils.metrics import evaluate_overlapping, membership_matrix_from_sets
from ANS_OCD.module1_representative.subspace_cluster import SubspaceClusterEstimator
from ANS_OCD.module1_representative.divrank_selector import (
    select_representatives_per_cluster,
    get_representative_texts,
)
from ANS_OCD.module2_gnn_entropy.gnn_model import MultiLabelGCN, multilabel_gnn_loss
from ANS_OCD.module2_gnn_entropy.label_entropy import select_high_entropy_nodes
from ANS_OCD.module3_expansion.community_expander import CommunityExpander
from ANS_OCD.module4_ctl_llm.prompt_engine import (
    build_theme_validation_prompt,
    build_conflict_resolution_prompt,
)
from ANS_OCD.module4_ctl_llm.ctl_mechanism import (
    validate_themes_with_llm,
    resolve_conflict_with_llm,
)
from ANS_OCD.module4_ctl_llm.conflict_detector import detect_conflicts, deduplicate_conflicts


def _to_numpy_adjacency(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    """Convert a PyG edge_index tensor to a dense numpy adjacency matrix."""
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    adj[src, dst] = 1.0
    adj[dst, src] = 1.0
    return adj


def _train_gnn_one_epoch(model: nn.Module, data, y_train: torch.Tensor,
                         optimizer: torch.optim.Optimizer) -> float:
    """Single training step using BCEWithLogitsLoss."""
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    loss = multilabel_gnn_loss(logits, y_train)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def run_ans_ocd(data_dir: Path = DATA_DIR,
                num_rounds: int = 3,
                use_text_embeddings: bool = True,
                return_synthetic: bool = False,
                dataset: str = "synthetic"):
    """
    Run the full ANS-OCD pipeline.

    Parameters
    ----------
    data_dir : Path
        Directory containing edges.csv, attributes.csv, labels.csv, texts.csv.
    num_rounds : int
        Number of self-training rounds.
    use_text_embeddings : bool
        Whether to extract Sentence-BERT embeddings from texts.
    return_synthetic : bool
        If True, generate a synthetic graph for quick smoke testing.
    dataset : str
        One of "synthetic", "cora", or "cora_csv". "cora" loads the pre-built
        cora_fixed_sbert.pt directly; "cora_csv" reads from data_dir/cora/ CSVs.

    Returns
    -------
    final_membership : np.ndarray, shape [N, K]
        Binary overlapping community membership matrix.
    metrics : dict or None
        Evaluation metrics if ground-truth labels are available.
    """
    # --- Stage 1: data loading + GNN initialization ---
    if dataset == "cora":
        print("[ANS-OCD] Running on Cora dataset (fixed SBERT split).")
        data = load_cora_pt()
        labels = data.y if data.y.ndim == 2 else None
        texts = data.texts if hasattr(data, "texts") else [f"Node {i}" for i in range(data.num_nodes)]
    elif return_synthetic or dataset == "synthetic":
        print("[ANS-OCD] Running on synthetic data for smoke testing.")
        from torch_geometric.data import Data
        N = 40
        x = torch.randn(N, 8)
        edge_index = torch.randint(0, N, (2, 120))
        data = Data(x=x, edge_index=edge_index)
        data.num_nodes = N
        labels = None
        texts = [f"Node {i} text" for i in range(N)]
    else:
        data = load_graph_data(data_dir, use_text_embeddings=use_text_embeddings)
        labels = data.y if data.y.ndim == 2 else None
        texts = read_texts(data_dir)

    num_nodes = data.num_nodes
    adj = _to_numpy_adjacency(data.edge_index, num_nodes)
    attr = data.x.detach().cpu().numpy()

    # --- Stage 2: adaptive K + representative selection ---
    print("[Stage 2] Subspace clustering and K estimation ...")
    estimator = SubspaceClusterEstimator(
        alpha=SUBSPACE_CONFIG["alpha"],
        max_k=SUBSPACE_CONFIG["max_k"],
    )
    estimator.fit(adj, attr)
    k_init = estimator.k_init
    print(f"  K_init estimated by Eigengap: {k_init}")

    cluster_labels = estimator.get_kmeans_labels()
    representatives, divrank_scores = select_representatives_per_cluster(
        adj, cluster_labels, reps_per_cluster=DIVRANK_CONFIG["reps_per_cluster"]
    )

    # LLM theme validation
    cluster_texts = get_representative_texts(representatives, texts or [f"Node {i}" for i in range(num_nodes)])
    validation_prompt = build_theme_validation_prompt(cluster_texts, k_init=k_init)
    k_final, themes, merges = validate_themes_with_llm(validation_prompt)
    print(f"  LLM validated K: {k_final}")
    print(f"  Themes: {themes}")
    print(f"  Merge suggestions: {merges}")

    # Initialize multi-label GNN with K_final communities.
    num_communities = k_final
    model = MultiLabelGCN(
        in_channels=data.x.shape[1],
        hidden_channels=GNNConfig["hidden_dim"],
        num_communities=num_communities,
        num_layers=GNNConfig["num_layers"],
        dropout=GNNConfig["dropout"],
    ).to(DEVICE)
    data = data.to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=GNNConfig["learning_rate"],
        weight_decay=GNNConfig["weight_decay"],
    )

    # Build initial pseudo-labels from representative seeds (one-hot per community).
    y_train = torch.zeros((num_nodes, num_communities), dtype=torch.float32, device=DEVICE)
    for cid, nodes in representatives.items():
        if cid >= num_communities:
            continue
        for n in nodes:
            if 0 <= n < num_nodes:
                y_train[n, cid] = 1.0

    # --- Stage 3-6: multi-round self-training ---
    prev_probs = None
    for r in range(num_rounds):
        print(f"\n[Stage 6] Self-training round {r + 1}/{num_rounds} ...")

        # Train GNN with current pseudo-labels.
        for _ in range(GNNConfig["epochs"] // max(1, num_rounds)):
            loss = _train_gnn_one_epoch(model, data, y_train, optimizer)

        # GNN predictions.
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            probs = torch.sigmoid(logits)

        # Stage 3: high-entropy nodes (report only; used in expansion/CTL).
        high_entropy_nodes = select_high_entropy_nodes(probs, top_ratio=0.1)
        print(f"  High-entropy nodes selected: {len(high_entropy_nodes)}")

        # Stage 4: GNN-guided expansion.
        expander = CommunityExpander(
            adj, attr, probs.detach().cpu().numpy(),
            pruning_threshold=MULTILABEL_CONFIG["pruning_threshold"],
        )
        expander.initialize(representatives)
        added = expander.expand_all(default_budget=20)
        print(f"  Expanded nodes per community: { {cid: len(nodes) for cid, nodes in added.items()} }")

        # Stage 5: CTL conflict detection and LLM arbitration.
        conflicts = detect_conflicts(expander)
        conflicts = deduplicate_conflicts(conflicts)
        print(f"  Conflicts detected: {len(conflicts)}")
        for node, cid, ctype in conflicts[:5]:  # resolve a few per round for speed
            neighbor_context = []
            if texts:
                neighbor_context = [texts[i] for i in range(num_nodes) if adj[node, i] > 0 and i < len(texts)]
            conflict_prompt = build_conflict_resolution_prompt(
                node_text=texts[node] if texts else f"Node {node}",
                node_id=node,
                neighbor_context=neighbor_context,
                themes={cid: name for cid, name in themes.items() if cid < num_communities},
                gnn_probs={cid: float(probs[node, cid].cpu()) for cid in range(num_communities)},
                heuristic_score=0.0,
            )
            assigned = resolve_conflict_with_llm(conflict_prompt)
            for c in assigned:
                if 0 <= c < num_communities:
                    expander.communities[c].add(node)

        # Update pseudo-labels for next round.
        y_train = torch.from_numpy(expander.get_membership_matrix()).float().to(DEVICE)
        # Also include high-confidence GNN predictions.
        y_train = torch.maximum(y_train, (probs >= MULTILABEL_CONFIG["prob_threshold"]).float())

        # Convergence check.
        if prev_probs is not None:
            change = torch.abs(probs - prev_probs).mean().item()
            print(f"  Mean probability change: {change:.6f}")
            if change < 1e-3:
                print("  Converged early.")
                break
        prev_probs = probs.clone()

    # --- Final soft overlapping assignment ---
    final_probs = probs.detach().cpu().numpy()
    final_membership = (final_probs >= MULTILABEL_CONFIG["overlap_threshold"]).astype(np.float32)
    communities = [np.where(final_membership[:, cid] == 1.0)[0].tolist() for cid in range(num_communities)]

    print("\n[ANS-OCD] Final overlapping communities:")
    for cid, nodes in enumerate(communities):
        print(f"  Community {cid}: {len(nodes)} nodes")

    metrics = None
    if labels is not None and labels.shape[1] == num_communities:
        true_communities = [np.where(labels[:, cid] == 1.0)[0].tolist() for cid in range(labels.shape[1])]
        metrics = evaluate_overlapping(communities, true_communities, num_nodes)
        print("\n[ANS-OCD] Evaluation metrics:", metrics)

    return final_membership, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ANS-OCD overlapping community detection.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "cora", "cora_csv"],
        help="Dataset to run on: synthetic (default), cora (pre-built .pt), or cora_csv (CSV folder).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing CSV data (used for cora_csv or custom datasets).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of self-training rounds.",
    )
    parser.add_argument(
        "--no-text-embeddings",
        action="store_true",
        help="Disable Sentence-BERT text embedding extraction for CSV datasets.",
    )
    args = parser.parse_args()

    membership, metrics = run_ans_ocd(
        data_dir=args.data_dir,
        num_rounds=args.rounds,
        use_text_embeddings=not args.no_text_embeddings,
        dataset=args.dataset,
    )
    print("\nANS-OCD completed. Final membership shape:", membership.shape)
