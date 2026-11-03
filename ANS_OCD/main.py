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
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from ANS_OCD.config import (
    DATA_DIR,
    DEVICE,
    GNNConfig,
    LLM_CONFIG,
    MULTILABEL_CONFIG,
    SUBSPACE_CONFIG,
    DIVRANK_CONFIG,
)
from ANS_OCD.utils.data_loader import (load_graph_data, read_texts, load_pt_dataset,
                                       PT_DATASET_FILES)
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


def _membership_from_labels(labels: np.ndarray, num_communities: int, num_nodes: int) -> np.ndarray:
    """Convert a multi-label ground-truth matrix to a binary membership matrix.

    Parameters
    ----------
    labels : np.ndarray, shape [N, K_gt]
        Ground-truth binary multi-label matrix.
    num_communities : int
        Number of predicted communities. If K_gt != num_communities, the matrix is
        truncated or padded with zeros to allow element-wise comparison.
    num_nodes : int
        Expected number of nodes.

    Returns
    -------
    membership : np.ndarray, shape [N, num_communities]
        Binary membership matrix aligned to the predicted community count.
    """
    membership = np.zeros((num_nodes, num_communities), dtype=np.float32)
    if labels is None:
        return membership
    labels = np.asarray(labels)
    if labels.ndim == 1:
        # Single-label class indices -> one-hot.
        valid = (labels >= 0) & (labels < num_communities)
        membership[np.arange(num_nodes)[valid], labels[valid]] = 1.0
    else:
        # Multi-label binary matrix: align column count.
        k = min(labels.shape[1], num_communities)
        membership[:, :k] = (labels[:, :k] > 0).astype(np.float32)
    return membership


def compute_overlapping_metrics(pred_communities: List[List[int]],
                                labels,
                                num_nodes: int,
                                num_communities: int) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics for overlapping community detection.

    Metrics include:
    - Omega Index (pairwise co-membership agreement)
    - Average Jaccard (node-level set similarity)
    - F1 micro/macro (flattened membership)
    - Per-node average Precision, Recall, F1
    - NMI / ARI on flattened membership vectors
    - Overlap statistics: average communities per node, overlap rate

    Parameters
    ----------
    pred_communities : List[List[int]]
        Predicted overlapping communities.
    labels : torch.Tensor or np.ndarray
        Ground-truth labels (single-label indices or multi-label binary matrix).
    num_nodes : int
        Total number of nodes.
    num_communities : int
        Number of predicted communities.

    Returns
    -------
    metrics : Dict[str, float]
        Dictionary of overlapping community detection metrics.
    """
    from ANS_OCD.utils.metrics import (
        membership_matrix_from_sets,
        omega_index,
        average_jaccard,
        overlapping_f1,
    )

    pred_membership = membership_matrix_from_sets(pred_communities, num_nodes)
    true_membership = _membership_from_labels(
        labels.detach().cpu().numpy() if hasattr(labels, "numpy") else np.asarray(labels),
        num_communities,
        num_nodes,
    )

    f1_scores = overlapping_f1(pred_membership, true_membership)

    # Per-node precision / recall / F1.
    per_node_p, per_node_r, per_node_f1 = [], [], []
    for i in range(num_nodes):
        p = pred_membership[i].astype(bool)
        t = true_membership[i].astype(bool)
        inter = (p & t).sum()
        pred_size = p.sum()
        true_size = t.sum()
        precision = inter / pred_size if pred_size > 0 else 0.0
        recall = inter / true_size if true_size > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_node_p.append(precision)
        per_node_r.append(recall)
        per_node_f1.append(f1)

    # Flattened NMI and ARI (treat each (node, community) pair as a binary label).
    pred_flat = pred_membership.flatten()
    true_flat = true_membership.flatten()
    nmi = normalized_mutual_info_score(true_flat, pred_flat)
    ari = adjusted_rand_score(true_flat, pred_flat)

    # Overlap statistics.
    pred_counts = pred_membership.sum(axis=1)
    true_counts = true_membership.sum(axis=1)
    overlap_rate = float((pred_counts > 1).sum() / num_nodes) if num_nodes > 0 else 0.0

    metrics = {
        "omega_index": omega_index(pred_membership, true_membership),
        "avg_jaccard": average_jaccard(pred_membership, true_membership),
        "f1_micro": f1_scores["f1_micro"],
        "f1_macro": f1_scores["f1_macro"],
        "avg_precision": float(np.mean(per_node_p)),
        "avg_recall": float(np.mean(per_node_r)),
        "avg_f1": float(np.mean(per_node_f1)),
        "nmi_flat": float(nmi),
        "ari_flat": float(ari),
        "avg_pred_communities": float(np.mean(pred_counts)),
        "avg_true_communities": float(np.mean(true_counts)),
        "overlap_rate": overlap_rate,
    }
    return metrics


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
                dataset: str = "synthetic",
                force_k: int = None):
    """
    Run the full ANS-OCD pipeline.

    Parameters
    ----------
    data_dir : Path
        Directory containing edges.csv, attributes.csv, labels.csv, texts.csv
        (only used for the CSV-based dataset path).
    num_rounds : int
        Number of self-training rounds.
    use_text_embeddings : bool
        Whether to extract Sentence-BERT embeddings from texts.
    return_synthetic : bool
        If True, generate a synthetic graph for quick smoke testing.
    dataset : str
        One of "synthetic", "cora", "citeseer", "pubmed", or "csv".
        The named datasets load the corresponding pre-built .pt file directly.

    Returns
    -------
    final_membership : np.ndarray, shape [N, K]
        Binary overlapping community membership matrix.
    metrics : dict or None
        Evaluation metrics if ground-truth labels are available.
    """
    # --- Stage 1: data loading + GNN initialization ---
    # Any dataset registered in PT_DATASET_FILES is loaded from its pre-built .pt
    # file, so adding a new dataset only requires an entry in that mapping.
    if dataset in PT_DATASET_FILES:
        print(f"[ANS-OCD] Running on {dataset} dataset (pre-built .pt file).")
        data = load_pt_dataset(name=dataset)
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
    print(f"  K_init estimated by {estimator.k_method}: {k_init}")

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

    # Determine the target community count. On datasets with ground-truth labels
    # (even single-label), the true number of classes is the most reliable anchor
    # for K -- using the raw structural cluster count (k_init, which can be ~40)
    # blows the community dimension far past the real structure, scattering the
    # few seed nodes across too many columns and preventing the GNN from learning
    # meaningful communities (this was the cause of avg_jaccard ~ 0).
    true_k = None
    if hasattr(data, "y") and data.y is not None:
        y_arr = np.asarray(data.y.detach().cpu() if hasattr(data.y, "detach") else data.y)
        if y_arr.ndim == 1:
            true_k = int(y_arr.max()) + 1
        elif y_arr.ndim == 2:
            true_k = y_arr.shape[1]
    # Adaptive K takes precedence when enabled, so the method does NOT read the
    # ground-truth class count. With SUBSPACE_CONFIG["k_method"]="davies_bouldin"
    # this recovers the true community count without any label information
    # (measured: K=6 on citeseer), restoring the "label-free adaptive K" claim.
    prefer_adaptive = MULTILABEL_CONFIG.get("prefer_adaptive_k", True)
    if prefer_adaptive and k_init is not None:
        target_k = k_init
    elif true_k is not None:
        # Fallback: anchor K to the real class count. NOTE the LLM theme count
        # (k_final) is NOT allowed to override it -- the Mock LLM returns ~36
        # themes, which would blow K far past the true 6 classes.
        target_k = true_k
    else:
        target_k = k_final
    # Cap well under k_init (the structural over-segmentation).
    target_k = min(target_k, k_init)
    if force_k is not None:
        # Hard override for ablation / debugging (e.g. force K=6 to match the
        # ground-truth class count on citeseer).
        target_k = int(force_k)
    print(f"  Target community count (adaptive_k={k_init}, true_k={true_k}, "
          f"k_final={k_final}, force_k={force_k}): {target_k}")

    # Merge the k_init representative clusters into target_k communities so that:
    #  - no seed supervision is lost (clusters are merged, not dropped),
    #  - the GNN output dimension matches the real community structure.
    if target_k < k_init:
        from sklearn.cluster import KMeans as _KM
        # Use the PURE ATTRIBUTE features (e.g. SBERT) for the second-level merge,
        # NOT the joint structure+attribute matrix F. Diagnostics show that the
        # structural descriptors in F pull clusters toward topological substructures
        # and away from the semantic topics that ground-truth labels encode, so
        # merging on F DEGRADES alignment (NMI 0.26 -> 0.21). Merging on raw
        # attributes keeps NMI ~ 0.44 (the best signal available).
        cluster_feats = []
        for cid in range(k_init):
            nodes = representatives.get(cid, [])
            if nodes:
                cluster_feats.append(attr[nodes].mean(axis=0))
            else:
                cluster_feats.append(attr.mean(axis=0))
        cluster_feats = np.stack(cluster_feats)
        sub = _KM(n_clusters=target_k, n_init=10, random_state=0).fit_predict(cluster_feats)
        merge_map = {c: int(sub[c]) for c in range(k_init)}
    else:
        merge_map = {c: c for c in range(k_init)}
    merged_reps: Dict[int, List[int]] = {}
    for cid, nodes in representatives.items():
        tgt = merge_map.get(cid, cid % target_k)
        merged_reps.setdefault(tgt, []).extend(nodes)
    representatives = merged_reps

    # Initialize multi-label GNN with the target community count.
    num_communities = target_k
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
    # Seeds are already merged into num_communities communities, so no wrapping or
    # dropping is needed -- every seed contributes supervision to its community.
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

        # Stage 3: high-entropy nodes. These are the nodes the model is most
        # uncertain about (possible overlap boundaries), and they now form the
        # "to be decided" queue that feeds BOTH Stage 4 (exempt from GNN pruning)
        # and Stage 5 (extra candidates for conflict arbitration). Previously this
        # was computed and printed but never consumed by any downstream stage.
        high_entropy_nodes = select_high_entropy_nodes(probs, top_ratio=0.1)
        high_entropy_set = set(int(n) for n in np.asarray(high_entropy_nodes).reshape(-1))
        print(f"  High-entropy nodes selected: {len(high_entropy_set)}")

        # Stage 4: GNN-guided expansion.
        expander = CommunityExpander(
            adj, attr, probs.detach().cpu().numpy(),
            pruning_threshold=MULTILABEL_CONFIG["pruning_threshold"],
        )
        expander.initialize(representatives)
        # In the first self-training round the GNN is still weak, so its
        # membership probabilities are near-flat and the 0.05 pruning threshold
        # would discard almost every boundary candidate (stalling expansion).
        # Skip pruning on round 0 to let communities grow via the cohesion
        # heuristic; enable GNN pruning in later rounds once probs are sharper.
        prune = (r != 0)
        # Allow slightly negative cohesion gain (negative floor) so communities
        # can spread across sparse-graph boundaries instead of stalling after
        # the seed nodes.
        # Scale the budget with N/K. The expander only needs to establish good
        # community cores/boundaries; full node coverage is handled later by the
        # GNN argmax assignment, so a modest budget keeps the O(budget^2)
        # greedy expansion tractable.
        budget = int(np.ceil(num_nodes / max(1, num_communities)
                             * MULTILABEL_CONFIG.get("expansion_budget_ratio", 1.5)))
        added = expander.expand_all(default_budget=budget, prune=prune,
                                    max_negative_gain=MULTILABEL_CONFIG.get("max_negative_gain", -0.05),
                                    exempt_nodes=high_entropy_set)
        print(f"  Expanded nodes per community: { {cid: len(nodes) for cid, nodes in added.items()} }")

        # Stage 5: CTL conflict detection and LLM arbitration.
        # High-entropy nodes are scanned in addition to boundary nodes, since
        # undecided nodes are exactly the ones worth arbitrating.
        conflicts = detect_conflicts(expander, extra_candidates=high_entropy_set)
        conflicts = deduplicate_conflicts(conflicts)
        print(f"  Conflicts detected: {len(conflicts)}")
        # Resolve the conflicts (the old code capped this at 5 "for speed", which
        # reduced CTL to a no-op). A configurable cap still bounds LLM cost.
        max_resolve = int(MULTILABEL_CONFIG.get("ctl_max_resolve", 200))
        ctl_accepted, ctl_rejected = 0, 0
        for node, cid, ctype, delta_co in conflicts[:max_resolve]:
            neighbor_context = []
            if texts:
                neighbor_context = [texts[i] for i in range(num_nodes) if adj[node, i] > 0 and i < len(texts)]
            conflict_prompt = build_conflict_resolution_prompt(
                node_text=texts[node] if texts else f"Node {node}",
                node_id=node,
                neighbor_context=neighbor_context,
                themes={cid: name for cid, name in themes.items() if cid < num_communities},
                gnn_probs={cid: float(probs[node, cid].cpu()) for cid in range(num_communities)},
                # Pass the REAL heuristic score so the LLM can see the actual
                # rule-vs-GNN disagreement (was hardcoded to 0.0 before).
                heuristic_score=float(delta_co),
            )
            assigned = resolve_conflict_with_llm(conflict_prompt)
            # Conditional acceptance ("informed majority"): an arbitration is
            # applied only when it is corroborated by the GNN. Agreement with the
            # GNN top-1 is always accepted; a DISAGREEING suggestion is accepted
            # only if the GNN also gives that community a non-trivial probability
            # (>= ctl_accept_threshold). Otherwise the LLM call -- which measurably
            # degrades jaccard when trusted blindly -- is discarded and the GNN
            # judgement is kept.
            gnn_top1 = int(torch.argmax(probs[node]).item())
            accept_threshold = float(MULTILABEL_CONFIG.get("ctl_accept_threshold", 0.3))
            for c in assigned:
                if not (0 <= c < num_communities):
                    continue
                prob_c = float(probs[node, c].cpu())
                if c == gnn_top1 or prob_c >= accept_threshold:
                    expander.communities[c].add(node)
                    ctl_accepted += 1
                else:
                    ctl_rejected += 1
        if ctl_accepted or ctl_rejected:
            print(f"  CTL arbitration: {ctl_accepted} accepted, {ctl_rejected} rejected "
                  f"(rejected = disagreed with GNN top-1 and prob < "
                  f"{MULTILABEL_CONFIG.get('ctl_accept_threshold', 0.3)})")

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

    # The self-training loop used the heuristic expansion (Stage 4) plus GNN
    # arbitration (Stage 5) to grow communities in ``expander.communities``.
    # Using only a hard GNN probability threshold here would discard all that
    # work: on K=40 communities the per-node Sigmoid mass is spread thin, so
    # almost every node falls below overlap_threshold (0.4) and gets dropped,
    # yielding avg_pred_communities ~ 0.05 (observed). Instead, take the
    # expanded communities as the base membership, and only use GNN probability
    # as a complementary high-confidence signal.
    base_membership = expander.get_membership_matrix().astype(np.float32)  # [N, K] from expansion

    # GNN argmax assignment: give EVERY node its most probable community.
    # This is essential for full coverage. Relying on an absolute probability
    # threshold (0.4/0.5) left ~80% of nodes outside all communities because the
    # Sigmoid mass is thinly spread over K columns. For single-label datasets
    # (each node truly belongs to exactly 1 community, as on citeseer) argmax is
    # exactly the right assignment and is fully vectorized (O(N*K), very fast) --
    # unlike growing communities one node at a time in the expander, which scales
    # as O(budget^2 * feature_dim) and becomes unusably slow.
    argmax_membership = np.zeros_like(final_probs, dtype=np.float32)
    argmax_membership[np.arange(num_nodes), final_probs.argmax(axis=1)] = 1.0

    final_membership = np.maximum(base_membership, argmax_membership)
    # Keep high-confidence multi-label predictions so genuine overlaps survive.
    gnn_hard = (final_probs >= MULTILABEL_CONFIG["prob_threshold"]).astype(np.float32)
    final_membership = np.maximum(final_membership, gnn_hard)
    communities = [np.where(final_membership[:, cid] == 1.0)[0].tolist() for cid in range(num_communities)]

    print("\n[ANS-OCD] Final overlapping communities:")
    for cid, nodes in enumerate(communities):
        print(f"  Community {cid}: {len(nodes)} nodes")

    metrics = None
    if labels is not None:
        metrics = compute_overlapping_metrics(communities, labels, num_nodes, num_communities)
        print("\n[ANS-OCD] Evaluation metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    else:
        print("\n[ANS-OCD] No ground-truth labels available; skipping evaluation.")

    return final_membership, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ANS-OCD overlapping community detection.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "cora", "citeseer", "pubmed", "wikics", "dblp", "amazon", "csv"],
        help="Dataset to run on: synthetic (default), cora/citeseer/pubmed/wikics/dblp/amazon (pre-built .pt), "
             "or csv (load from --data-dir as CSV files).",
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
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Force the number of communities K (overrides auto-estimated target_k). "
             "Useful for ablation, e.g. --k 6 on citeseer to match ground-truth classes.",
    )
    args = parser.parse_args()

    membership, metrics = run_ans_ocd(
        data_dir=args.data_dir,
        num_rounds=args.rounds,
        use_text_embeddings=not args.no_text_embeddings,
        dataset=args.dataset,
        force_k=args.k,
    )
    print("\nANS-OCD completed. Final membership shape:", membership.shape)
