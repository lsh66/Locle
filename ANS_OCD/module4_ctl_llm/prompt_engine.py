"""
Prompt engineering for the LLM-based theme validation / conflict resolution module.

Responsibilities:
- Build prompts that ask the LLM to validate community themes from representative nodes.
- Build prompts that ask the LLM to resolve conflicts between heuristic rules and GNN predictions.

Author: ANS-OCD project
"""
from typing import Dict, List, Optional


def build_theme_validation_prompt(cluster_texts: Dict[int, str],
                                  k_init: int,
                                  candidate_merge: bool = True) -> str:
    """
    Build a prompt asking the LLM to validate community themes and the initial K.

    The prompt supplies representative-node text blocks for each cluster and asks
    the LLM to return the final number of communities, theme names, and merge
    suggestions.

    Parameters
    ----------
    cluster_texts : Dict[int, str]
        Mapping from cluster id to a text block describing representative nodes.
    k_init : int
        Initial K estimated by subspace clustering.
    candidate_merge : bool
        Whether to ask the LLM to evaluate whether some clusters should be merged.

    Returns
    -------
    prompt : str
        The formatted prompt.
    """
    blocks = [f"Initial number of clusters (K_init) estimated by subspace clustering: {k_init}\n"]
    for cid in sorted(cluster_texts):
        blocks.append(cluster_texts[cid])
        blocks.append("")

    merge_question = (
        "If some clusters are too similar or actually describe the same research theme, "
        "please suggest merging them."
    ) if candidate_merge else ""

    prompt = (
        "You are an expert in scientific literature analysis and community detection.\n"
        "The following clusters were obtained from an academic citation network. "
        "Each cluster is represented by a few high-impact and diverse nodes.\n\n"
        + "\n".join(blocks) +
        "\nTasks:\n"
        "1. Determine the most appropriate number of communities K (an integer). "
        "K must be at least 1 and no larger than the initial K_init shown above.\n"
        "2. Provide a short, human-readable theme name for each confirmed community.\n"
        "3. " + merge_question + "\n\n"
        "Return your answer ONLY in the following strict format, no extra text:\n\n"
        "K: <integer>\n"
        "Theme_1: <name for community 1>\n"
        "Theme_2: <name for community 2>\n"
        "...\n"
        "Merge: <semicolon-separated list of cluster ids to merge, or 'None'>\n"
    )
    return prompt


def build_conflict_resolution_prompt(node_text: str,
                                     node_id: int,
                                     neighbor_context: List[str],
                                     themes: Dict[int, str],
                                     gnn_probs: Dict[int, float],
                                     heuristic_score: float) -> str:
    """
    Build a prompt for the upgraded CTL mechanism: ask the LLM to resolve a
    conflict between GNN prediction and heuristic rule expansion.

    Parameters
    ----------
    node_text : str
        Raw text of the conflict node.
    node_id : int
        Node id.
    neighbor_context : List[str]
        Short text snippets of neighboring nodes, optionally grouped by their
        current communities.
    themes : Dict[int, str]
        Mapping from community id to theme name.
    gnn_probs : Dict[int, float]
        GNN-predicted membership probabilities for the conflict node per community.
    heuristic_score : float
        Heuristic delta C_o score (positive means the rule strongly suggests
        adding the node to the community with the highest score).

    Returns
    -------
    prompt : str
        The formatted prompt.
    """
    theme_lines = "\n".join([f"  Community {cid}: {name}" for cid, name in themes.items()])
    prob_lines = "\n".join([f"  Community {cid}: {p:.4f}" for cid, p in gnn_probs.items()])

    neighbor_text = "\n".join([f"  - {t[:200]}" for t in neighbor_context[:5]])

    prompt = (
        f"You are an expert in overlapping community detection.\n"
        f"A conflict node is at the boundary of multiple communities. "
        f"The graph-neural-network prediction and the heuristic expansion rule disagree.\n\n"
        f"Node ID: {node_id}\n"
        f"Node text: {node_text[:500]}\n\n"
        f"Community themes:\n{theme_lines}\n\n"
        f"GNN membership probabilities for this node:\n{prob_lines}\n\n"
        f"Heuristic rule score (higher means the rule strongly favors membership): {heuristic_score:.4f}\n\n"
        f"Top neighbor texts:\n{neighbor_text}\n\n"
        f"Task: Based on the node text and neighbor context, decide which communities "
        f"this node belongs to. The answer can contain multiple communities.\n\n"
        f"Return ONLY a comma-separated list of community ids, e.g., '1,3' or '2'. "
        f"If the node does not belong to any community, return 'None'."
    )
    return prompt


if __name__ == "__main__":
    # Self-test: display a generated validation prompt.
    cluster_texts = {
        0: "Cluster 0 representative nodes:\n  [1] Node 0: deep learning for graphs\n  [2] Node 5: graph neural networks",
        1: "Cluster 1 representative nodes:\n  [1] Node 2: community detection in social networks\n  [2] Node 8: overlapping community discovery",
    }
    print(build_theme_validation_prompt(cluster_texts, k_init=2))
    print("\n" + "=" * 60 + "\n")
    print(build_conflict_resolution_prompt(
        node_text="We study graph transformers and attention mechanisms.",
        node_id=42,
        neighbor_context=["graph neural networks", "transformer architecture", "attention models"],
        themes={0: "Graph Neural Networks", 1: "Transformers & Attention"},
        gnn_probs={0: 0.85, 1: 0.62},
        heuristic_score=0.05
    ))
