"""Convert the pre-built Cora .pt file into CSV format for ANS-OCD.

The script reads `data/cora_fixed_sbert.pt` and writes:
- data/cora/edges.csv
- data/cora/attributes.csv
- data/cora/labels.csv
- data/cora/texts.csv

After conversion, run:
    python -m ANS_OCD.main --dataset cora_csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "cora"


def main(pt_path: Path = None, output_dir: Path = None):
    if pt_path is None:
        pt_path = DATA_DIR / "cora_fixed_sbert.pt"
    if output_dir is None:
        output_dir = OUTPUT_DIR

    if not pt_path.exists():
        raise FileNotFoundError(
            f"Cora .pt file not found at {pt_path}. "
            "Please place cora_fixed_sbert.pt in the data/ directory."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[prepare_cora] Loading {pt_path}")
    data = torch.load(pt_path, map_location="cpu")

    num_nodes = data.num_nodes
    edge_index = data.edge_index.numpy()

    # 1. edges.csv (deduplicate undirected reverse edges)
    edges = np.stack([edge_index[0], edge_index[1]], axis=1)
    edges = pd.DataFrame(edges, columns=["source", "target"])
    # Keep only one direction for undirected edges.
    edges = edges[edges["source"] < edges["target"]].reset_index(drop=True)
    edges_path = output_dir / "edges.csv"
    edges.to_csv(edges_path, index=False)
    print(f"[prepare_cora] Wrote {len(edges)} edges to {edges_path}")

    # 2. attributes.csv (node features, e.g. SBERT embeddings)
    x = data.x.numpy().astype(np.float32)
    attrs_path = output_dir / "attributes.csv"
    np.savetxt(attrs_path, x, delimiter=",", fmt="%.6f")
    print(f"[prepare_cora] Wrote {x.shape} attributes to {attrs_path}")

    # 3. labels.csv (multi-label compatible format, one class per node)
    y = data.y.numpy() if data.y.dim() == 1 else data.y.argmax(axis=1).numpy()
    # Use multi-label separator so ANS-OCD reads it as a binary matrix.
    label_strs = [f"{int(label)};" for label in y]
    labels = pd.DataFrame({"label": label_strs})
    labels_path = output_dir / "labels.csv"
    labels.to_csv(labels_path, index=False)
    print(f"[prepare_cora] Wrote {len(labels)} labels to {labels_path}")

    # 4. texts.csv
    texts = data.raw_texts if hasattr(data, "raw_texts") else data.raw_text
    texts_df = pd.DataFrame({"text": texts})
    texts_path = output_dir / "texts.csv"
    texts_df.to_csv(texts_path, index=False)
    print(f"[prepare_cora] Wrote {len(texts_df)} texts to {texts_path}")

    print(f"[prepare_cora] Done. Run: python -m ANS_OCD.main --dataset cora_csv")


if __name__ == "__main__":
    main()
