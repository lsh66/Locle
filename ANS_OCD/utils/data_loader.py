"""
Data loading utilities for overlapping community detection.

Supports:
- Reading edge lists and node attributes from CSV files.
- Extracting text embeddings using Sentence-BERT.
- Building PyTorch Geometric Data objects.
- Multi-label label loading where each node may belong to multiple communities.

Author: ANS-OCD project
"""
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from torch_geometric.data import Data

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from ANS_OCD.config import DEVICE, SBERT_MODEL, PROJECT_ROOT


# Mapping from dataset name to its pre-built .pt filename in data/.
PT_DATASET_FILES = {
    "cora": "cora_fixed_sbert.pt",
    "citeseer": "citeseer2_fixed_sbert.pt",
    "pubmed": "pubmed_fixed_sbert.pt",
    "wikics": "wikics_fixed_sbert.pt",
    "dblp": "dblp_fixed_sbert.pt",
    "amazon": "amazon_fixed_attr.pt",
    "ppi": "ppi_fixed_attr.pt",
    "ppi_bp": "ppi_bp_fixed_struct.pt",
    "blogcatalog": "blogcatalog_fixed_struct.pt",
    "facebook": "facebook_fixed_attr.pt",
}


def load_pt_dataset(name: str = "cora", pt_path: Union[str, Path] = None) -> Data:
    """
    Load a dataset from a PyTorch Geometric .pt file.

    Supported names (see PT_DATASET_FILES): cora, citeseer, pubmed.

    The .pt file is expected to contain a torch_geometric.data.Data object with:
    - edge_index: torch.LongTensor [2, E]
    - x: torch.FloatTensor [N, F] (e.g. SBERT embeddings)
    - y: torch.LongTensor [N] (single-label class indices)
    - raw_texts: List[str] of length N (paper titles/abstracts)
    - label_names: List[str] of class names

    Parameters
    ----------
    name : str
        Dataset name key in PT_DATASET_FILES. Ignored if pt_path is given.
    pt_path : str or Path, optional
        Explicit path to a .pt file. Overrides `name`.

    Returns
    -------
    data : torch_geometric.data.Data
        PyG Data object compatible with the ANS-OCD pipeline.
    """
    if pt_path is None:
        if name not in PT_DATASET_FILES:
            raise ValueError(
                f"Unknown dataset '{name}'. Available: {list(PT_DATASET_FILES)}"
            )
        pt_path = PROJECT_ROOT / "data" / PT_DATASET_FILES[name]
    pt_path = Path(pt_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"Dataset .pt file not found: {pt_path}")

    data = torch.load(pt_path, map_location="cpu")
    # Ensure required fields exist.
    if not hasattr(data, "num_nodes") or data.num_nodes is None:
        data.num_nodes = data.x.shape[0]

    # Convert labels to one-hot multi-label matrix if they are class indices.
    if data.y.dim() == 1:
        num_classes = int(data.y.max().item()) + 1
        one_hot = torch.zeros((data.num_nodes, num_classes), dtype=torch.float32)
        one_hot[torch.arange(data.num_nodes), data.y] = 1.0
        data.y = one_hot

    # Attach raw texts as a node-level attribute if available.
    if hasattr(data, "raw_texts") and data.raw_texts:
        data.texts = data.raw_texts
    elif hasattr(data, "raw_text") and data.raw_text:
        data.texts = data.raw_text

    return data




def _find_csv_path(data_dir: Path, name: str) -> Optional[Path]:
    """Find a CSV file by name (case-insensitive) in data_dir."""
    if not data_dir.exists():
        return None
    candidates = [f for f in data_dir.iterdir() if f.is_file() and f.stem.lower() == name.lower()]
    if not candidates:
        return None
    return candidates[0]


def read_edge_list(data_dir: Union[str, Path], delimiter: str = ",",
                   src_col: str = "source", dst_col: str = "target") -> Tuple[np.ndarray, int]:
    """
    Read edge list from CSV and return an edge_index array plus number of nodes.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing edges.csv (or similar).
    delimiter : str
        CSV delimiter.
    src_col, dst_col : str
        Column names for source/target nodes.

    Returns
    -------
    edge_index : np.ndarray, shape [2, E]
        Undirected edge indices (duplicated reverse edges).
    num_nodes : int
        Number of nodes inferred from max node id + 1.
    """
    data_dir = Path(data_dir)
    edge_path = _find_csv_path(data_dir, "edges")
    if edge_path is None:
        raise FileNotFoundError(f"No edges file found in {data_dir}")

    df = pd.read_csv(edge_path, delimiter=delimiter)
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    src_col = src_col.lower()
    dst_col = dst_col.lower()
    if src_col not in df.columns or dst_col not in df.columns:
        raise ValueError(f"Edge CSV columns {df.columns} do not contain {src_col}/{dst_col}")

    src = df[src_col].astype(int).values
    dst = df[dst_col].astype(int).values
    num_nodes = int(max(src.max(), dst.max()) + 1)

    # Add reverse edges to make the graph undirected.
    edge_index = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0)
    return edge_index, num_nodes


def read_attributes(data_dir: Union[str, Path], delimiter: str = ",",
                    num_nodes: Optional[int] = None) -> Optional[np.ndarray]:
    """
    Read numeric node attribute matrix from CSV.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing attributes.csv.
    delimiter : str
        CSV delimiter.
    num_nodes : int, optional
        Expected number of rows; if not provided, inferred from file.

    Returns
    -------
    attr_matrix : np.ndarray or None
        Numeric attribute matrix of shape [num_nodes, num_features].
    """
    data_dir = Path(data_dir)
    attr_path = _find_csv_path(data_dir, "attributes")
    if attr_path is None:
        attr_path = _find_csv_path(data_dir, "attr")
    if attr_path is None:
        return None

    df = pd.read_csv(attr_path, delimiter=delimiter, header=None)
    attr_matrix = df.values.astype(np.float32)
    if num_nodes is not None and attr_matrix.shape[0] != num_nodes:
        raise ValueError(f"Attribute matrix rows {attr_matrix.shape[0]} != num_nodes {num_nodes}")
    return attr_matrix


def read_texts(data_dir: Union[str, Path], delimiter: str = ",",
               text_col: str = "text", default_text: str = "") -> List[str]:
    """
    Read node texts from CSV. If no text file is found, return empty list.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing texts.csv or similar.
    delimiter : str
        CSV delimiter.
    text_col : str
        Column name containing text.
    default_text : str
        Fallback text when file is missing.

    Returns
    -------
    texts : List[str]
        List of raw texts per node.
    """
    data_dir = Path(data_dir)
    for name in ["texts", "text", "content", "abstract"]:
        text_path = _find_csv_path(data_dir, name)
        if text_path is not None:
            break
    else:
        return []

    df = pd.read_csv(text_path, delimiter=delimiter, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    col = text_col.lower()
    if col not in df.columns:
        col = df.columns[0]
    texts = df[col].fillna(default_text).astype(str).tolist()
    return texts


def extract_text_embeddings(texts: List[str], model_name: Optional[str] = None,
                            batch_size: int = 64) -> np.ndarray:
    """
    Extract Sentence-BERT embeddings for a list of texts.

    Parameters
    ----------
    texts : List[str]
        Raw text list.
    model_name : str, optional
        Sentence-BERT model name; defaults to config.SBERT_MODEL.
    batch_size : int
        Encoding batch size.

    Returns
    -------
    embeddings : np.ndarray, shape [num_nodes, embedding_dim]
        L2-normalized text embeddings.
    """
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is required for text embedding extraction.")
    model_name = model_name or SBERT_MODEL
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    # L2 normalization for stable downstream similarity computations.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-12)
    return embeddings.astype(np.float32)


def read_labels(data_dir: Union[str, Path], delimiter: str = ",",
                label_col: str = "label", multi_label_sep: str = ";") -> Optional[np.ndarray]:
    """
    Read node labels. Supports both single-label and multi-label formats.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing labels.csv.
    delimiter : str
        CSV delimiter.
    label_col : str
        Column name for labels.
    multi_label_sep : str
        Separator for multiple labels in a single cell (e.g., "1;3").

    Returns
    -------
    labels : np.ndarray or None
        For single label: shape [num_nodes] with integer labels.
        For multi label: shape [num_nodes, num_classes] binary matrix.
    """
    data_dir = Path(data_dir)
    label_path = _find_csv_path(data_dir, "labels")
    if label_path is None:
        return None

    df = pd.read_csv(label_path, delimiter=delimiter, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    col = label_col.lower()
    if col not in df.columns:
        col = df.columns[0]
    raw = df[col].fillna("").astype(str).tolist()

    # Detect multi-label by separator presence.
    if any(multi_label_sep in r for r in raw if r):
        label_sets = []
        for r in raw:
            if r.strip() == "":
                label_sets.append([])
            else:
                label_sets.append([int(x.strip()) for x in r.split(multi_label_sep) if x.strip()])
        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(label_sets).astype(np.float32)
    else:
        labels = np.array([int(x.strip()) if x.strip() else -1 for x in raw], dtype=np.int64)
    return labels


def load_graph_data(data_dir: Union[str, Path], use_text_embeddings: bool = True,
                    delimiter: str = ",", label_col: str = "label") -> Data:
    """
    Load graph topology, attributes, texts, and labels into a PyG Data object.

    The returned Data object has:
    - edge_index : torch.LongTensor [2, E]
    - x : torch.FloatTensor [num_nodes, feature_dim]
    - y : torch.Tensor [num_nodes, ...] (multi-label binary or single-label indices)
    - num_nodes : int

    Parameters
    ----------
    data_dir : str or Path
        Directory containing edges.csv, attributes.csv, labels.csv, texts.csv (optional).
    use_text_embeddings : bool
        Whether to extract Sentence-BERT embeddings when texts are available.
    delimiter : str
        CSV delimiter.
    label_col : str
        Label column name.

    Returns
    -------
    data : torch_geometric.data.Data
        PyG data object ready for GNN training.
    """
    data_dir = Path(data_dir)
    edge_index, num_nodes = read_edge_list(data_dir, delimiter=delimiter)
    edge_index = torch.from_numpy(edge_index).long()

    attr_matrix = read_attributes(data_dir, delimiter=delimiter, num_nodes=num_nodes)
    texts = read_texts(data_dir, delimiter=delimiter)

    if use_text_embeddings and texts:
        text_emb = extract_text_embeddings(texts)
        if attr_matrix is not None:
            x = np.concatenate([attr_matrix, text_emb], axis=1)
        else:
            x = text_emb
    elif attr_matrix is not None:
        x = attr_matrix
    else:
        # Fallback: one-hot degree + identity just so the model can run.
        adj = coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0].numpy(), edge_index[1].numpy())),
                         shape=(num_nodes, num_nodes))
        degrees = np.array(adj.sum(axis=1)).flatten()
        x = np.eye(num_nodes, dtype=np.float32)
        x = np.concatenate([x, degrees[:, None]], axis=1)

    x = torch.from_numpy(x).float()

    labels = read_labels(data_dir, delimiter=delimiter, label_col=label_col)
    if labels is None:
        y = torch.zeros(num_nodes, dtype=torch.float32)
    elif labels.ndim == 2:
        y = torch.from_numpy(labels).float()
    else:
        y = torch.from_numpy(labels).long()

    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_nodes = num_nodes
    return data





if __name__ == "__main__":
    # 自测代码：构建小型合成数据集，验证数据加载器运行逻辑
    import tempfile
    import shutil

    # 创建临时文件夹
    tmp = tempfile.mkdtemp()
    try:
        # 转为路径对象方便文件操作
        tmp_path = Path(tmp)
        # 构建边文件：0、1、2三个节点构成三角形环图 0-1,1-2,2-0
        pd.DataFrame({"source": [0, 1, 2], "target": [1, 2, 0]}).to_csv(tmp_path / "edges.csv", index=False)
        # 构建节点属性特征文件
        np.random.seed(0)  # 固定随机种子，保证测试结果可复现
        pd.DataFrame(np.random.rand(3, 5)).to_csv(tmp_path / "attributes.csv", index=False, header=False)
        # 构建多标签文件，标签使用分号;分隔
        pd.DataFrame({"label": ["0;1", "1", "0"]}).to_csv(tmp_path / "labels.csv", index=False)
        # 构建节点文本描述文件
        pd.DataFrame({"text": ["machine learning", "graph neural network", "community detection"]}).to_csv(
            tmp_path / "texts.csv", index=False)

        # 加载图数据集，开启文本嵌入特征
        data = load_graph_data(tmp_path, use_text_embeddings=True)
        # 打印图边索引维度
        print("Edge index shape:", data.edge_index.shape)
        # 打印节点特征矩阵维度
        print("Feature matrix shape:", data.x.shape)
        # 打印标签张量维度
        print("Label tensor shape:", data.y.shape)
        # 输出解析后的标签
        print("Labels:\n", data.y)
    finally:
        # 无论测试是否报错，最终删除临时文件夹，释放磁盘空间
        shutil.rmtree(tmp)
