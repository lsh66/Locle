"""
Multi-label GNN models for overlapping community detection.

The models here use Sigmoid-activated independent outputs so each node can
belong to multiple communities simultaneously.

Author: ANS-OCD project
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from torch_geometric.nn import GCNConv, BatchNorm
except ImportError:
    raise ImportError("torch_geometric is required. Install it via `pip install torch_geometric==2.5.3`.")

from ANS_OCD.config import DEVICE


class MultiLabelGCN(nn.Module):
    """
    Graph Convolutional Network for multi-label node classification.

    The final layer outputs logits of shape [num_nodes, num_communities].
    During training use `torch.nn.BCEWithLogitsLoss`.
    During inference apply `torch.sigmoid` to obtain independent membership
    probabilities per community.

    Parameters
    ----------
    in_channels : int
        Dimension of input node features.
    hidden_channels : int
        Dimension of hidden GCN layers.
    num_communities : int
        Number of overlapping communities (output classes).
    num_layers : int, optional
        Number of GCN convolution layers (default: 2).
    dropout : float, optional
        Dropout probability applied between layers (default: 0.5).
    use_bn : bool, optional
        Whether to apply batch normalization between conv layers (default: True).
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_communities: int,
                 num_layers: int = 2, dropout: float = 0.5, use_bn: bool = True):
        super().__init__()
        assert num_layers >= 1, "num_layers must be at least 1"
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_bn = use_bn

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_bn else None

        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        if use_bn:
            self.bns.append(BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            if use_bn:
                self.bns.append(BatchNorm(hidden_channels))

        # Last layer: project to num_communities without activation.
        # Sigmoid is applied externally during inference/loss to allow multi-label outputs.
        self.convs.append(GCNConv(hidden_channels, num_communities))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape [num_nodes, in_channels]
            Node feature matrix.
        edge_index : torch.Tensor, shape [2, num_edges]
            Graph edge indices.

        Returns
        -------
        logits : torch.Tensor, shape [num_nodes, num_communities]
            Unnormalized scores for each community membership.
        """
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def predict_proba(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Predict independent membership probabilities.

        Returns
        -------
        probs : torch.Tensor, shape [num_nodes, num_communities]
            Values in [0, 1] representing community membership probabilities.
        """
        logits = self.forward(x, edge_index)
        return torch.sigmoid(logits)


class MultiLabelSBERTGCN(nn.Module):
    """
    Convenience wrapper that combines an optional SBERT-like projection with
    the MultiLabelGCN. Useful when node features are raw text embeddings and
    may benefit from an additional linear projection before the GCN.
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_communities: int,
                 num_layers: int = 2, dropout: float = 0.5, use_bn: bool = True,
                 project_in: bool = False):
        super().__init__()
        self.project_in = project_in
        if project_in:
            self.projector = nn.Linear(in_channels, hidden_channels)
            gcn_in = hidden_channels
        else:
            gcn_in = in_channels
        self.gcn = MultiLabelGCN(gcn_in, hidden_channels, num_communities,
                                 num_layers, dropout, use_bn)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.project_in:
            x = self.projector(x)
        return self.gcn(x, edge_index)

    def predict_proba(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x, edge_index)
        return torch.sigmoid(logits)


def multilabel_gnn_loss(logits: torch.Tensor, y: torch.Tensor,
                        pos_weight: Optional[torch.Tensor] = None,
                        reduction: str = "mean") -> torch.Tensor:
    """
    Compute multi-label classification loss using BCEWithLogitsLoss.

    Parameters
    ----------
    logits : torch.Tensor, shape [num_nodes, num_communities]
        GNN output logits.
    y : torch.Tensor, shape [num_nodes, num_communities]
        Ground-truth multi-label binary matrix (float32).
    pos_weight : torch.Tensor, optional
        Weight for positive class to mitigate class imbalance.
    reduction : str
        PyTorch reduction mode.

    Returns
    -------
    loss : torch.Tensor
        Scalar loss value.
    """
    return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction=reduction)


if __name__ == "__main__":
    # Self-test: create a small synthetic graph and verify output shapes.
    from torch_geometric.data import Data

    num_nodes = 5
    num_features = 8
    num_communities = 3
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 1, 2, 3, 4, 0],
                               [1, 2, 3, 4, 0, 0, 1, 2, 3, 4]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)

    model = MultiLabelGCN(num_features, 16, num_communities, num_layers=2, dropout=0.5)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = model.predict_proba(data.x, data.edge_index)
    print("Logits shape:", logits.shape)
    print("Probabilities shape:", probs.shape)
    print("Probs:\n", probs)

    # Verify loss with synthetic multi-label targets.
    y = torch.zeros(num_nodes, num_communities)
    y[0, [0, 1]] = 1.0
    y[1, 1] = 1.0
    y[2, [0, 2]] = 1.0
    model.train()
    logits = model(data.x, data.edge_index)
    loss = multilabel_gnn_loss(logits, y)
    print("BCE loss:", loss.item())
