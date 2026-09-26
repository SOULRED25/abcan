"""
abCAN-v2 — Module A: MPNN Structural Encoder

3-layer Message Passing Neural Network on residue KNN graphs.
Each layer: message computation → aggregation → update with residual skip.

Input:  Per-residue structural features + KNN graph
Output: Structurally-aware residue embeddings (N, hidden_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import add_self_loops
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class MPNNLayer(nn.Module):
    """
    Single MPNN layer with edge-conditioned message passing.
    
    Messages are computed as:
      m_{ij} = MLP([h_i || h_j || e_{ij}])
    
    Update:
      h_i' = GRU(h_i, agg(m_{ij})) + h_i  (residual)
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Message MLP
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # GRU for node update
        self.gru = nn.GRUCell(hidden_dim, node_dim)
        
        # Layer norm for output
        self.layer_norm = nn.LayerNorm(node_dim)
        
        # Residual projection (if dimensions differ)
        self.residual_proj = None
        if hidden_dim != node_dim:
            self.residual_proj = nn.Linear(hidden_dim, node_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, node_dim) node features.
            edge_index: (2, E) edge indices.
            edge_attr: (E, edge_dim) edge features.
            
        Returns:
            (N, node_dim) updated node features.
        """
        src, dst = edge_index[0], edge_index[1]
        N = x.shape[0]
        
        # Compute messages
        src_features = x[src]    # (E, node_dim)
        dst_features = x[dst]    # (E, node_dim)
        
        messages = torch.cat([src_features, dst_features, edge_attr], dim=-1)
        messages = self.message_mlp(messages)  # (E, hidden_dim)
        
        # Aggregate messages (sum)
        agg = torch.zeros(N, messages.shape[-1], device=x.device)
        agg.index_add_(0, dst, messages)
        
        # Project if needed
        if self.residual_proj is not None:
            agg = self.residual_proj(agg)
        
        # GRU update
        x_updated = self.gru(agg, x)
        
        # Residual connection + layer norm
        x_updated = self.layer_norm(x_updated + x)
        
        return x_updated


class MPNNStructuralEncoder(nn.Module):
    """
    Module A: MPNN Structural Encoder
    
    3-layer graph convolution with residual skip connections
    operating on the KNN graph of residue Cα coordinates.
    
    Architecture (from diagram):
      - 3-layer Graph Conv
      - Residual Skip connections
      - Input: structural node features
      - Output: (N, hidden_dim) structural embeddings
    """
    
    def __init__(
        self,
        input_dim: int = 32,     # Structural feature dim
        hidden_dim: int = 128,
        edge_dim: int = 1,       # Distance-based edge features
        num_layers: int = 3,
        dropout: float = 0.1,
        residual_skip: bool = True,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.residual_skip = residual_skip
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Edge feature projection
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        # MPNN layers
        self.layers = nn.ModuleList([
            MPNNLayer(
                node_dim=hidden_dim,
                edge_dim=hidden_dim // 2,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: (N, input_dim) structural features.
            edge_index: (2, E) KNN graph edges.
            edge_attr: (E, edge_dim) edge features (distances).
            
        Returns:
            (N, hidden_dim) structural embeddings.
        """
        # Project inputs
        x = self.input_proj(node_features)
        edge_features = self.edge_proj(edge_attr)
        
        # Apply MPNN layers with residual
        for layer in self.layers:
            if self.residual_skip:
                x_residual = x
                x = layer(x, edge_index, edge_features)
                x = x + x_residual
            else:
                x = layer(x, edge_index, edge_features)
        
        # Output projection
        x = self.output_proj(x)
        
        return x
