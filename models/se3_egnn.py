"""
Dual-Stream Feature Encoders: Structure Stream
SE(3)-Equivariant Graph Neural Network (EGNN)

Ensures that the extracted structural features (h_struct) strictly
preserve 3D rotational and translational equivariance.

Reference Architecture: EGNN (Satorras et al., 2021)
"""

import torch
import torch.nn as nn
from typing import Tuple

class EGNNLayer(nn.Module):
    """
    A single SE(3)-Equivariant Graph Neural Network Layer.
    
    Updates node features (invariant) and node coordinates (equivariant).
    """
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        act_fn=nn.SiLU()
    ):
        super().__init__()
        
        # Message network: computes message from node i, node j, squared distance, and edge attr
        # Input dim: 2 * node_dim + 1 (sq_dist) + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + 1 + edge_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn
        )
        
        # Node update network: updates node features using aggregated messages
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, node_dim)
        )
        
        # Coordinate update network: determines how much to shift coordinates along the vector (x_i - x_j)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
    def forward(
        self, 
        h: torch.Tensor, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Node features (N, node_dim)
            x: Node 3D coordinates (N, 3)
            edge_index: Edge connectivity (2, E)
            edge_attr: Edge features (E, edge_dim)
            
        Returns:
            Updated node features (N, node_dim)
            Updated coordinates (N, 3)
        """
        row, col = edge_index
        
        # 1. Compute relative squared distances
        coord_diff = x[row] - x[col]                  # (E, 3)
        sq_dist = torch.sum(coord_diff**2, dim=-1, keepdim=True)  # (E, 1)
        
        # 2. Compute messages
        edge_input = torch.cat([h[row], h[col], sq_dist, edge_attr], dim=-1)
        m_ij = self.edge_mlp(edge_input)              # (E, hidden_dim)
        
        # 3. Coordinate update (Equivariant)
        # Shift proportional to the coord difference
        shift_weight = self.coord_mlp(m_ij)           # (E, 1)
        coord_shift = shift_weight * coord_diff       # (E, 3)
        
        # Aggregate coordinate shifts
        agg_shift = torch.zeros_like(x)
        agg_shift.index_add_(0, row, coord_shift)
        x_new = x + agg_shift
        
        # 4. Node feature update (Invariant)
        m_i = torch.zeros(h.shape[0], m_ij.shape[-1], device=h.device)
        m_i.index_add_(0, row, m_ij)
        
        node_input = torch.cat([h, m_i], dim=-1)
        h_new = h + self.node_mlp(node_input)         # Residual update
        
        return h_new, x_new


class SE3_EGNN(nn.Module):
    """
    Structure Stream: SE(3)-Equivariant GNN.
    
    Maps raw structural features and coordinates to rich, SE(3)-invariant
    node representations (h_struct) for downstream cross-attention.
    """
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4
    ):
        super().__init__()
        
        self.node_embedding = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_in_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            EGNNLayer(
                node_dim=hidden_dim, 
                edge_dim=hidden_dim, 
                hidden_dim=hidden_dim
            ) for _ in range(num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, 
        h: torch.Tensor, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h: Initial node features (N, node_in_dim)
            x: 3D coordinates (N, 3)
            edge_index: Graph connectivity (2, E)
            edge_attr: Initial edge features (E, edge_in_dim)
            
        Returns:
            h_struct: SE(3)-invariant structural embeddings (N, hidden_dim)
        """
        h = self.node_embedding(h)
        edge_attr = self.edge_embedding(edge_attr)
        
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)
            
        h = self.output_norm(h)
        return h
