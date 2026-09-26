"""
abCAN-v2 — Module D: Physicochemical Residual Injection

Injects 6D biochemical prior features into the main representation
through a gated residual connection.

Input:  6-dim biochemical features + main hidden representation
Output: Hidden representation enriched with physicochemical priors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GatedResidualConnection(nn.Module):
    """
    Gated connection for injecting auxiliary features.
    
    gate = σ(W_g · [h || aux] + b_g)
    h' = h + gate * MLP(aux)
    
    The gate learns how much biochemical information to inject
    at each position, allowing the model to adaptively use priors.
    """
    
    def __init__(
        self,
        main_dim: int,
        aux_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        
        # Transform auxiliary features
        self.aux_transform = nn.Sequential(
            nn.Linear(aux_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, main_dim),
        )
        
        # Gate: decides how much of the auxiliary signal to inject
        self.gate = nn.Sequential(
            nn.Linear(main_dim + aux_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, main_dim),
            nn.Sigmoid(),
        )
        
        self.layer_norm = nn.LayerNorm(main_dim)
    
    def forward(
        self,
        main_features: torch.Tensor,
        aux_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            main_features: (B, N, main_dim) main representation.
            aux_features: (B, N, aux_dim) auxiliary biochemical features.
            
        Returns:
            (B, N, main_dim) enriched representation.
        """
        # Transform aux features to main dim
        aux_transformed = self.aux_transform(aux_features)
        
        # Compute gate
        gate_input = torch.cat([main_features, aux_features], dim=-1)
        gate = self.gate(gate_input)
        
        # Gated injection + residual
        enriched = main_features + gate * aux_transformed
        enriched = self.layer_norm(enriched)
        
        return enriched


class PhysicochemicalResidualInjection(nn.Module):
    """
    Module D: Physicochemical Residual Injection
    
    Injects the 6D biochemical prior features into the model's
    hidden representations via a learned, gated residual connection.
    
    Architecture (from diagram):
      - 6-dim Contribution input
      - MLP + Residual Gate
      - Biochemical Prior integration
      - Gated Connection to main stream
      
    The 6D features are:
      [BLOSUM62, polarity, charge, size, hydrophobicity, contact_distance]
    """
    
    def __init__(
        self,
        input_dim: int = 6,       # 6D biochemical features
        hidden_dim: int = 64,
        output_dim: int = 128,    # Must match main representation dim
        residual_gate: bool = True,
    ):
        super().__init__()
        
        self.use_residual_gate = residual_gate
        
        # Biochemical feature encoder
        self.biochem_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Gated residual connection
        if residual_gate:
            self.gated_residual = GatedResidualConnection(
                main_dim=output_dim,
                aux_dim=output_dim,
                hidden_dim=hidden_dim,
            )
        else:
            # Simple addition
            self.proj = nn.Linear(output_dim, output_dim)
    
    def forward(
        self,
        main_features: torch.Tensor,
        biochem_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            main_features: (B, N, output_dim) main hidden representation.
            biochem_features: (B, N, input_dim) 6D biochemical features.
            
        Returns:
            (B, N, output_dim) enriched representation.
        """
        # Encode biochemical features
        biochem_encoded = self.biochem_encoder(biochem_features)  # (B, N, output_dim)
        
        # Inject via gated residual
        if self.use_residual_gate:
            enriched = self.gated_residual(main_features, biochem_encoded)
        else:
            enriched = main_features + self.proj(biochem_encoded)
        
        return enriched
