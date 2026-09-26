"""
abCAN-v2 — Module E: Antisymmetric Difference

Computes the antisymmetric difference representation between the mutant
and wild-type branches to strictly obey the thermodynamic property:
    ΔΔG(A→B) = -ΔΔG(B→A)

Input:  Wild-type features, Mutant features
Output: Antisymmetric representation (Mut - WT)
"""

import torch
import torch.nn as nn
from typing import Optional


class AntisymmetricDifference(nn.Module):
    """
    Module E: Antisymmetric Difference
    
    Ensures that the representation is strictly antisymmetric with respect
    to swapping the wild-type and mutant.
    
    Architecture (from diagram):
      - MLP(WT, Mutant) and MLP(Mutant, WT)
      - Difference: Mut_features - WT_features
      
    For a strict antisymmetric property:
    f(x, y) = g(x, y) - g(y, x)
    
    If g is the identity, then f(x, y) = x - y.
    If we use MLPs, we apply the same MLP to both sides:
    f(wt, mut) = MLP(mut) - MLP(wt)
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        mode: str = "subtract",  # "subtract", "mlp_subtract"
    ):
        super().__init__()
        
        self.mode = mode
        
        if self.mode == "mlp_subtract":
            # Shared MLP applied to both streams before subtraction
            self.shared_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.shared_mlp = nn.Identity()
            
    def forward(
        self,
        wt_features: torch.Tensor,
        mut_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            wt_features: (B, N, D) wild-type representations.
            mut_features: (B, N, D) mutant representations.
            
        Returns:
            (B, N, D) antisymmetric difference features.
        """
        # Apply shared transformation (identity by default)
        wt_transformed = self.shared_mlp(wt_features)
        mut_transformed = self.shared_mlp(mut_features)
        
        # Antisymmetric difference: Mut - WT
        # This guarantees that swapping WT and Mut yields the negative of the representation.
        diff_representation = mut_transformed - wt_transformed
        
        return diff_representation
