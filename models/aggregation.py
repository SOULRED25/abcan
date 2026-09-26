"""
abCAN-v2 — Module F: Mutation-Aware Weighted Aggregation

Aggregates residue-level representations into a single graph-level
representation, focusing heavily on the mutated residues and their spatial neighbors.

Input:  Antisymmetric difference representations (B, N, D), mutation mask (B, N)
Output: Graph-level vector (B, D)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MutationAwareWeightedAggregation(nn.Module):
    """
    Module F: Mutation-Aware Weighted Aggregation
    
    NEW CONTRIBUTION as highlighted in the diagram.
    
    Rather than simple mean or max pooling, this module learns to aggregate
    residue features by attending specifically to mutated sites and their context.
    
    Architecture:
      - Query vector: Learned representation of the mutations
      - Keys/Values: All residue representations
      - Attention pooling to yield a single fixed-length vector
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        attention_weighted: bool = True,
    ):
        super().__init__()
        
        self.attention_weighted = attention_weighted
        
        if attention_weighted:
            # Linear layers for attention-based pooling
            self.q_proj = nn.Linear(hidden_dim, hidden_dim)
            self.k_proj = nn.Linear(hidden_dim, hidden_dim)
            self.v_proj = nn.Linear(hidden_dim, hidden_dim)
            
            # Additional MLP to process mutated residues to form the query
            self.mut_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            
            self.num_heads = num_heads
            self.head_dim = hidden_dim // num_heads
            self.scale = self.head_dim ** -0.5
            
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)
            self.layer_norm = nn.LayerNorm(hidden_dim)
            
    def forward(
        self,
        x: torch.Tensor,
        mutation_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) per-residue antisymmetric representations.
            mutation_mask: (B, N) boolean mask indicating mutated residues.
            padding_mask: (B, N) boolean mask for valid residues (True = valid).
            
        Returns:
            (B, D) aggregated representation.
        """
        B, N, D = x.shape
        
        if not self.attention_weighted:
            # Simple fallback: mean pooling over mutated residues
            # (or all residues if no mutations are flagged)
            out = torch.zeros(B, D, device=x.device)
            for i in range(B):
                mut_idx = torch.where(mutation_mask[i])[0]
                if len(mut_idx) > 0:
                    out[i] = x[i, mut_idx].mean(dim=0)
                elif padding_mask is not None:
                    valid_idx = torch.where(padding_mask[i])[0]
                    if len(valid_idx) > 0:
                        out[i] = x[i, valid_idx].mean(dim=0)
                else:
                    out[i] = x[i].mean(dim=0)
            return out
        
        # --- Attention-weighted pooling ---
        
        # 1. Form the query by pooling only the mutated residues
        queries = torch.zeros(B, 1, D, device=x.device)
        for i in range(B):
            mut_idx = torch.where(mutation_mask[i])[0]
            if len(mut_idx) > 0:
                # Mean of mutated residues as context
                mut_context = x[i, mut_idx].mean(dim=0, keepdim=True)
            else:
                # Fallback to mean of all valid residues if mask is empty
                if padding_mask is not None:
                    valid_idx = torch.where(padding_mask[i])[0]
                    if len(valid_idx) > 0:
                        mut_context = x[i, valid_idx].mean(dim=0, keepdim=True)
                    else:
                        mut_context = x[i].mean(dim=0, keepdim=True)
                else:
                    mut_context = x[i].mean(dim=0, keepdim=True)
            queries[i] = mut_context
            
        queries = self.mut_mlp(queries)  # (B, 1, D)
        
        # 2. Multi-head attention pooling
        Q = self.q_proj(queries).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, d)
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)       # (B, H, N, d)
        V = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)       # (B, H, N, d)
        
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, 1, N)
        
        if padding_mask is not None:
            # padding_mask is (B, N), True means valid
            # we want to fill False with -inf
            # Expand to (B, 1, 1, N)
            mask_expanded = padding_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask_expanded, float("-inf"))
            
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, V)  # (B, H, 1, d)
        out = out.transpose(1, 2).contiguous().view(B, D)  # (B, D)
        
        out = self.out_proj(out)
        out = self.layer_norm(out + queries.squeeze(1))  # Residual connection with query
        
        return out
