"""
Bi-Directional Cross-Attention Engine

Injects additive attention biases:
  + lambda_iface for interface residues (<= 5.0 Å)
  + lambda_mut for mutation sites

Cross-Attention: Structure <---> Sequence Context
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class BiDirectionalCrossAttention(nn.Module):
    """
    Engine to fuse Structure Stream (h_struct) and Sequence Stream (h_seq).
    
    Employs additive attention biases to heavily upweight the attention logits
    for interface residues and the specifically mutated residues.
    """
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        lambda_iface: float = 2.0,
        lambda_mut: float = 5.0
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.lambda_iface = lambda_iface
        self.lambda_mut = lambda_mut
        
        # Structure -> Sequence Attention
        self.q_struct = nn.Linear(hidden_dim, hidden_dim)
        self.k_seq = nn.Linear(hidden_dim, hidden_dim)
        self.v_seq = nn.Linear(hidden_dim, hidden_dim)
        self.out_struct = nn.Linear(hidden_dim, hidden_dim)
        
        # Sequence -> Structure Attention
        self.q_seq = nn.Linear(hidden_dim, hidden_dim)
        self.k_struct = nn.Linear(hidden_dim, hidden_dim)
        self.v_struct = nn.Linear(hidden_dim, hidden_dim)
        self.out_seq = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.norm_struct = nn.LayerNorm(hidden_dim)
        self.norm_seq = nn.LayerNorm(hidden_dim)
        
    def _compute_additive_bias(
        self, 
        interface_mask: torch.Tensor, 
        mutation_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the additive bias matrix.
        Args:
            interface_mask: (B, N) boolean mask for interface residues (<= 5.0A).
            mutation_mask: (B, N) boolean mask for mutated residues.
        Returns:
            (B, 1, 1, N) bias tensor to be added to attention logits.
        """
        B, N = interface_mask.shape
        bias = torch.zeros((B, N), device=interface_mask.device)
        
        bias = bias.masked_fill(interface_mask, self.lambda_iface)
        bias = bias.masked_fill(mutation_mask, self.lambda_mut)
        
        # Expand for multi-head attention: (B, Heads, L_query, L_key)
        # Here we apply bias over the key dimension (N).
        bias = bias.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
        return bias

    def forward(
        self,
        h_struct: torch.Tensor,
        h_seq: torch.Tensor,
        interface_mask: torch.Tensor,
        mutation_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            h_struct: (B, N, D) structural embeddings from SE(3)-EGNN.
            h_seq: (B, N, D) sequence embeddings from ESM2.
            interface_mask: (B, N) boolean mask.
            mutation_mask: (B, N) boolean mask.
            padding_mask: (B, N) optional valid residue mask.
            
        Returns:
            h_fused: (B, N, D) fused representation.
        """
        B, N, D = h_struct.shape
        
        # Compute Additive Bias
        additive_bias = self._compute_additive_bias(interface_mask, mutation_mask)
        
        # Pad mask for attention logits
        pad_bias = 0.0
        if padding_mask is not None:
            pad_bias = torch.zeros((B, 1, 1, N), device=h_struct.device)
            pad_bias = pad_bias.masked_fill(~padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
            
        total_bias = additive_bias + pad_bias
        
        # ==========================================
        # 1. Structure attends to Sequence Context
        # ==========================================
        Q_struct = self.q_struct(h_struct).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K_seq = self.k_seq(h_seq).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V_seq = self.v_seq(h_seq).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_s2q = (torch.matmul(Q_struct, K_seq.transpose(-2, -1)) / self.scale) + total_bias
        attn_s2q = self.dropout(F.softmax(attn_s2q, dim=-1))
        
        out_s2q = torch.matmul(attn_s2q, V_seq).transpose(1, 2).contiguous().view(B, N, D)
        h_struct_new = self.norm_struct(h_struct + self.out_struct(out_s2q))
        
        # ==========================================
        # 2. Sequence attends to Structure Context
        # ==========================================
        Q_seq = self.q_seq(h_seq).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K_struct = self.k_struct(h_struct).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V_struct = self.v_struct(h_struct).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_q2s = (torch.matmul(Q_seq, K_struct.transpose(-2, -1)) / self.scale) + total_bias
        attn_q2s = self.dropout(F.softmax(attn_q2s, dim=-1))
        
        out_q2s = torch.matmul(attn_q2s, V_struct).transpose(1, 2).contiguous().view(B, N, D)
        h_seq_new = self.norm_seq(h_seq + self.out_seq(out_q2s))
        
        # ==========================================
        # Fusion
        # ==========================================
        h_fused = h_struct_new + h_seq_new
        return h_fused
