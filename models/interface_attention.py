"""
abCAN-v2 — Module C: Interface-Aware Dual-Stream Attention

Fuses structural and sequence representations with:
  - Multi-Head Self-Attention
  - Wild-Mutant Cross-Attention
  - Interface-Zone Gating (focus on binding interface residues)

Input:  Structural + Sequence embeddings
Output: Interface-aware fused representations
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention with pre-norm."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) input.
            mask: (B, L) boolean mask.
        Returns:
            (B, L, D) self-attended output.
        """
        B, L, D = x.shape
        residual = x
        x = self.layer_norm(x)
        
        qkv = self.qkv_proj(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, d)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        
        return out + residual


class InterfaceZoneGating(nn.Module):
    """
    Interface-Zone Gating mechanism.
    
    Learns to upweight residues at the antibody-antigen interface
    and downweight non-interface residues.
    
    gate = σ(W_g · [h_i || interface_flag_i] + b_g)
    h_i' = gate * h_i
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # Gate computation: takes features + interface indicator
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        
        # Learnable interface embedding
        self.interface_embedding = nn.Embedding(2, hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        interface_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) residue features.
            interface_mask: (B, N) boolean mask — True for interface residues.
            
        Returns:
            (B, N, D) gated features (interface residues emphasized).
        """
        # Interface indicator as float
        interface_flag = interface_mask.float().unsqueeze(-1)  # (B, N, 1)
        
        # Compute gate
        gate_input = torch.cat([x, interface_flag], dim=-1)
        gate = self.gate_net(gate_input)  # (B, N, D)
        
        # Add interface embedding
        interface_idx = interface_mask.long()  # (B, N)
        interface_emb = self.interface_embedding(interface_idx)  # (B, N, D)
        
        # Apply gating + interface embedding
        gated = gate * x + interface_emb
        
        return gated


class FeedForward(nn.Module):
    """Standard FFN with GELU activation."""
    
    def __init__(self, hidden_dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * expansion, hidden_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class InterfaceAwareDualStreamAttention(nn.Module):
    """
    Module C: Interface-Aware Dual-Stream Attention
    
    Fuses structural and sequence information with special
    attention to the antibody-antigen interface.
    
    Architecture (from diagram):
      - Multi-Head Attention (self-attention on fused features)
      - Wild-Mutant Cross-Attention
      - Interface-Zone Gating
      
    Two streams:
      Stream 1: Structural embeddings
      Stream 2: Sequence embeddings (WT and Mut)
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        cross_attention: bool = True,
        interface_zone_gating: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.use_cross_attention = cross_attention
        self.use_interface_gating = interface_zone_gating
        
        # Fusion projection: struct + seq → hidden
        self.struct_proj = nn.Linear(hidden_dim, hidden_dim)
        self.seq_proj = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        
        # Self-attention layers
        self.self_attn_layers = nn.ModuleList([
            MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Feed-forward layers
        self.ffn_layers = nn.ModuleList([
            FeedForward(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Cross-attention: WT stream ↔ Mut stream
        if cross_attention:
            from models.mpnn_sequence import CrossAttention
            self.wt_mut_cross_attn = CrossAttention(hidden_dim, num_heads, dropout)
            self.mut_wt_cross_attn = CrossAttention(hidden_dim, num_heads, dropout)
        
        # Interface-zone gating
        if interface_zone_gating:
            self.interface_gate = InterfaceZoneGating(hidden_dim)
        
        # Output norm
        self.output_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        struct_features: torch.Tensor,
        wt_seq_features: torch.Tensor,
        mut_seq_features: torch.Tensor,
        interface_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            struct_features: (B, N, D) structural embeddings from Module A.
            wt_seq_features: (B, L, D) WT sequence embeddings from Module B.
            mut_seq_features: (B, L, D) Mut sequence embeddings from Module B.
            interface_mask: (B, N) boolean mask of interface residues.
            padding_mask: (B, N) boolean mask for valid residues.
            
        Returns:
            (wt_fused, mut_fused): each (B, N, D)
        """
        # Fuse structural + sequence for each stream
        # Note: struct is shared, seq differs between WT/Mut
        struct_proj = self.struct_proj(struct_features)
        
        wt_fused = self.fusion_norm(struct_proj + self.seq_proj(wt_seq_features))
        mut_fused = self.fusion_norm(struct_proj + self.seq_proj(mut_seq_features))
        
        # Self-attention + FFN layers
        for self_attn, ffn in zip(self.self_attn_layers, self.ffn_layers):
            wt_fused = self_attn(wt_fused, padding_mask)
            wt_fused = ffn(wt_fused)
            
            mut_fused = self_attn(mut_fused, padding_mask)
            mut_fused = ffn(mut_fused)
        
        # Cross-attention between WT and Mut streams
        if self.use_cross_attention:
            wt_cross = self.wt_mut_cross_attn(wt_fused, mut_fused, padding_mask)
            mut_cross = self.mut_wt_cross_attn(mut_fused, wt_fused, padding_mask)
            wt_fused = wt_cross
            mut_fused = mut_cross
        
        # Interface-zone gating
        if self.use_interface_gating and interface_mask is not None:
            wt_fused = self.interface_gate(wt_fused, interface_mask)
            mut_fused = self.interface_gate(mut_fused, interface_mask)
        
        # Output normalization
        wt_fused = self.output_norm(wt_fused)
        mut_fused = self.output_norm(mut_fused)
        
        return wt_fused, mut_fused
