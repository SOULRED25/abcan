"""
abCAN-v2 — Module B: MPNN Sequence Encoder v2

Encodes wild-type and mutant protein sequences using:
  - Pretrained ESM-2 embeddings
  - Cross-attention between wild-type and mutant representations
  - Message passing for sequence-level interactions

Input:  ESM-2 embeddings for WT and Mutant sequences
Output: Cross-attended sequence embeddings (N, hidden_dim) × 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class CrossAttention(nn.Module):
    """
    Cross-attention between two sequences.
    
    Query from one sequence, Key/Value from the other.
    Used to capture mutation-induced changes in context.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, L_q, D) query sequence.
            key_value: (B, L_kv, D) key/value sequence.
            mask: Optional attention mask.
            
        Returns:
            (B, L_q, D) cross-attended output.
        """
        B, L_q, D = query.shape
        L_kv = key_value.shape[1]
        
        # Project Q, K, V
        Q = self.q_proj(query).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key_value).view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(key_value).view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, L_q, L_kv)
        
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)  # (B, H, L_q, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L_q, D)
        out = self.out_proj(out)
        
        # Residual + LayerNorm
        out = self.layer_norm(out + query)
        
        return out


class SequenceMPNNLayer(nn.Module):
    """
    Sequence-level message passing layer.
    
    Messages flow along the sequence (local window) to capture
    sequential dependencies in protein structure.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        window_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.window_size = window_size
        
        # Message computation
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Update
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) sequence features.
            
        Returns:
            (B, L, D) updated features.
        """
        B, L, D = x.shape
        
        # Pad for windowed message passing
        pad = self.window_size // 2
        x_padded = F.pad(x, (0, 0, pad, pad), mode="constant", value=0)
        
        # Compute messages from local window
        messages = []
        for offset in range(-pad, pad + 1):
            if offset == 0:
                continue
            shifted = x_padded[:, pad + offset:pad + offset + L, :]
            msg = torch.cat([x, shifted], dim=-1)
            msg = self.message_mlp(msg)
            messages.append(msg)
        
        # Aggregate messages (mean)
        agg_message = torch.stack(messages, dim=0).mean(dim=0)  # (B, L, D)
        
        # Update
        updated = torch.cat([x, agg_message], dim=-1)
        updated = self.update_mlp(updated)
        
        # Residual + norm
        return self.layer_norm(updated + x)


class MPNNSequenceEncoderV2(nn.Module):
    """
    Module B: MPNN Sequence Encoder v2
    
    Processes wild-type and mutant ESM-2 embeddings with:
      1. Input projection from ESM-2 dim (1280) to hidden_dim
      2. Sequence-level MPNN for local context
      3. Cross-attention between WT and Mutant representations
    
    Architecture (from diagram):
      - Wild / Mutant inputs
      - Pre-trained ESM-2 Embeddings
      - Cross-Attention between WT and Mut
    """
    
    def __init__(
        self,
        esm2_dim: int = 1280,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_mpnn_layers: int = 2,
        cross_attention: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.cross_attention_enabled = cross_attention
        
        # ESM-2 → hidden_dim projection
        self.input_proj = nn.Sequential(
            nn.Linear(esm2_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Sequence MPNN layers (shared for WT and Mut)
        self.mpnn_layers = nn.ModuleList([
            SequenceMPNNLayer(hidden_dim, dropout=dropout)
            for _ in range(num_mpnn_layers)
        ])
        
        # Cross-attention: WT attends to Mut and vice versa
        if cross_attention:
            self.wt_to_mut_attn = CrossAttention(hidden_dim, num_heads, dropout)
            self.mut_to_wt_attn = CrossAttention(hidden_dim, num_heads, dropout)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
    
    def forward(
        self,
        wt_embedding: torch.Tensor,
        mut_embedding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            wt_embedding: (B, L, esm2_dim) wild-type ESM-2 embeddings.
            mut_embedding: (B, L, esm2_dim) mutant ESM-2 embeddings.
            mask: (B, L) optional padding mask.
            
        Returns:
            (wt_encoded, mut_encoded): each (B, L, hidden_dim)
        """
        # Project ESM-2 embeddings
        wt = self.input_proj(wt_embedding)    # (B, L, hidden_dim)
        mut = self.input_proj(mut_embedding)   # (B, L, hidden_dim)
        
        # Apply sequence MPNN
        for mpnn in self.mpnn_layers:
            wt = mpnn(wt)
            mut = mpnn(mut)
        
        # Cross-attention
        if self.cross_attention_enabled:
            wt_cross = self.wt_to_mut_attn(wt, mut, mask)
            mut_cross = self.mut_to_wt_attn(mut, wt, mask)
            wt = wt_cross
            mut = mut_cross
        
        # Output projection
        wt = self.output_proj(wt)
        mut = self.output_proj(mut)
        
        return wt, mut
