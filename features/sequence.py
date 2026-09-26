"""
abCAN-v2 — Stage 3B: Sequence Feature Extraction

Extracts sequence-level features:
  - Wild-type / Mutant sequence encoding
  - Pretrained ESM-2 embeddings (esm2_t33_650M_UR50D)
  - 8th-dimensional projected embedding representation
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# One-Hot Encoding (Fallback)
# ─────────────────────────────────────────────

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
NUM_AAS = len(AMINO_ACIDS)


def one_hot_encode(sequence: str) -> torch.Tensor:
    """
    One-hot encode a protein sequence.
    
    Args:
        sequence: Amino acid sequence string.
        
    Returns:
        (L, 20) one-hot tensor.
    """
    L = len(sequence)
    encoding = torch.zeros(L, NUM_AAS, dtype=torch.float32)
    
    for i, aa in enumerate(sequence):
        if aa in AA_TO_IDX:
            encoding[i, AA_TO_IDX[aa]] = 1.0
        # Unknown AAs get zero vector
    
    return encoding


def tokenize_sequence(sequence: str) -> torch.Tensor:
    """Convert sequence to integer tokens."""
    tokens = torch.zeros(len(sequence), dtype=torch.long)
    for i, aa in enumerate(sequence):
        tokens[i] = AA_TO_IDX.get(aa, NUM_AAS)  # Unknown → NUM_AAS
    return tokens


# ─────────────────────────────────────────────
# ESM-2 Embedding Extractor
# ─────────────────────────────────────────────

class ESM2Embedder:
    """
    Extracts residue-level embeddings using pretrained ESM-2.
    
    Default model: esm2_t33_650M_UR50D (1280-dim embeddings).
    Output is projected to a lower dimension for the model.
    
    From architecture: 8th-dimensional (1280-dim) embedding.
    """
    
    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        device: str = "cpu",
        max_length: int = 512,
    ):
        """
        Args:
            model_name: ESM-2 model identifier.
            device: Device to run on.
            max_length: Maximum sequence length.
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        
        self._model = None
        self._alphabet = None
        self._batch_converter = None
        self._repr_layer = 33  # Last layer for esm2_t33
    
    def _load_model(self):
        """Lazy-load ESM-2 model."""
        if self._model is not None:
            return
        
        try:
            import esm
            
            logger.info(f"Loading ESM-2 model: {self.model_name}")
            self._model, self._alphabet = esm.pretrained.load_model_and_alphabet(
                self.model_name
            )
            self._batch_converter = self._alphabet.get_batch_converter()
            self._model = self._model.to(self.device)
            self._model.eval()
            
            # Determine representation layer
            if "t33" in self.model_name:
                self._repr_layer = 33
            elif "t30" in self.model_name:
                self._repr_layer = 30
            elif "t12" in self.model_name:
                self._repr_layer = 12
            elif "t6" in self.model_name:
                self._repr_layer = 6
            
            logger.info(
                f"ESM-2 loaded: {self.model_name}, "
                f"repr_layer={self._repr_layer}, device={self.device}"
            )
            
        except ImportError:
            raise ImportError(
                "fair-esm is required for ESM-2 embeddings. "
                "Install with: pip install fair-esm"
            )
    
    @torch.no_grad()
    def embed(self, sequence: str, label: str = "protein") -> torch.Tensor:
        """
        Get per-residue ESM-2 embeddings for a single sequence.
        
        Args:
            sequence: Amino acid sequence.
            label: Sequence label.
            
        Returns:
            (L, 1280) embedding tensor.
        """
        self._load_model()
        
        # Truncate if needed
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
            logger.warning(f"Sequence truncated to {self.max_length} residues")
        
        # Prepare batch
        data = [(label, sequence)]
        batch_labels, batch_strs, batch_tokens = self._batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        
        # Forward pass
        results = self._model(
            batch_tokens,
            repr_layers=[self._repr_layer],
            return_contacts=False,
        )
        
        # Extract representations (remove BOS/EOS tokens)
        representations = results["representations"][self._repr_layer]
        # Shape: (1, L+2, 1280) → (L, 1280) (remove BOS at idx 0, EOS at end)
        embeddings = representations[0, 1:len(sequence) + 1, :]
        
        return embeddings.cpu()
    
    @torch.no_grad()
    def embed_batch(
        self,
        sequences: List[Tuple[str, str]],
    ) -> List[torch.Tensor]:
        """
        Batch embedding of multiple sequences.
        
        Args:
            sequences: List of (label, sequence) tuples.
            
        Returns:
            List of (L_i, 1280) tensors.
        """
        self._load_model()
        
        # Truncate sequences
        truncated = [
            (label, seq[:self.max_length]) for label, seq in sequences
        ]
        
        batch_labels, batch_strs, batch_tokens = self._batch_converter(truncated)
        batch_tokens = batch_tokens.to(self.device)
        
        results = self._model(
            batch_tokens,
            repr_layers=[self._repr_layer],
            return_contacts=False,
        )
        
        representations = results["representations"][self._repr_layer]
        
        embeddings = []
        for i, (_, seq) in enumerate(truncated):
            emb = representations[i, 1:len(seq) + 1, :]
            embeddings.append(emb.cpu())
        
        return embeddings

    @torch.no_grad()
    def compute_zero_shot_llr(self, wt_seq: str, mut_seq: str, mutation_idx: int) -> float:
        """
        Stream 2: Zero-Shot ΔLLR Prior Calculator.
        Computes log p(mut) - log p(wt) using ESM2 masked marginal probability.
        """
        self._load_model()
        
        # Mask the mutation site
        masked_seq = list(wt_seq)
        if mutation_idx >= len(masked_seq):
            return 0.0
            
        masked_seq[mutation_idx] = "<mask>"
        masked_seq_str = "".join(masked_seq)
        
        data = [("masked", masked_seq_str)]
        _, _, batch_tokens = self._batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        
        # Get logits
        results = self._model(batch_tokens, repr_layers=[self._repr_layer], return_contacts=False)
        logits = results["logits"]  # (1, L+2, vocab_size)
        
        import torch.nn.functional as F
        # Log probabilities at the mutation site (+1 for BOS token)
        log_probs = F.log_softmax(logits[0, mutation_idx + 1], dim=-1)
        
        wt_token = self._alphabet.get_idx(wt_seq[mutation_idx])
        mut_token = self._alphabet.get_idx(mut_seq[mutation_idx])
        
        llr = log_probs[mut_token].item() - log_probs[wt_token].item()
        
        return float(llr)


# ─────────────────────────────────────────────
# Embedding Projection
# ─────────────────────────────────────────────

class EmbeddingProjection(nn.Module):
    """
    Projects ESM-2 embeddings (1280-dim) to model hidden dimension.
    
    Also supports learned positional information injection.
    """
    
    def __init__(
        self,
        input_dim: int = 1280,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, 1280) ESM-2 embeddings.
            
        Returns:
            (B, L, output_dim) projected embeddings.
        """
        return self.projection(x)


# ─────────────────────────────────────────────
# Main Sequence Feature Extractor
# ─────────────────────────────────────────────

class SequenceFeatureExtractor:
    """
    Main Sequence Branch feature extractor.
    
    Extracts:
      - Wild-type ESM-2 embeddings
      - Mutant ESM-2 embeddings
      - One-hot encoded sequences (fallback)
      
    Output feeds into MPNN Sequence Encoder v2 (Stage 4B).
    """
    
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        seq_cfg = config.get("features", {}).get("sequence", {})
        
        self.use_esm2 = seq_cfg.get("use_pretrained", True)
        self.esm2_model = seq_cfg.get("esm2_model", "esm2_t33_650M_UR50D")
        self.embedding_dim = seq_cfg.get("embedding_dim", 1280)
        self.projection_dim = seq_cfg.get("projection_dim", 128)
        
        self._embedder = None
        
        # Device
        device = config.get("project", {}).get("device", "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
    
    @property
    def embedder(self) -> ESM2Embedder:
        """Lazy-load ESM-2 embedder."""
        if self._embedder is None:
            self._embedder = ESM2Embedder(
                model_name=self.esm2_model,
                device=self.device,
            )
        return self._embedder
    
    def extract(
        self,
        wt_sequence: str,
        mut_sequence: str,
        use_esm2: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract sequence features for wild-type and mutant.
        
        Args:
            wt_sequence: Wild-type amino acid sequence.
            mut_sequence: Mutant amino acid sequence.
            use_esm2: Override ESM-2 usage. None uses config default.
            
        Returns:
            Dict of tensors:
              - wt_embedding: (L, D) wild-type embeddings
              - mut_embedding: (L, D) mutant embeddings
              - wt_onehot: (L, 20) wild-type one-hot
              - mut_onehot: (L, 20) mutant one-hot
              - mutation_mask: (L,) boolean mask of mutated positions
        """
        use_esm = use_esm2 if use_esm2 is not None else self.use_esm2
        
        # One-hot encodings (always computed)
        wt_onehot = one_hot_encode(wt_sequence)
        mut_onehot = one_hot_encode(mut_sequence)
        
        # Mutation mask
        mutation_mask = torch.tensor(
            [wt_sequence[i] != mut_sequence[i] for i in range(len(wt_sequence))],
            dtype=torch.bool,
        )
        
        result = {
            "wt_onehot": wt_onehot,
            "mut_onehot": mut_onehot,
            "mutation_mask": mutation_mask,
            "sequence_length": len(wt_sequence),
        }
        
        # ESM-2 embeddings
        if use_esm:
            try:
                wt_emb = self.embedder.embed(wt_sequence, label="wt")
                mut_emb = self.embedder.embed(mut_sequence, label="mut")
                
                result["wt_embedding"] = wt_emb     # (L, 1280)
                result["mut_embedding"] = mut_emb    # (L, 1280)
                
                # Compute Zero-Shot LLR Prior
                mut_idx = torch.where(mutation_mask)[0]
                llr = 0.0
                if len(mut_idx) > 0:
                    # Sum LLR for multiple mutations
                    for idx in mut_idx.tolist():
                        llr += self.embedder.compute_zero_shot_llr(wt_sequence, mut_sequence, idx)
                result["zero_shot_llr"] = torch.tensor([llr], dtype=torch.float32)
                
                logger.debug(
                    f"ESM-2 embeddings: wt={wt_emb.shape}, mut={mut_emb.shape}, llr={llr:.3f}"
                )
                
            except Exception as e:
                logger.warning(f"ESM-2 embedding failed: {e}. Using one-hot fallback.")
                result["wt_embedding"] = wt_onehot
                result["mut_embedding"] = mut_onehot
                result["zero_shot_llr"] = torch.tensor([0.0], dtype=torch.float32)
        else:
            result["wt_embedding"] = wt_onehot
            result["mut_embedding"] = mut_onehot
            result["zero_shot_llr"] = torch.tensor([0.0], dtype=torch.float32)
        
        return result
