"""
abCAN-v2 — Stage 3C: Biochemical Prior Feature Extraction

Extracts the 6D biochemical feature vector:
  1. BLOSUM62 substitution score
  2. Polarity change
  3. Charge change
  4. Size (molecular weight) change
  5. Hydrophobicity change
  6. Contact distance at mutation site
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# BLOSUM62 Substitution Matrix
# ─────────────────────────────────────────────

# Standard BLOSUM62 matrix (symmetric)
BLOSUM62 = {
    ("A", "A"):  4, ("A", "R"): -1, ("A", "N"): -2, ("A", "D"): -2, ("A", "C"):  0,
    ("A", "Q"): -1, ("A", "E"): -1, ("A", "G"):  0, ("A", "H"): -2, ("A", "I"): -1,
    ("A", "L"): -1, ("A", "K"): -1, ("A", "M"): -1, ("A", "F"): -2, ("A", "P"): -1,
    ("A", "S"):  1, ("A", "T"):  0, ("A", "W"): -3, ("A", "Y"): -2, ("A", "V"):  0,
    ("R", "R"):  5, ("R", "N"):  0, ("R", "D"): -2, ("R", "C"): -3, ("R", "Q"):  1,
    ("R", "E"):  0, ("R", "G"): -2, ("R", "H"):  0, ("R", "I"): -3, ("R", "L"): -2,
    ("R", "K"):  2, ("R", "M"): -1, ("R", "F"): -3, ("R", "P"): -2, ("R", "S"): -1,
    ("R", "T"): -1, ("R", "W"): -3, ("R", "Y"): -2, ("R", "V"): -3,
    ("N", "N"):  6, ("N", "D"):  1, ("N", "C"): -3, ("N", "Q"):  0, ("N", "E"):  0,
    ("N", "G"):  0, ("N", "H"):  1, ("N", "I"): -3, ("N", "L"): -3, ("N", "K"):  0,
    ("N", "M"): -2, ("N", "F"): -3, ("N", "P"): -2, ("N", "S"):  1, ("N", "T"):  0,
    ("N", "W"): -4, ("N", "Y"): -2, ("N", "V"): -3,
    ("D", "D"):  6, ("D", "C"): -3, ("D", "Q"):  0, ("D", "E"):  2, ("D", "G"): -1,
    ("D", "H"): -1, ("D", "I"): -3, ("D", "L"): -4, ("D", "K"): -1, ("D", "M"): -3,
    ("D", "F"): -3, ("D", "P"): -1, ("D", "S"):  0, ("D", "T"): -1, ("D", "W"): -4,
    ("D", "Y"): -3, ("D", "V"): -3,
    ("C", "C"):  9, ("C", "Q"): -3, ("C", "E"): -4, ("C", "G"): -3, ("C", "H"): -3,
    ("C", "I"): -1, ("C", "L"): -1, ("C", "K"): -3, ("C", "M"): -1, ("C", "F"): -2,
    ("C", "P"): -3, ("C", "S"): -1, ("C", "T"): -1, ("C", "W"): -2, ("C", "Y"): -2,
    ("C", "V"): -1,
    ("Q", "Q"):  5, ("Q", "E"):  2, ("Q", "G"): -2, ("Q", "H"):  0, ("Q", "I"): -3,
    ("Q", "L"): -2, ("Q", "K"):  1, ("Q", "M"):  0, ("Q", "F"): -3, ("Q", "P"): -1,
    ("Q", "S"):  0, ("Q", "T"): -1, ("Q", "W"): -2, ("Q", "Y"): -1, ("Q", "V"): -2,
    ("E", "E"):  5, ("E", "G"): -2, ("E", "H"):  0, ("E", "I"): -3, ("E", "L"): -3,
    ("E", "K"):  1, ("E", "M"): -2, ("E", "F"): -3, ("E", "P"): -1, ("E", "S"):  0,
    ("E", "T"): -1, ("E", "W"): -3, ("E", "Y"): -2, ("E", "V"): -2,
    ("G", "G"):  6, ("G", "H"): -2, ("G", "I"): -4, ("G", "L"): -4, ("G", "K"): -2,
    ("G", "M"): -3, ("G", "F"): -3, ("G", "P"): -2, ("G", "S"):  0, ("G", "T"): -2,
    ("G", "W"): -2, ("G", "Y"): -3, ("G", "V"): -3,
    ("H", "H"):  8, ("H", "I"): -3, ("H", "L"): -3, ("H", "K"): -1, ("H", "M"): -2,
    ("H", "F"): -1, ("H", "P"): -2, ("H", "S"): -1, ("H", "T"): -2, ("H", "W"): -2,
    ("H", "Y"):  2, ("H", "V"): -3,
    ("I", "I"):  4, ("I", "L"):  2, ("I", "K"): -3, ("I", "M"):  1, ("I", "F"):  0,
    ("I", "P"): -3, ("I", "S"): -2, ("I", "T"): -1, ("I", "W"): -3, ("I", "Y"): -1,
    ("I", "V"):  3,
    ("L", "L"):  4, ("L", "K"): -2, ("L", "M"):  2, ("L", "F"):  0, ("L", "P"): -3,
    ("L", "S"): -2, ("L", "T"): -1, ("L", "W"): -2, ("L", "Y"): -1, ("L", "V"):  1,
    ("K", "K"):  5, ("K", "M"): -1, ("K", "F"): -3, ("K", "P"): -1, ("K", "S"):  0,
    ("K", "T"): -1, ("K", "W"): -3, ("K", "Y"): -2, ("K", "V"): -2,
    ("M", "M"):  5, ("M", "F"):  0, ("M", "P"): -2, ("M", "S"): -1, ("M", "T"): -1,
    ("M", "W"): -1, ("M", "Y"): -1, ("M", "V"):  1,
    ("F", "F"):  6, ("F", "P"): -4, ("F", "S"): -2, ("F", "T"): -2, ("F", "W"):  1,
    ("F", "Y"):  3, ("F", "V"): -1,
    ("P", "P"):  7, ("P", "S"): -1, ("P", "T"): -1, ("P", "W"): -4, ("P", "Y"): -3,
    ("P", "V"): -2,
    ("S", "S"):  4, ("S", "T"):  1, ("S", "W"): -3, ("S", "Y"): -2, ("S", "V"): -2,
    ("T", "T"):  5, ("T", "W"): -2, ("T", "Y"): -2, ("T", "V"):  0,
    ("W", "W"): 11, ("W", "Y"):  2, ("W", "V"): -3,
    ("Y", "Y"):  7, ("Y", "V"): -1,
    ("V", "V"):  4,
}


def get_blosum62_score(aa1: str, aa2: str) -> int:
    """Get BLOSUM62 substitution score for aa1→aa2."""
    key = (aa1.upper(), aa2.upper())
    if key in BLOSUM62:
        return BLOSUM62[key]
    # Try reversed
    key_rev = (aa2.upper(), aa1.upper())
    if key_rev in BLOSUM62:
        return BLOSUM62[key_rev]
    return 0  # Unknown pair


# ─────────────────────────────────────────────
# Amino Acid Properties
# ─────────────────────────────────────────────

# Property tables for 20 standard amino acids
AA_PROPERTIES = {
    # Polarity: 0 = nonpolar, 1 = polar uncharged, 2 = positive, 3 = negative
    "polarity": {
        "A": 0, "V": 0, "I": 0, "L": 0, "M": 0, "F": 0, "W": 0, "P": 0, "G": 0,
        "S": 1, "T": 1, "C": 1, "Y": 1, "N": 1, "Q": 1,
        "K": 2, "R": 2, "H": 2,
        "D": 3, "E": 3,
    },
    
    # Charge at pH 7: -1, 0, +1
    "charge": {
        "A": 0, "V": 0, "I": 0, "L": 0, "M": 0, "F": 0, "W": 0, "P": 0,
        "G": 0, "S": 0, "T": 0, "C": 0, "Y": 0, "N": 0, "Q": 0, "H": 0,
        "K": 1, "R": 1,
        "D": -1, "E": -1,
    },
    
    # Molecular weight (Da) — normalized to [0, 1]
    "size": {
        "G": 57.05, "A": 71.08, "V": 99.13, "L": 113.16, "I": 113.16,
        "P": 97.12, "F": 147.18, "W": 186.21, "M": 131.20, "S": 87.08,
        "T": 101.10, "C": 103.14, "Y": 163.18, "H": 137.14, "D": 115.09,
        "E": 129.12, "N": 114.10, "Q": 128.13, "K": 128.17, "R": 156.19,
    },
    
    # Kyte-Doolittle hydrophobicity scale
    "hydrophobicity": {
        "I":  4.5, "V":  4.2, "L":  3.8, "F":  2.8, "C":  2.5,
        "M":  1.9, "A":  1.8, "G": -0.4, "T": -0.7, "S": -0.8,
        "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5,
        "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
    },
}

# Normalize size to [0, 1]
_size_values = list(AA_PROPERTIES["size"].values())
_size_min, _size_max = min(_size_values), max(_size_values)
AA_PROPERTIES["size_normalized"] = {
    aa: (val - _size_min) / (_size_max - _size_min)
    for aa, val in AA_PROPERTIES["size"].items()
}

# Normalize hydrophobicity to [0, 1]
_hydro_values = list(AA_PROPERTIES["hydrophobicity"].values())
_hydro_min, _hydro_max = min(_hydro_values), max(_hydro_values)
AA_PROPERTIES["hydrophobicity_normalized"] = {
    aa: (val - _hydro_min) / (_hydro_max - _hydro_min)
    for aa, val in AA_PROPERTIES["hydrophobicity"].items()
}


def get_property_change(aa_wt: str, aa_mut: str, prop: str) -> float:
    """Compute property change: property(mut) - property(wt)."""
    prop_table = AA_PROPERTIES.get(prop, {})
    wt_val = prop_table.get(aa_wt.upper(), 0)
    mut_val = prop_table.get(aa_mut.upper(), 0)
    return float(mut_val - wt_val)


# ─────────────────────────────────────────────
# Main Biochemical Feature Extractor
# ─────────────────────────────────────────────

class BiochemicalFeatureExtractor:
    """
    Extracts the 6D biochemical prior feature vector.
    
    For each mutation site, computes:
      [0] BLOSUM62 score (wt→mut substitution)
      [1] Polarity change
      [2] Charge change
      [3] Size change (normalized MW)
      [4] Hydrophobicity change (Kyte-Doolittle)
      [5] Contact distance at mutation site
      
    Output feeds into Physicochemical Residual Injection (Stage 4D).
    """
    
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        bio_cfg = config.get("features", {}).get("biochemical", {})
        
        self.output_dim = bio_cfg.get("output_dim", 6)
        self.use_blosum = bio_cfg.get("blosum62", True)
        self.properties = bio_cfg.get("amino_acid_properties", [
            "polarity", "charge", "size_normalized", "hydrophobicity_normalized"
        ])
        self.use_contact_dist = bio_cfg.get("contact_distance", True)
    
    def extract_per_mutation(
        self,
        wild_aa: str,
        mut_aa: str,
        contact_distance: float = 0.0,
    ) -> np.ndarray:
        """
        Extract 6D biochemical feature for a single mutation.
        
        Args:
            wild_aa: Wild-type amino acid (1-letter).
            mut_aa: Mutant amino acid (1-letter).
            contact_distance: Nearest contact distance at mutation site (Å).
            
        Returns:
            (6,) feature vector.
        """
        features = []
        
        # 1. BLOSUM62 score
        if self.use_blosum:
            blosum_score = get_blosum62_score(wild_aa, mut_aa)
            # Normalize BLOSUM62 to roughly [-1, 1]
            features.append(blosum_score / 11.0)  # Max BLOSUM62 score = 11 (W→W)
        
        # 2–5. Property changes
        property_names = ["polarity", "charge", "size_normalized", "hydrophobicity_normalized"]
        for prop in property_names:
            change = get_property_change(wild_aa, mut_aa, prop)
            features.append(change)
        
        # 6. Contact distance (normalized)
        if self.use_contact_dist:
            # Normalize distance: 0-20Å → 0-1
            features.append(min(contact_distance / 20.0, 1.0))
        
        # Pad or truncate to output_dim
        while len(features) < self.output_dim:
            features.append(0.0)
        features = features[:self.output_dim]
        
        return np.array(features, dtype=np.float32)
    
    def extract(
        self,
        mutations: List[Dict],
        distance_matrix: Optional[np.ndarray] = None,
        residue_indices: Optional[Dict[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract biochemical features for all mutations in an entry.
        
        Args:
            mutations: List of mutation dicts (wild_aa, mut_aa, resnum, etc.).
            distance_matrix: (N, N) distance matrix for contact distances.
            residue_indices: Mapping of residue number → index in distance matrix.
            
        Returns:
            Dict with:
              - biochemical_features: (M, 6) tensor for M mutations
              - per_residue_features: (N, 6) tensor (zeros for non-mutated)
        """
        per_mutation_features = []
        
        for mut in mutations:
            # Compute contact distance if available
            contact_dist = 0.0
            if distance_matrix is not None and residue_indices is not None:
                resnum = mut.get("matched_resnum", mut["resnum"])
                if resnum in residue_indices:
                    idx = residue_indices[resnum]
                    # Minimum distance to any other residue
                    dists = distance_matrix[idx].copy()
                    dists[idx] = float("inf")
                    contact_dist = float(np.min(dists))
            
            feat = self.extract_per_mutation(
                wild_aa=mut["wild_aa"],
                mut_aa=mut["mut_aa"],
                contact_distance=contact_dist,
            )
            per_mutation_features.append(feat)
        
        if not per_mutation_features:
            per_mutation_features = [np.zeros(self.output_dim, dtype=np.float32)]
        
        features_array = np.stack(per_mutation_features, axis=0)  # (M, 6)
        
        result = {
            "biochemical_features": torch.tensor(features_array, dtype=torch.float32),
            "num_mutations": len(mutations),
        }
        
        logger.debug(
            f"Biochemical features: {len(mutations)} mutations → "
            f"shape={features_array.shape}"
        )
        
        return result
    
    def extract_global(
        self,
        wt_sequence: str,
        mut_sequence: str,
    ) -> torch.Tensor:
        """
        Extract per-residue biochemical features for the full sequence.
        Non-mutated positions get zero vectors.
        
        Args:
            wt_sequence: Wild-type sequence.
            mut_sequence: Mutant sequence.
            
        Returns:
            (L, 6) tensor of biochemical features.
        """
        L = len(wt_sequence)
        features = np.zeros((L, self.output_dim), dtype=np.float32)
        
        for i in range(L):
            if wt_sequence[i] != mut_sequence[i]:
                features[i] = self.extract_per_mutation(
                    wild_aa=wt_sequence[i],
                    mut_aa=mut_sequence[i],
                )
        
        return torch.tensor(features, dtype=torch.float32)
