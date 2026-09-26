"""
abCAN-v2 — Data Preprocessing
Stage 2: Curated amino acid sequence processing, replicate handling,
and PDB residue concordance matching.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from data.pdb_utils import (
    PDBHandler,
    ResidueConcordanceMatcher,
    ComplexStructure,
    AA_3TO1,
    AA_1TO3,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Amino Acid Sequence Curation
# ─────────────────────────────────────────────

class SequenceCurator:
    """
    Curates amino acid sequences from PDB structures.
    
    From architecture diagram (Stage 2):
      - Extract curated sequences (variable ΔΔG > 8 kcal/mol filter already applied)
      - Replicate experimental condition handling (σ = 0.984 kcal/mol)
      - PDB residue concordance matching
    """
    
    STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")
    
    def __init__(
        self,
        max_length: int = 512,
        pad_token: str = "X",
    ):
        """
        Args:
            max_length: Maximum sequence length (truncate if longer).
            pad_token: Padding character for short sequences.
        """
        self.max_length = max_length
        self.pad_token = pad_token
    
    def extract_wild_mutant_sequences(
        self,
        complex_struct: ComplexStructure,
        mutations: List[Dict[str, Any]],
        chain_id: str,
    ) -> Tuple[str, str]:
        """
        Extract wild-type and mutant sequences for a given chain.
        
        Args:
            complex_struct: Parsed PDB complex.
            mutations: List of mutation dicts (chain, resnum, wild_aa, mut_aa).
            chain_id: Which chain to extract.
            
        Returns:
            (wild_type_sequence, mutant_sequence)
        """
        if chain_id not in complex_struct.residues:
            raise ValueError(f"Chain {chain_id} not in complex {complex_struct.pdb_id}")
        
        residues = complex_struct.residues[chain_id]
        wt_seq = [r.aa_one_letter for r in residues]
        mut_seq = wt_seq.copy()
        
        # Build residue number → index mapping
        resnum_to_idx = {r.residue_number: i for i, r in enumerate(residues)}
        
        # Apply mutations
        chain_mutations = [m for m in mutations if m["chain"] == chain_id]
        for mut in chain_mutations:
            resnum = mut.get("matched_resnum", mut["resnum"])
            if resnum in resnum_to_idx:
                idx = resnum_to_idx[resnum]
                # Verify wild-type matches
                if wt_seq[idx] == mut["wild_aa"]:
                    mut_seq[idx] = mut["mut_aa"]
                else:
                    logger.warning(
                        f"WT mismatch at {chain_id}:{resnum}: "
                        f"expected {mut['wild_aa']}, found {wt_seq[idx]}"
                    )
        
        wt_str = "".join(wt_seq)
        mut_str = "".join(mut_seq)
        
        return wt_str, mut_str
    
    def clean_sequence(self, sequence: str) -> str:
        """
        Clean a protein sequence:
        - Remove non-standard characters
        - Truncate to max_length
        """
        # Replace non-standard AAs with X
        cleaned = ""
        for aa in sequence:
            if aa in self.STANDARD_AAS:
                cleaned += aa
            else:
                cleaned += self.pad_token
        
        # Truncate
        if len(cleaned) > self.max_length:
            cleaned = cleaned[:self.max_length]
        
        return cleaned
    
    def pad_sequence(self, sequence: str, target_length: Optional[int] = None) -> str:
        """Pad sequence to target length."""
        length = target_length or self.max_length
        if len(sequence) < length:
            sequence += self.pad_token * (length - len(sequence))
        return sequence


# ─────────────────────────────────────────────
# Replicate Handler
# ─────────────────────────────────────────────

class ReplicateHandler:
    """
    Handles replicate experimental conditions.
    
    From architecture: σ = 0.984 kcal/mol experimental noise.
    
    - Averages replicate ΔΔG measurements
    - Computes per-measurement uncertainty weights
    - Flags high-variance replicates
    """
    
    def __init__(self, experimental_sigma: float = 0.984):
        """
        Args:
            experimental_sigma: Known experimental noise standard deviation.
        """
        self.sigma = experimental_sigma
    
    def process_replicates(
        self,
        entries: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Process replicate measurements.
        
        Groups entries by (PDB, mutation) and averages ΔΔG values.
        Adds uncertainty weight based on number of replicates and variance.
        
        Returns:
            Processed entries with averaged replicates and weights.
        """
        # Group by PDB + mutation combination
        groups: Dict[str, List[Dict]] = {}
        for entry in entries:
            key = f"{entry['pdb_id']}_{entry['mutation_string']}"
            if key not in groups:
                groups[key] = []
            groups[key].append(entry)
        
        processed = []
        for key, group in groups.items():
            if len(group) == 1:
                # Single measurement
                entry = group[0].copy()
                entry["ddg_std"] = self.sigma
                entry["ddg_weight"] = 1.0
                entry["n_replicates"] = 1
                processed.append(entry)
            else:
                # Multiple replicates — average
                ddg_values = [e["ddg"] for e in group]
                mean_ddg = np.mean(ddg_values)
                std_ddg = np.std(ddg_values)
                
                # Uncertainty weight: more replicates → higher confidence
                # σ_mean = σ / √n (standard error of the mean)
                n = len(group)
                sem = self.sigma / np.sqrt(n)
                weight = 1.0 / (sem ** 2)  # Inverse variance weight
                
                entry = group[0].copy()
                entry["ddg"] = mean_ddg
                entry["ddg_std"] = std_ddg
                entry["ddg_weight"] = weight
                entry["n_replicates"] = n
                
                # Flag if replicate variance is high
                if std_ddg > 2 * self.sigma:
                    logger.warning(
                        f"High-variance replicates for {key}: "
                        f"σ_rep={std_ddg:.3f} vs σ_exp={self.sigma:.3f}"
                    )
                    entry["high_variance_flag"] = True
                else:
                    entry["high_variance_flag"] = False
                
                processed.append(entry)
        
        logger.info(
            f"Processed replicates: {len(entries)} → {len(processed)} entries "
            f"({len(entries) - len(processed)} replicates merged)"
        )
        
        return processed
    
    def compute_sample_weight(self, ddg_std: float, n_replicates: int) -> float:
        """
        Compute a training sample weight based on measurement uncertainty.
        
        Higher weight = more reliable measurement.
        """
        sem = self.sigma / np.sqrt(max(n_replicates, 1))
        total_uncertainty = np.sqrt(sem**2 + ddg_std**2)
        weight = 1.0 / total_uncertainty
        return weight


# ─────────────────────────────────────────────
# Full Preprocessing Pipeline
# ─────────────────────────────────────────────

class DataPreprocessor:
    """
    Orchestrates the full Stage 2 preprocessing pipeline.
    
    Pipeline:
      1. Sequence curation (wild-type and mutant)
      2. Replicate handling
      3. PDB residue concordance matching
      4. Quality checks
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dict (from config/default_config.yaml).
        """
        self.config = config
        
        prep_cfg = config.get("preprocessing", {})
        seq_cfg = prep_cfg.get("sequence", {})
        struct_cfg = prep_cfg.get("structure", {})
        dataset_cfg = config.get("dataset", {})
        
        self.sequence_curator = SequenceCurator(
            max_length=seq_cfg.get("max_length", 512),
            pad_token=seq_cfg.get("pad_token", "X"),
        )
        
        self.replicate_handler = ReplicateHandler(
            experimental_sigma=dataset_cfg.get("experimental_sigma", 0.984),
        )
        
        self.concordance_matcher = ResidueConcordanceMatcher(
            max_offset=dataset_cfg.get("concordance", {}).get("max_offset", 5),
            fallback=dataset_cfg.get("concordance", {}).get("fallback", "skip"),
        )
        
        self.pdb_handler = PDBHandler(
            pdb_dir=dataset_cfg.get("pdb_dir", "data/raw/pdbs/"),
        )
    
    def preprocess(
        self,
        entries: List[Dict[str, Any]],
        complexes: Dict[str, ComplexStructure],
    ) -> List[Dict[str, Any]]:
        """
        Run the full preprocessing pipeline.
        
        Args:
            entries: Raw dataset entries from SKEMPIDataset.
            complexes: Dict of PDB ID → ComplexStructure.
            
        Returns:
            Preprocessed entries ready for feature extraction.
        """
        logger.info("=" * 60)
        logger.info("Stage 2: Data Preprocessing")
        logger.info("=" * 60)
        
        # Step 1: Handle replicates
        logger.info("Step 1: Processing replicates...")
        entries = self.replicate_handler.process_replicates(entries)
        
        # Step 2: Concordance matching + sequence extraction
        logger.info("Step 2: Concordance matching & sequence extraction...")
        processed = []
        concordance_stats = {"exact": 0, "offset": 0, "not_found": 0, "skipped": 0}
        
        for entry in entries:
            pdb_id = entry["pdb_id"]
            
            # Load complex if available
            if pdb_id not in complexes:
                try:
                    complex_struct = self.pdb_handler.load_complex(pdb_id)
                    complexes[pdb_id] = complex_struct
                except Exception as e:
                    logger.debug(f"Cannot load PDB {pdb_id}: {e}")
                    concordance_stats["skipped"] += 1
                    continue
            
            complex_struct = complexes[pdb_id]
            
            # Concordance matching for each mutation
            all_concordant = True
            for mut in entry["mutations"]:
                result = self.concordance_matcher.match_residue(
                    query_chain=mut["chain"],
                    query_resnum=mut["resnum"],
                    query_aa=mut["wild_aa"],
                    complex_struct=complex_struct,
                )
                
                concordance_stats[result.status] += 1
                
                if result.status == "not_found":
                    all_concordant = False
                elif result.matched_resnum is not None:
                    mut["matched_resnum"] = result.matched_resnum
                    mut["concordance_status"] = result.status
                    mut["concordance_offset"] = result.offset
            
            # Skip entries with unresolved concordance (configurable)
            if not all_concordant and self.concordance_matcher.fallback == "skip":
                concordance_stats["skipped"] += 1
                continue
            
            # Extract wild-type and mutant sequences
            try:
                for chain_id in set(m["chain"] for m in entry["mutations"]):
                    wt_seq, mut_seq = self.sequence_curator.extract_wild_mutant_sequences(
                        complex_struct, entry["mutations"], chain_id
                    )
                    entry[f"wt_sequence_{chain_id}"] = self.sequence_curator.clean_sequence(wt_seq)
                    entry[f"mut_sequence_{chain_id}"] = self.sequence_curator.clean_sequence(mut_seq)
                
                # Store the primary mutation chain's sequences at top level
                primary_chain = entry["mutations"][0]["chain"]
                entry["wt_sequence"] = entry.get(f"wt_sequence_{primary_chain}", "")
                entry["mut_sequence"] = entry.get(f"mut_sequence_{primary_chain}", "")
                
            except Exception as e:
                logger.debug(f"Sequence extraction failed for {pdb_id}: {e}")
                continue
            
            # Compute sample weight
            entry["sample_weight"] = self.replicate_handler.compute_sample_weight(
                ddg_std=entry.get("ddg_std", self.replicate_handler.sigma),
                n_replicates=entry.get("n_replicates", 1),
            )
            
            processed.append(entry)
        
        # Report concordance statistics
        total_muts = sum(concordance_stats.values())
        logger.info(
            f"Concordance results: "
            f"exact={concordance_stats['exact']}, "
            f"offset={concordance_stats['offset']}, "
            f"not_found={concordance_stats['not_found']}, "
            f"skipped={concordance_stats['skipped']}"
        )
        logger.info(f"Preprocessing complete: {len(processed)}/{len(entries)} entries retained")
        
        return processed
