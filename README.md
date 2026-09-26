# abCAN-v2: Antibody–Antigen ΔΔG Prediction

A deep learning architecture for predicting changes in binding free energy (ΔΔG) 
upon mutations at antibody-antigen interfaces.

## Architecture Overview

abCAN-v2 combines:
- **MPNN Structural Encoder** — 3-layer graph convolution on residue contact graphs
- **MPNN Sequence Encoder v2** — ESM-2 pretrained embeddings with cross-attention
- **Interface-Aware Dual-Stream Attention** — multi-head + interface-zone gating
- **Physicochemical Residual Injection** — 6D biochemical priors with gated connections
- **Antisymmetric Difference** — ΔRepresentation (Mut − WT)
- **Mutation-Aware Weighted Aggregation** — attention-weighted pooling (novel contribution)
- **Prediction Head** — 128 → 64 → 16 → 1

## Dataset

Antibody-Antigen Mutation Benchmark:
- **1,525 rows** across **61 PDB complexes**
- Experimental ΔΔG values with σ = 0.984 kcal/mol

## Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
abcan/
├── config/              # Configuration files
├── data/                # Dataset loading & preprocessing
├── features/            # Feature extraction (structural, sequence, biochemical)
├── models/              # abCAN-v2 model components (A–G)
├── training/            # Training loop, losses, schedulers
├── evaluation/          # Metrics, ablation studies
├── deployment/          # Flask/FastAPI backend + React frontend
├── scripts/             # Entry-point scripts (train, evaluate, predict)
└── tests/               # Unit and integration tests
```

## Usage

```bash
# Train
python -m scripts.train --config config/default_config.yaml

# Evaluate
python -m scripts.evaluate --checkpoint checkpoints/best_model.pt

# Predict
python -m scripts.predict --pdb complex.pdb --mutations "A:K45R,A:D52N"
```

## License

Research use only.
