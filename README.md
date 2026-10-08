# abCAN-v2: antibody–antigen ΔΔG prediction

This checkout is focused on the original **abCAN** training data and its held-out **M515** test benchmark. SAbDab2 and AbAgym files remain in the repository for future work, but the current training and score-reporting path does not use them.

## Benchmark and reference

The published abCAN paper reports **RMSE 1.460 kcal/mol** and **PCC 0.731** on M515. Its comparison table lists RMSE as 1.476, so this project records the discrepancy instead of presenting either value as a result from this checkout.

The paper's leakage-filtered abCAN train/validation split and M515 test manifest are **not present in this checkout**. AB-Bind and SKEMPI source tables are now available, but they are broader protein-interaction sources and are not the paper's final curated split. PROXiMATE and the M515 membership/labels are still missing. SAbDab2 and AbAgym remain separate future datasets; AbAgym DMS scores are not direct ΔΔG measurements in kcal/mol.

The [official abCAN repository](https://github.com/ChenGong57/abCAN) lists M305.csv and a pretrained checkpoint, but not the paper’s M515 manifest or the cleaned training manifest used for the reported experiment. M305 is not a substitute for the M515 held-out test set. The paper describes M515 as the held-out combination of abS1131 and abM1707, with separate cleaned abCAN training data and family/similarity filtering to prevent leakage.


## Available source tables

The current source staging combines 1,101 AB-Bind records with 7,074 rows from the SKEMPI workbook's Processed Data sheet, including SKEMPI 1 and 2. It recalculates missing saved ΔΔG formula results where all affinity and temperature inputs are present, retaining rows with missing measurements as blank. The standalone skempi_v2.csv is not merged again because it overlaps the workbook source.

Run python -m scripts.prepare_abcan_sources to create database/abcan/source_candidates_unfiltered.csv and database/abcan/source_audit.json. The candidate file retains source provenance and has no split assignments. It is not ready for training until antibody–antigen records are identified, PROXiMATE and M515 data are added, and the paper's leakage-safe split is reconstructed.

## Active data path

The config expects the original data manifests and one prepared feature cache at:

- database/abcan/training.csv — original, leakage-filtered abCAN training data
- database/abcan/M515.csv — held-out abS1131 + abM1707 benchmark data
- database/abcan/abcan_training_and_m515_features.pt — prepared real model inputs

The feature cache must contain sample dictionaries for the train, validation, and `m515_test` splits. Before caching, reproduce the paper’s protein-family and 80% sequence-identity exclusions between training data and M515. The M515 test split must remain held out and must never guide training, checkpoint selection, or hyperparameter tuning. Each sample needs the tensors consumed by data/dataset.py: structure node features and coordinates, graph edges and edge features, WT and mutant sequence embeddings, interface and mutation masks, zero-shot LLR, and the experimental ddG label. The loader rejects synthetic placeholders; it does not build features or verify the split exclusions itself.

Training uses only abCAN train and validation splits, selects the checkpoint by validation RMSE, then evaluates the held-out M515 test split once. Test metrics are written to evaluation/results/abcan_m515.json. The current repository does not yet contain the original abCAN manifests or a feature-cache builder, so training stops with a clear missing-input message instead of falling back to another dataset.

## Other datasets and approaches

SAbDab2 and AbAgym assets and their configuration remain available for future work. The frontend marks their benchmark scores and the sequence-only, structure-only, and ensemble comparisons as **Coming soon**. They are inactive in the current abCAN M515 training path.

## Run

    python -m scripts.train

The API status endpoint is GET /benchmark. Mutation prediction returns unavailable until a validated model and real prediction feature pipeline are in place; the backend no longer returns a fixed mock prediction.

## Project layout

    config/                 Active abCAN training and M515 test configuration
    data/                   abCAN feature-cache loader and graph batching
    database/                Retained benchmark and structure files
    evaluation/              Regression metrics and saved M515 results
    models/                  abCAN-v2 model components
    training/                Training, validation, checkpoint selection, test scoring
    deployment/backend/      abCAN M515 status API
    deployment/frontend/     Benchmark dashboard and future-work indicators
    scripts/train.py         abCAN training and held-out M515 scoring

## Installation

    python -m venv .venv
    pip install -r requirements.txt

