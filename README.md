# abCAN-v2: antibody–antigen ΔΔG prediction

This checkout actively trains only from the available abCAN source tables: **AB-Bind** and the **SKEMPI Processed Data** workbook. SAbDab2 and AbAgym remain in the repository for future work, but the active script never reads them.

## Benchmark and reference

The published abCAN paper reports **RMSE 1.460 kcal/mol** and **PCC 0.731** on M515. Its comparison table lists RMSE as 1.476, so this project records the discrepancy instead of presenting either value as a result from this checkout.

The paper's leakage-filtered abCAN train/validation split and M515 test manifest are **not present in this checkout**. The active workflow therefore makes grouped train, validation, and held-out source splits using AB-Bind and SKEMPI only. Its results are source-set results, not M515 benchmark results. PROXiMATE and the M515 membership/labels are still missing. SAbDab2 and AbAgym remain separate future datasets; AbAgym DMS scores are not direct ΔΔG measurements in kcal/mol.

The [official abCAN repository](https://github.com/ChenGong57/abCAN) lists M305.csv and a pretrained checkpoint, but not the paper’s M515 manifest or the cleaned training manifest used for the reported experiment. M305 is not a substitute for the M515 held-out test set. The paper describes M515 as the held-out combination of abS1131 and abM1707, with separate cleaned abCAN training data and family/similarity filtering to prevent leakage.


## Available source tables

The current source staging combines 1,101 AB-Bind records with 7,074 rows from the SKEMPI workbook's Processed Data sheet, including SKEMPI 1 and 2. It recalculates missing saved ΔΔG formula results where all affinity and temperature inputs are present, retaining rows with missing measurements as blank. The standalone skempi_v2.csv is not merged again because it overlaps the workbook source.

Run `python -m scripts.prepare_abcan_sources` to create `database/abcan/source_candidates_unfiltered.csv` and `database/abcan/source_audit.json`. `scripts.train` uses only that candidate file, derives mutation-level biochemical features, and makes grouped splits by complex ID. It does not read SAbDab2 or AbAgym.

## Active data path

The active run reads `database/abcan/source_candidates_unfiltered.csv`, generated only from `database/AB-Bind_experimental_data.csv` and `database/SKEMPI_Processed_Dataset.xlsx`. It trains a reproducible mutation-feature baseline and saves its model, held-out source-set predictions, and RMSE/PCC results under `checkpoints/` and `evaluation/results/`.

M515 is not currently evaluated. When the original M515 manifest and leakage-safe abCAN split are available, they must be added as a separate final evaluation path and must never guide training or model selection.

## Other datasets and approaches

SAbDab2 and AbAgym assets and their configuration remain available for future work. The frontend marks their benchmark scores and the sequence-only, structure-only, and ensemble comparisons as **Coming soon**. They are inactive in the current abCAN M515 training path.

## Run

    python -m scripts.prepare_abcan_sources
    python -m scripts.train

The API status endpoint is GET /benchmark. Mutation prediction returns unavailable until a validated model and real prediction feature pipeline are in place; the backend no longer returns a fixed mock prediction.

## Project layout

    config/                 Active AB-Bind/SKEMPI source-only configuration
    data/                   Retained feature-cache components for future M515 work
    database/                Active abCAN sources and retained future datasets
    evaluation/              Regression metrics and saved source-only results
    models/                  abCAN-v2 model components
    training/                Training, validation, checkpoint selection, test scoring
    deployment/backend/      abCAN M515 status API
    deployment/frontend/     Benchmark dashboard and future-work indicators
    scripts/train.py         AB-Bind/SKEMPI-only training and grouped evaluation

## Installation

    python -m venv .venv
    pip install -r requirements.txt

