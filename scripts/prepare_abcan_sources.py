"""Normalize available AB-Bind and SKEMPI sources without assigning splits."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "database"
OUT = DB / "abcan"
CSV_OUT = OUT / "source_candidates_unfiltered.csv"
AUDIT_OUT = OUT / "source_audit.json"
AB_PATH = DB / "AB-Bind_experimental_data.csv"
SK_PATH = DB / "SKEMPI_Processed_Dataset.xlsx"
FIELDS = [
    "source_dataset", "source_file", "source_record_id", "source_row",
    "complex_id", "pdb_id", "partner_1_chains", "partner_2_chains",
    "protein_1", "protein_2", "mutation_pdb", "mutation_cleaned",
    "mutation_count", "ddg_kcal_mol", "label_method", "reference",
    "data_quality_flag", "split", "preparation_status",
]


def num(value: Any) -> Optional[float]:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def mut_count(value: Any) -> Optional[int]:
    value = str(value or "").strip()
    return sum(bool(part.strip()) for part in value.split(",")) if value else None


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    for path in (AB_PATH, SK_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    # If AB_PATH is a Git LFS pointer file, pull the real data
    try:
        is_lfs = False
        with AB_PATH.open("r", encoding="latin1", errors="ignore") as f:
            is_lfs = f.readline().startswith("version https://git-lfs.github.com/spec/v1")
        if is_lfs:
            import subprocess
            print("Detected Git LFS pointer for AB-Bind, pulling actual data...")
            subprocess.run(["git", "lfs", "install"], cwd=str(ROOT), check=False)
            res = subprocess.run(["git", "lfs", "pull"], cwd=str(ROOT), capture_output=True, text=True, check=False)
            if res.stdout:
                print(res.stdout)
            if res.stderr:
                print(res.stderr)
    except Exception as exc:
        print("Git LFS auto-pull encountered:", exc)

    ab = pd.read_csv(AB_PATH, encoding="latin1", dtype=str, keep_default_na=False)
    ab_required = {"#PDB", "Partners(A_B)", "Protein-1", "Protein-2", "Mutation", "ddG(kcal/mol)"}
    if ab_required - set(ab.columns):
        raise ValueError(
            f"AB-Bind missing columns: {sorted(ab_required - set(ab.columns))}. "
            "If this file is a Git LFS pointer, run `git lfs pull` to download the dataset."
        )

    candidates = []
    for i, r in ab.iterrows():
        partners = str(r["Partners(A_B)"]).strip()
        chain1, sep, chain2 = partners.partition("_")
        mutation = str(r["Mutation"]).strip()
        candidates.append({
            "source_dataset": "AB-Bind",
            "source_file": str(AB_PATH.relative_to(ROOT)),
            "source_record_id": f"AB-Bind:row-{i + 2}",
            "source_row": i + 2,
            "complex_id": f"{r['#PDB']}:{partners}",
            "pdb_id": str(r["#PDB"]).strip(),
            "partner_1_chains": chain1,
            "partner_2_chains": chain2 if sep else "",
            "protein_1": str(r["Protein-1"]).strip(),
            "protein_2": str(r["Protein-2"]).strip(),
            "mutation_pdb": mutation,
            "mutation_cleaned": mutation,
            "mutation_count": mut_count(mutation),
            "ddg_kcal_mol": num(r["ddG(kcal/mol)"]),
            "label_method": "experimental_ddg_from_AB-Bind",
            "reference": str(r.get("PDB DOI", "")).strip(),
            "data_quality_flag": "source label present",
            "split": "",
            "preparation_status": "unfiltered_candidate_no_split",
        })

    sk = pd.read_excel(SK_PATH, sheet_name="Processed Data")
    sk_required = {
        "Record_ID", "Source_Row", "Complex_ID", "PDB_ID", "Partner_1_Chains",
        "Partner_2_Chains", "Mutation_PDB", "Mutation_Cleaned", "Mutation_Count",
        "Mutant_Affinity_M", "Wild_Type_Affinity_M", "Temperature_K",
        "Delta_Delta_G_kcal_mol", "SKEMPI_Version", "Data_Quality_Flag",
    }
    if sk_required - set(sk.columns):
        raise ValueError(f"SKEMPI workbook missing columns: {sorted(sk_required - set(sk.columns))}")

    sk_cached = 0
    sk_recomputed = 0
    for _, r in sk.iterrows():
        mutant, wt, temp = num(r["Mutant_Affinity_M"]), num(r["Wild_Type_Affinity_M"]), num(r["Temperature_K"])
        ddg = num(r["Delta_Delta_G_kcal_mol"])
        method = "source_workbook_cached_ddg"
        if mutant is not None and wt is not None and temp is not None and mutant > 0 and wt > 0:
            formula_value = 0.0019872041 * temp * math.log(mutant / wt)
            if ddg is None:
                ddg = formula_value
                method = "recomputed_RT_ln_Kd_mut_over_wt"
                sk_recomputed += 1
            else:
                if not math.isclose(ddg, formula_value, rel_tol=1e-8, abs_tol=1e-8):
                    raise ValueError(f"Cached SKEMPI ΔΔG disagrees with formula at source row {r['Source_Row']}")
                sk_cached += 1

        version = num(r["SKEMPI_Version"])
        version_label = str(int(version)) if version is not None else "unknown"
        source = f"SKEMPI {version_label}" if version_label in {"1", "2"} else "SKEMPI"
        mutation = str(r["Mutation_Cleaned"] or "").strip()
        candidates.append({
            "source_dataset": source,
            "source_file": str(SK_PATH.relative_to(ROOT)),
            "source_record_id": f"{source}:{str(r['Record_ID']).strip()}",
            "source_row": num(r["Source_Row"]),
            "complex_id": str(r["Complex_ID"]).strip(),
            "pdb_id": str(r["PDB_ID"]).strip(),
            "partner_1_chains": str(r["Partner_1_Chains"]).strip(),
            "partner_2_chains": str(r["Partner_2_Chains"]).strip(),
            "protein_1": str(r.get("Protein_1", "") or "").strip(),
            "protein_2": str(r.get("Protein_2", "") or "").strip(),
            "mutation_pdb": str(r["Mutation_PDB"] or "").strip(),
            "mutation_cleaned": mutation,
            "mutation_count": num(r["Mutation_Count"]),
            "ddg_kcal_mol": ddg,
            "label_method": method if ddg is not None else "missing_required_measurement",
            "reference": str(r.get("Reference", "") or "").strip(),
            "data_quality_flag": str(r["Data_Quality_Flag"] or "").strip(),
            "split": "",
            "preparation_status": "unfiltered_candidate_no_split",
        })

    OUT.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidates)

    names = [p.name.lower() for p in DB.rglob("*") if p.is_file()]
    missing = [
        label for label, present in (
            ("PROXiMATE source table", any("proximate" in n for n in names)),
            ("M515 held-out labels/manifest", any("m515" in n for n in names)),
            ("abS1131 constituent data", any("abs1131" in n for n in names)),
            ("abM1707 constituent data", any("abm1707" in n for n in names)),
        ) if not present
    ]
    versions = sk["SKEMPI_Version"].value_counts(dropna=False).to_dict()
    audit = {
        "status": "candidate_sources_normalized_not_train_ready",
        "candidate_file": str(CSV_OUT.relative_to(ROOT)),
        "candidate_records": len(candidates),
        "split_assignments": "none; this is not a training or test manifest",
        "deduplication": "no additional deduplication; the workbook Processed Data sheet already removes 11 exact duplicates, with raw rows retained in its Raw Data sheet",
        "sources": [
            {
                "name": "AB-Bind",
                "file": str(AB_PATH.relative_to(ROOT)),
                "sha256": digest(AB_PATH),
                "records": len(ab),
                "unique_pdb_ids": int(ab["#PDB"].nunique()),
                "records_with_ddg": sum(row["source_dataset"] == "AB-Bind" and row["ddg_kcal_mol"] is not None for row in candidates),
            },
            {
                "name": "SKEMPI processed workbook",
                "file": str(SK_PATH.relative_to(ROOT)),
                "sha256": digest(SK_PATH),
                "sheet": "Processed Data",
                "records": len(sk),
                "records_by_version": {str(k): int(v) for k, v in versions.items()},
                "cached_ddg_values_verified": sk_cached,
                "ddg_values_recomputed_from_documented_formula": sk_recomputed,
                "records_with_ddg_after_recalculation": sum(row["source_dataset"].startswith("SKEMPI") and row["ddg_kcal_mol"] is not None for row in candidates),
            },
        ],
        "skempi_v2_csv": "present but not merged because it overlaps the SKEMPI v1+v2 workbook source",
        "missing_requirements": missing,
        "remaining_preparation": [
            "Identify the paper's antibody-antigen rows and M515 membership.",
            "Apply CATH family and 80% sequence-identity exclusion against M515.",
            "Use official train/validation membership; keep M515 held out.",
            "Build features using the original abCAN representation.",
        ],
    }
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"records": len(candidates), "candidate_file": str(CSV_OUT), "audit_file": str(AUDIT_OUT), "status": audit["status"]}, indent=2))


if __name__ == "__main__":
    main()

