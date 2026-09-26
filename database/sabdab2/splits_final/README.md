SAbDab2 ML Data
===============

Overview
--------

A subset of clean, curated antibody and antibody–antigen-complex structures from [SAbDab2](https://sabdab2.opig.stats.ox.ac.uk), specifically post-processed for ML applications, alongside standardised train/test-splits.
Each structure file contains a single paired-chain or single-chain antibody, cropped to the variable region, alongside any antigen chains.

The current release contains 15810 structures from 8796 PDB IDs, corresponding to 5424 unique antibodies (SAbDab2 IDs), composed as follows: 

| antibody type                  | with antigen     | no antigen         | total structures  |
|--------------------------------|------------------|--------------------|-------------------|
| paired-chain (FV-like)         | 8230 (52.1%)     | 4137 (26.2%)       | 12367 (78.2%)     |
| single-domain heavy (VHH-like) | 2083 (13.2%)     | 1235 (7.8%)        | 3318 (21.0%)      |
| single-domain light (VL-like)  | 37 (0.2%)        | 27 (0.2%)          | 64 (0.4%)         |
| VNAR                           | 45 (0.3%)        | 16 (0.1%)          | 61 (0.4%)         |
| **total structures**           | **10395 (65.7%)** | **5415 (34.3%)**  | **15810 (100.0%)**|

Around 7.9% of structures bind antigens consisting of multiple polymer chains.

We provide two distinct train/test splits of these structures, both based on sequence similarity.

- The ab-split (`ab_split.csv`) accounts for similarity between antibody sequences. 
  To avoid data leakage via the antigen, use this only **for antigen-agnostic tasks** like prediction of antibody structure in solution.
- The ab-ag-split (`abag_split.csv`) additionally considers similarity between any protein, peptide, DNA and RNA antigen sequences. 
  Use this **for antigen-aware tasks** like antibody–antigen complex modelling.

These same splits are also available filtered down to just single-domain (VHH-like and VL-like) structures (`ab_split_sd.csv` and `abag_split_sd.csv`).
Unless otherwise noted, all splits are backward-compatible, allowing models trained on prior versions of this dataset to be benchmarked on the most recent test split.

This dataset and splits were last updated in August 2026.  
[SAbDab2](https://sabdab2.opig.stats.ox.ac.uk) itself is updated weekly.


Data Format
-----------

The dataset consists of individual `cif` files derived from X-ray and cryo-EM structures with ≤3.5 Angstrom resolution in SAbDab2. 

Each structure file contains an individual paired-chain or single-chain antibody structure, cropped to the variable region, alongside any bound antigen chains. 
Structure filenames have the form `pdb_<PDB_ID>_<VH_CHAINID>_<VL_CHAINID>.cif`. 
The missing chain of single-domain structures is represented by a `+`.

All antibody chains cover at least IMGT indices 5–123 and contain no non-standard or unknown residues. 
Polymeric antigen chains are at least five residues long and represent a single contiguous segment containing no non-standard or unknown residues.
Low-quality antigen chains violating these criteria have been removed, as have antigen chains that are themselves antibodies.
For full filtering and processing details see the SAbDab2 publication.

The `ab_split.csv` and `abag_split.csv` (and their single-domain `..._sd.csv` counterparts) define the respective train/test-split, and contain the following metadata columns for each antibody structure:

| Column Name           | Description |
|-----------------------|-------------|
| INSTANCE              | unique identifier of the structure instance |
| PDB_ID                | PDB accession where the structure originates |
| SABDAB_ID             | unique identifier of the variable region's IMGT numbering(s) |
| HEAVY_ID              | unique identifier of the heavy chain's IMGT numbering |
| LIGHT_ID              | unique identifier of the light chain's IMGT numbering |
| PDBdepo               | date of initial structure deposition in the PDB |
| SABDABdepo            | date of initial structure deposition in SAbDab2 |
| SABDABupdate          | date of last update to structure in SAbDab2 |
| method                | the structure determination method ('XRAY' or 'EM') |
| resolution            | the resolution of the structure, in Angstrom |
| type                  | the type of the original antibody structure ('FV', 'FAB', 'FAB+FC', 'VNAR', 'SD-H' (single-domain heavy, e.g. NANBODY), 'SD-L' (single-domain light)) |
| construct             | the synthetic construct the antibody instance forms part of, if any ('SCFV', 'DIABODY', 'CODV', 'SCDB' (single-chain diabody), 'DVD', 'OTHER') |
| holo                  | whether this is a structure of the bound (True) or free state (False). Note that low-quality antigen chains have been removed and so some holo structures appear to have no antigens. |
| Hchain                | author chain identifier of the antibody heavy chain (if present) |
| Lchain                | author chain identifier of the antibody light chain (if present) |
| Hseq                  | sequence of heavy-chain residues which are structurally resolved. These are always sequence-contiguous. |
| Lseq                  | sequence of light-chain residues which are structurally resolved. These are always sequence-contiguous. |
| Hseq_expected         | complete sequence of heavy-chain residues, as annotated in the PDB, including unresolved ones |
| Lseq_expected         | complete sequence of light-chain residues, as annotated in the PDB, including unresolved ones |
| VH_numerable_seq      | MGT-numerable sequence segment of the heavy chain |
| VL_numerable_seq      | MGT-numerable sequence segment of the light chain |
| VH_numbering_list     | IMGT numbering of heavy chain as a nested python list |
| VL_numbering_list     | IMGT numbering of light chain as a nested python list |
| CDRH1                 | sequence of CDRH1, defined by IMGT numbering |
| CDRH2                 | sequence of CDRH2, defined by IMGT numbering |
| CDRH3                 | sequence of CDRH3, defined by IMGT numbering |
| CDRL1                 | sequence of CDRL1, defined by IMGT numbering |
| CDRL2                 | sequence of CDRL2, defined by IMGT numbering |
| CDRL3                 | sequence of CDRL3, defined by IMGT numbering |
| concatCDRseq          | sequence of all CDRs, concatenated |
| CDRH123               | sequence of all heavy-chain CDRs, concatenated |
| CDRL123               | sequence of all light-chain CDRs, concatenated |
| agchains              | author chain identifiers of all antigen chains retained after filtering, as a slash-separated string |
| agtypes               | SAbDab2 antigen types of all antigen chains retained after filtering, as a slash-separated string (PROTEIN, SUGAR, PEPTIDE, HAPTEN, ION, DNA, RNA) |
| ag_index_list         | author-label residue indices retained in each polymer antigen chain after filtering, as a slash-separated string. These are each sequence-contiguous. |
| agresolvedseqs        | sequence of residues which are structurally resolved in each polymer antigen chain, as a slash-separated string. These are each sequence-contiguous. |
| agexpectedseqs        | complete sequence of antigen chain residues, as annotated in the PDB, including unresolved residues, as a slash-separated string. |
| VH_imgt_keep_interval | section of heavy-chain variable region retained after filtering, as a range of IMGT residue indices |
| VL_imgt_keep_interval | section of light-chain variable region retained after filtering, as a range of IMGT residue indices |
| cdr3_cluster          | cluster representative of CDRH3 cluster to which this structure was assigned |
| cdrh123_cluster       | cluster representative of CDRH123 cluster to which this structure was assigned |
| cdrl123_cluster       | cluster representative of CDRL123 cluster to which this structure was assigned |
| ab_cluster            | name of antibody cluster to which this structure was assigned. This is the final cluster assignment for the ab-split. |
| agclusters            | name of antigen cluster to which each antigen chain was assigned, as a slash-separated string |
| ab_ag_cluster         | name of the antibody-antigen cluster to which this structure was assigned. This is the final cluster assignment for the ab-ag-split. |
| ab_split              | train/test assignment of this structure, based on the antibody similarities only (the ab-split). Use for antigen-agnostic tasks only. |
| ab_ag_split           | train/test assignment of this structure, based on antibody and antigen similarities (the ab-ag-split). Use for antigen-aware tasks. |
| cdrh3_cluster_95      | cluster representative of CDRH3 cluster to which this structure was assigned, when clustering at 95% sequence identity. This column is not used for overall ab-cluster assignment (which determines the train/test split). You can use this column to sub-sample representatives from the train set during model training to ensure greater dataset balance.


Splits
------

We provide standardised 80:20 train/test splits for machine-learning tasks.
These are designed to avoid data leakage—that is, identical or very similar antibodies or antigens do not end up in both the train and test set.
There are two distinct splits of the same data. 
The ab-split is suitable for antigen-agnostic settings. 
For antigen-aware tasks, like antibody-antigen complex prediction, please use the ab-ag-split.

Briefly, split creation proceeds in two stages.

1. Antibodies and antigens are first clustered by pairwise sequence identity (number of exact non-gap matches divided by length of covered span on the longer sequence).
   - We first cluster antibodies separately on the sequences of CDRH3; 
     the concatenated CDRH1, 2 and 3; and the concatenated CDRL1, 2, and 3.
   - Polymeric antigen sequences (protein, peptide, DNA, and RNA types 
     separately) are similarly clustered.

2. Using connected-component clustering, we then assign a final cluster to each antibody structure, by grouping together any two structures that share a base cluster of any type.
   - Hence, two antibody structures always end up in the same cluster if they share the same CDRH3, CDRH123, or CDRL123 cluster. 
   - For the ab-ag-split, they also end up in the same cluster, if any of the antigen chains they bind happen to share a cluster 
     (i.e. if both structures bind a protein from protein cluster 1, for example).

Full sequence alignment and clustering parameters used for this release are given below.

| Alignment Parameter | Polypeptide Alignments    | Polynucleotide Alignments |
|---------------------|---------------------------|---------------------------|
| mode                | global (Needleman–Wunsch) | global (Needleman–Wunsch) |
| match               | +1                        | +1                        |
| mismatch            | 0                         | 0                         |
| gap-open            | -2                        | -1                        |
| gap-extend          | -1                        | -1                        |
  
---

| Clustering Threshold     | ab-ag-split | ab-split |
|--------------------------|-------------|----------|
| CDRH3 identity           | 95%         | 85%      |
| CDRH123 identity         | 100%        | 85%      |
| CDRL123 identity         | 100%        | 85%      |
| protein antigen identity | 40%         | -        |
| peptide antigen identity | 80%         | -        |
| DNA antigen identity     | 80%         | -        |
| RNA antigen identity     | 80%         | -        |


How to Cite
-----------

If you find this dataset useful, please consider citing

```bibtex
@article {Capel2026.06.16.732554,
    author = {
        Capel, Henriette L. and Vavourakis, Odysseas and
        Williams,Benjamin H. and Taylor, Christopher
    },
    title = {{SAbDab2}: The structural antibody database in the age of machine learning}, 
    elocation-id = {2026.06.16.7325543},
    year = {2026},
    doi = (10.64898/2026.06.16.732554}, 
    publisher = {Cold Spring Harbor Laboratory}, 
    URL = {https://www.biorxiv.org/content/early/2026/06/20/2026.06.16.732554}, 
    eprint = (https://www.biorxiv.org/content/early/2026/06/20/2026.06.16.732554.full.pdf}, 
    journal = {bioRxiv}}
```
