# Do blood-count-based dengue classifiers generalize across clinical cohorts?

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](requirements.txt)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23851004.svg)](https://doi.org/10.5281/zenodo.23851004)

Code, results and figures for the paper:

> A. S. Gedow, M. M. U. Islam and A. H. Mohamed, "Do Blood-Count-Based Dengue Classifiers Generalize Across Clinical Cohorts? A Leakage-Free External Validation and Data-Integrity Audit of Public Bangladeshi Datasets," manuscript, 2026. (Update with the journal reference once published.)

The study trains dengue classifiers on routine complete blood count (CBC) data from one Bangladeshi cohort and tests them on another, in both directions. It also audits four public datasets and measures how much common data-leakage mistakes inflate reported accuracy.

## Main findings

| | Jamalpur 2024 | Dhaka 2025 |
|---|---|---|
| Best internal AUC (leakage-free CV) | 0.68 | 0.95 |
| Best external AUC (model trained on the other cohort) | 0.56 | 0.79 |
| Random-forest accuracy, correct vs. leaky resampling | 72–76% vs. 83–91% | 85–91% vs. 93–97% |

- Models trained in Dhaka fall to near chance in Jamalpur, in every sensitivity analysis and every sex and age subgroup.
- One public dataset shares 220 records with Jamalpur; in another, a single WBC cut-off reproduces 99.9% of labels.
- Resampling before the train/test split reproduces the 90%+ accuracies reported in the literature.

## Repository structure

```
.
├── src/                 analysis code (one script per experiment)
│   └── common.py        data loading, harmonisation, models, metrics
├── data/                put the four downloaded CSV files here (not included)
│   ├── README.md        download instructions
│   └── CHECKSUMS.md5    checksums of the exact file versions used
├── results/             all numerical results reported in the paper (JSON / CSV)
│   └── preds/           saved external-validation predictions
├── figs/                Figures 2–8 (PDF and PNG)
├── run_all.sh           reruns the full analysis
├── requirements.txt     exact package versions
├── CITATION.cff         citation metadata
└── LICENSE              MIT
```

## Quick start

```bash
git clone https://github.com/abdulkadirgedow/dengue-cbc-crosscohort.git
cd dengue-cbc-crosscohort
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the four datasets as described in [`data/README.md`](data/README.md), check them with

```bash
md5sum -c data/CHECKSUMS.md5
```

and run

```bash
bash run_all.sh     # about 40 minutes on 2 CPU cores
```

Outputs go to `results/` and `figs/`. The results in the paper are already in `results/`, so you can inspect them without rerunning anything.

## From paper to code

| Paper item | Script | Output |
|---|---|---|
| Table 1 (literature audit) | — (manual, verified) | `results/literature_audit.json` |
| Table 3, Section 3.1 (data audit) | `src/data_audit.py` | `results/data_audit.json` |
| Tables 4–6, Figure 3 (internal, external, sensitivity) | `src/exp_main.py` | `results/main_results.json` |
| Table S3 (bootstrap CIs) | `src/exp_bootstrap.py` | `results/external_bootstrap_ci.json` |
| Table S8 (two-sample bootstrap) | `src/exp_twosample.py` | `results/external_twosample_bootstrap.json` |
| Table S9 (subgroups by sex and age) | `src/exp_subgroup.py` | `results/subgroup_results.json` |
| Table S7 (duplicates removed) | `src/exp_dedup.py` | `results/dedup_results.json` |
| Figure 7, Table S5 (local updating) | `src/exp_update.py` | `results/local_update_*.csv` |
| Table 7, Figure 8, Table S6 (leakage) | `src/exp_leakage.py` | `results/leakage_results.json` |
| Figures 2–8, SHAP values | `src/figures.py` | `figs/`, `results/shap_importance.json` |
| Supplementary Tables S1–S10 | `src/make_supplement.py` | `supplement.tex` |

Every script can also be run on its own; the usage line is at the top of each file.

## How leakage is avoided

- Imputation, scaling, resampling and threshold selection are fitted on training data only, inside each fold.
- Class imbalance is handled with class weights, not resampling, in the main analyses.
- Decision thresholds maximise Youden's J on out-of-fold training predictions.
- Hyperparameters are fixed in advance; all random seeds are fixed (`SEED = 42`).

## Citation

If you use this code, please cite the paper and the archived code: doi:[10.5281/zenodo.23851004](https://doi.org/10.5281/zenodo.23851004). The "Cite this repository" button on GitHub (from [`CITATION.cff`](CITATION.cff)) gives the full reference.

## Licence

Code is released under the MIT licence. The datasets belong to their original authors and are available under CC BY 4.0 from Mendeley Data; please cite them as listed in `data/README.md`.

## Contact

Abdulkadir Salad Gedow, Department of Software Engineering, Daffodil International University, Dhaka, Bangladesh. a.gedowscientist@gmail.com
