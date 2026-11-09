#!/usr/bin/env bash
set -euo pipefail
J=data/jamalpur_2024.csv; D=data/dhaka_2025.csv; R=data/riya_dhaka_2023.csv
python3 -I src/data_audit.py "$J" "$D" data/multisite_fmkp6fwrrk.csv data/kalai_xrsbyjs24t.csv results
python3 -I src/exp_main.py "$J" "$D" results
python3 -I src/exp_update.py "$J" "$D" results
python3 -I src/exp_leakage.py "$J" "$D" results
python3 -I src/exp_dedup.py "$J" "$D" results
python3 -I src/exp_sensitivity2.py "$J" "$D" results
python3 -I src/exp_third_cohort.py "$J" "$D" "$R" results
python3 -I src/figures.py "$J" "$D" results figs
python3 -I src/exp_bootstrap.py results
python3 -I src/exp_twosample.py "$J" "$D" results 1000
python3 -I src/exp_subgroup.py "$J" "$D" results
python3 -I src/make_main_tables.py results tables
python3 -I src/make_supplement.py results supplement.tex
