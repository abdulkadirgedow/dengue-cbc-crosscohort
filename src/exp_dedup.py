"""Sensitivity: internal and external RF/GBM performance after removing exact duplicate records.
Usage: python3 -I src/exp_dedup.py <jamalpur.csv> <dhaka.csv> <out_dir>
"""
import sys, os, json, warnings
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_jamalpur, load_dhaka, make_model, boot_ci, FEATSETS

warnings.filterwarnings('ignore')
jam, dha, out = sys.argv[1:4]
XJ, yJ, rawJ = load_jamalpur(jam)
XD, yD, rawD, _ = load_dhaka(dha)
kJ = ~rawJ.duplicated().values
kD = ~rawD.drop(columns=['SN']).duplicated().values
COH = {'Jamalpur': (XJ[kJ].reset_index(drop=True), yJ[kJ]), 'Dhaka': (XD[kD].reset_index(drop=True), yD[kD])}
cols = FEATSETS['all8']
R = {'n_after_dedup': {k: int(len(v[1])) for k, v in COH.items()}, 'internal': {}, 'external': {}}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
for c, (X, y) in COH.items():
    for m in ['RF', 'GBM']:
        a = [roc_auc_score(y[te], make_model(m).fit(X.iloc[tr][cols], y[tr]).predict_proba(X.iloc[te][cols])[:, 1])
             for tr, te in cv.split(X, y)]
        R['internal'][f'{c}|{m}'] = [float(np.mean(a)), float(np.std(a))]
for s, t in [('Jamalpur', 'Dhaka'), ('Dhaka', 'Jamalpur')]:
    for m in ['RF', 'GBM']:
        p = make_model(m).fit(COH[s][0][cols], COH[s][1]).predict_proba(COH[t][0][cols])[:, 1]
        R['external'][f'{s}->{t}|{m}'] = [float(roc_auc_score(COH[t][1], p))] + boot_ci(COH[t][1], p)
json.dump(R, open(f'{out}/dedup_results.json', 'w'), indent=1)
print(json.dumps(R, indent=1))
