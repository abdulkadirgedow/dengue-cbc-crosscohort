"""Subgroup performance of the external models (TRIPOD+AI items 14 and 23a).

Re-uses the saved external predictions (results/preds) of the models trained on the
full source cohort, so no model is refitted. Reports AUC with 95% bootstrap CI and
calibration-in-the-large by sex and age band in each target cohort.

Usage: python3 -I src/exp_subgroup.py <jamalpur.csv> <dhaka.csv> <results_dir>
"""
import sys, os, json, warnings
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_jamalpur, load_dhaka, boot_ci, calibration, MODEL_NAMES

warnings.filterwarnings('ignore')
jam, dha, res = sys.argv[1:4]
XJ, yJ, _ = load_jamalpur(jam)
XD, yD, _, _ = load_dhaka(dha)
assert np.array_equal(np.load(f'{res}/preds/yJ.npy'), yJ)
assert np.array_equal(np.load(f'{res}/preds/yD.npy'), yD)

BANDS = [('<15', 0, 14.999), ('15-60', 15, 60), ('>60', 60.001, 200)]
out = {}
for src, tgt, X, y in [('Jamalpur', 'Dhaka', XD, yD), ('Dhaka', 'Jamalpur', XJ, yJ)]:
    groups = {'Female': (X.Male == 0).values, 'Male': (X.Male == 1).values}
    for lab, lo, hi in BANDS:
        groups[f'Age {lab}'] = ((X.Age >= lo) & (X.Age <= hi)).values
    key = f'{src}->{tgt}'; out[key] = {}
    for g, m in groups.items():
        row = {'n': int(m.sum()), 'n_pos': int(y[m].sum())}
        for mname in MODEL_NAMES:
            p = np.load(f'{res}/preds/{src}_to_{tgt}_all8_{mname}.npy')
            if row['n_pos'] < 10 or row['n'] - row['n_pos'] < 10:
                row[mname] = None; continue
            c = calibration(y[m], p[m])
            row[mname] = {'auc': float(roc_auc_score(y[m], p[m])), 'auc_ci': boot_ci(y[m], p[m]),
                          'citl': float(c['cal_intercept']), 'slope': float(c['cal_slope'])}
        out[key][g] = row
        print(key, g, row['n'], row['n_pos'],
              ', '.join(f"{k} {v['auc']:.3f}" for k, v in row.items() if isinstance(v, dict)))

json.dump(out, open(f'{res}/subgroup_results.json', 'w'), indent=1)
