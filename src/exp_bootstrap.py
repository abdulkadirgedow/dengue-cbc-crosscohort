"""Bootstrap 95% CIs for every external-validation metric (all 8 features), using the saved
external predictions and the training-selected thresholds stored in main_results.json.

Usage: python3 -I src/exp_bootstrap.py <results_dir>
"""
import sys, os, json, warnings
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import threshold_metrics, calibration

warnings.filterwarnings('ignore')
res = sys.argv[1]
M = json.load(open(f'{res}/main_results.json'))
P = f'{res}/preds'
y = {'Jamalpur': np.load(f'{P}/yJ.npy'), 'Dhaka': np.load(f'{P}/yD.npy')}
B = 1000
out = {}
for src, tgt in [('Jamalpur', 'Dhaka'), ('Dhaka', 'Jamalpur')]:
    yt = y[tgt]
    for m in ['LR', 'RF', 'GBM', 'Stacked']:
        p = np.load(f'{P}/{src}_to_{tgt}_all8_{m}.npy')
        thr = M['external'][f'all8|{src}->{tgt}'][m]['thr']
        rng = np.random.default_rng(42)
        rows = []
        for _ in range(B):
            i = rng.integers(0, len(yt), len(yt))
            if yt[i].min() == yt[i].max():
                continue
            r = threshold_metrics(yt[i], p[i], thr); r.update(calibration(yt[i], p[i])); r['auc'] = roc_auc_score(yt[i], p[i])
            rows.append(r)
        point = threshold_metrics(yt, p, thr); point.update(calibration(yt, p)); point['auc'] = roc_auc_score(yt, p)
        keys = ['auc', 'bal_acc', 'sens', 'spec', 'ppv', 'npv', 'brier', 'cal_slope', 'cal_intercept']
        out[f'{src}->{tgt}|{m}'] = {k: [float(point[k]), float(np.percentile([r[k] for r in rows], 2.5)),
                                        float(np.percentile([r[k] for r in rows], 97.5))] for k in keys}
    print(src, '->', tgt, 'done', flush=True)
json.dump(out, open(f'{res}/external_bootstrap_ci.json', 'w'), indent=1)
for k, v in out.items():
    print(k, ' '.join(f"{m}={a:.3f}[{l:.3f},{h:.3f}]" for m, (a, l, h) in v.items()))
