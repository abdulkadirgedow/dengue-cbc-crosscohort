"""Two-sample bootstrap for external AUC and calibration slope: resample BOTH the source cohort
(and refit the model) and the target cohort, so intervals also reflect source-model uncertainty.

Usage: python3 -I src/exp_twosample.py <jamalpur.csv> <dhaka.csv> <out_dir> [B]
"""
import sys, os, json, time, warnings
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_jamalpur, load_dhaka, make_model, calibration, FEATSETS

warnings.filterwarnings('ignore')
jam, dha, out = sys.argv[1:4]
B = int(sys.argv[4]) if len(sys.argv) > 4 else 200
XJ, yJ, _ = load_jamalpur(jam)
XD, yD, _, _ = load_dhaka(dha)
COH = {'Jamalpur': (XJ, yJ), 'Dhaka': (XD, yD)}
cols = FEATSETS['all8']
t0 = time.time()
R = {'B': B, 'note': 'source resampled and model refitted in each replicate; target resampled independently'}
for src, tgt in [('Jamalpur', 'Dhaka'), ('Dhaka', 'Jamalpur')]:
    Xs, ys = COH[src]; Xt, yt = COH[tgt]
    for m in ['LR', 'RF', 'GBM']:
        rng = np.random.default_rng(2024)
        aucs, slopes = [], []
        for b in range(B):
            i = rng.integers(0, len(ys), len(ys)); j = rng.integers(0, len(yt), len(yt))
            model = make_model(m, seed=b).fit(Xs.iloc[i][cols], ys[i])
            p = model.predict_proba(Xt.iloc[j][cols])[:, 1]
            aucs.append(roc_auc_score(yt[j], p)); slopes.append(calibration(yt[j], p)['cal_slope'])
        R[f'{src}->{tgt}|{m}'] = {'auc': [float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
                                  'cal_slope': [float(np.mean(slopes)), float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))]}
        print(f'[{time.time()-t0:5.0f}s] {src}->{tgt} {m}: AUC {R[f"{src}->{tgt}|{m}"]["auc"]}', flush=True)
json.dump(R, open(f'{out}/external_twosample_bootstrap.json', 'w'), indent=1)
print('ALL DONE')
