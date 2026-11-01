"""Experiment 4: how much local (target-hospital) data is needed?

For each direction and fraction f of the target cohort used for local adaptation,
the remaining (1-f) of the target cohort is the test set. Strategies:
  source_only   : model trained on source only
  recalibrated  : source model + logistic recalibration (intercept & slope) and
                  threshold re-selection on the local sample
  target_only   : model trained on the local sample only
  pooled        : model trained on source + local sample
Usage: python3 -I src/exp_update.py <jamalpur.csv> <dhaka.csv> <out_dir>
"""
import sys, os, json, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_jamalpur, load_dhaka, make_model, youden_threshold, threshold_metrics, calibration, FEATSETS

warnings.filterwarnings('ignore')
jam, dha, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
t0 = time.time()
XJ, yJ, _ = load_jamalpur(jam)
XD, yD, _, _ = load_dhaka(dha)
COH = {'Jamalpur': (XJ, yJ), 'Dhaka': (XD, yD)}
cols = FEATSETS['all8']
FRACS = [0.05, 0.10, 0.20, 0.30]
REPS = 10
MODEL = 'RF'


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


rows = []
for src, tgt in [('Jamalpur', 'Dhaka'), ('Dhaka', 'Jamalpur')]:
    Xs, ys = COH[src]; Xt, yt = COH[tgt]
    src_model = make_model(MODEL).fit(Xs[cols], ys)
    src_thr, _ = youden_threshold(MODEL, Xs[cols], ys, folds=5)
    for f in FRACS:
        for r in range(REPS):
            Xl, Xte, yl, yte = train_test_split(Xt[cols], yt, train_size=f, stratify=yt, random_state=100 + r)
            p_src_te = src_model.predict_proba(Xte)[:, 1]
            p_src_l = src_model.predict_proba(Xl)[:, 1]
            out_r = {}
            # source only
            m = threshold_metrics(yte, p_src_te, src_thr); m['auc'] = roc_auc_score(yte, p_src_te)
            m.update(calibration(yte, p_src_te)); out_r['source_only'] = m
            # recalibrated
            rc = LogisticRegression(C=1e8, max_iter=5000).fit(logit(p_src_l).reshape(-1, 1), yl)
            p_rc_l = rc.predict_proba(logit(p_src_l).reshape(-1, 1))[:, 1]
            p_rc_te = rc.predict_proba(logit(p_src_te).reshape(-1, 1))[:, 1]
            fpr, tpr, th = roc_curve(yl, p_rc_l); thr = th[np.argmax(tpr - fpr)]
            m = threshold_metrics(yte, p_rc_te, thr); m['auc'] = roc_auc_score(yte, p_rc_te)
            m.update(calibration(yte, p_rc_te)); out_r['recalibrated'] = m
            # target only
            thr, _ = youden_threshold(MODEL, Xl, yl, folds=3, seed=r)
            mt = make_model(MODEL, seed=r).fit(Xl, yl); p = mt.predict_proba(Xte)[:, 1]
            m = threshold_metrics(yte, p, thr); m['auc'] = roc_auc_score(yte, p); m.update(calibration(yte, p))
            out_r['target_only'] = m
            # pooled
            Xp = pd.concat([Xs[cols], Xl]); yp = np.r_[ys, yl]
            thr, _ = youden_threshold(MODEL, Xp, yp, folds=3, seed=r)
            mp = make_model(MODEL, seed=r).fit(Xp, yp); p = mp.predict_proba(Xte)[:, 1]
            m = threshold_metrics(yte, p, thr); m['auc'] = roc_auc_score(yte, p); m.update(calibration(yte, p))
            out_r['pooled'] = m
            for strat, mm in out_r.items():
                rows.append(dict(direction=f'{src}->{tgt}', frac=f, rep=r, strategy=strat, n_local=int(len(yl)), **mm))
        print(f'[{time.time()-t0:5.0f}s] {src}->{tgt} frac {f} done', flush=True)

df = pd.DataFrame(rows)
df.to_csv(f'{out}/local_update_raw.csv', index=False)
summ = df.groupby(['direction', 'frac', 'strategy'])[['auc', 'bal_acc', 'sens', 'spec', 'brier', 'cal_slope', 'n_local']].agg(['mean', 'std'])
summ.to_csv(f'{out}/local_update_summary.csv')
print(summ.round(3).to_string())
