"""Additional sensitivity analyses requested in review.

  diff_le100 : exclude records whose lymphocyte + neutrophil percentages sum to more than 100
               (46 Dhaka, 5 Jamalpur)
  raw_extremes: keep the flagged extreme Dhaka values (platelets <5, WBC <1, HCT <15, RBC >8)
               instead of setting them to missing
  unweighted : fit every model without class weights

For each variant: bidirectional external validation (all eight features) and duplicate-aware
internal cross-validation (5 folds x 3 repeats) for every cohort the variant changes.

Usage: python3 -I src/exp_sensitivity2.py <jamalpur.csv> <dhaka.csv> <results_dir>
"""
import sys, os, json, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load_jamalpur, load_dhaka, make_model, youden_threshold, threshold_metrics, duplicate_groups,
                    calibration, boot_ci, FEATS_ALL, MODEL_NAMES, SEED)

warnings.filterwarnings('ignore')
jam, dha, res = sys.argv[1:4]
t0 = time.time()
log = lambda m: print(f'[{time.time() - t0:6.0f}s] {m}', flush=True)


def external(Xs, ys, Xt, yt, weighted):
    out = {}
    for mname in MODEL_NAMES:
        thr, _ = youden_threshold(mname, Xs, ys, folds=5, groups=duplicate_groups(Xs), weighted=weighted)
        p = make_model(mname, weighted=weighted).fit(Xs, ys).predict_proba(Xt)[:, 1]
        r = dict(auc=float(roc_auc_score(yt, p)), auc_ci=boot_ci(yt, p), n_target=int(len(yt)))
        r.update(threshold_metrics(yt, p, thr)); r.update(calibration(yt, p))
        out[mname] = r
    return out


def internal(X, y, weighted):
    g = duplicate_groups(X)
    splits = [s for r in range(3) for s in StratifiedGroupKFold(5, shuffle=True, random_state=SEED + r).split(X, y, g)]
    out = {}
    for mname in MODEL_NAMES:
        rows = []
        for tr, te in splits:
            thr, _ = youden_threshold(mname, X.iloc[tr], y[tr], folds=3, groups=g[tr], weighted=weighted)
            p = make_model(mname, weighted=weighted).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
            r = threshold_metrics(y[te], p, thr); r['auc'] = roc_auc_score(y[te], p); r.update(calibration(y[te], p))
            rows.append(r)
        d = pd.DataFrame(rows)
        out[mname] = {k: [float(d[k].mean()), float(d[k].std())] for k in d.columns}
    return out


XJ, yJ, _ = load_jamalpur(jam)
XD, yD, _, _ = load_dhaka(dha)
XDr, yDr, _, _ = load_dhaka(dha, flag_extremes=False)
okJ = (XJ.Lymph + XJ.Neut <= 100).values
okD = (XDr.Lymph + XDr.Neut <= 100).values          # judged on the recorded values

variants = {
    'diff_le100': dict(J=(XJ[okJ].reset_index(drop=True), yJ[okJ]), D=(XD[okD].reset_index(drop=True), yD[okD]),
                       weighted=True, internal=['Jamalpur', 'Dhaka'],
                       excluded={'Jamalpur': int((~okJ).sum()), 'Dhaka': int((~okD).sum())}),
    'raw_extremes': dict(J=(XJ, yJ), D=(XDr, yDr), weighted=True, internal=['Dhaka']),
    'unweighted': dict(J=(XJ, yJ), D=(XD, yD), weighted=False, internal=['Jamalpur', 'Dhaka']),
}
R = {}
for name, v in variants.items():
    (Xj, yj), (Xd, yd), w = v['J'], v['D'], v['weighted']
    R[name] = {'n': {'Jamalpur': int(len(yj)), 'Dhaka': int(len(yd))}, 'excluded': v.get('excluded', {}),
               'external': {'Jamalpur->Dhaka': external(Xj[FEATS_ALL], yj, Xd[FEATS_ALL], yd, w),
                            'Dhaka->Jamalpur': external(Xd[FEATS_ALL], yd, Xj[FEATS_ALL], yj, w)},
               'internal': {}}
    log(f"{name} external: " + '; '.join(f"{d}: " + ', '.join(f"{m} {r['auc']:.3f}" for m, r in e.items())
                                        for d, e in R[name]['external'].items()))
    for c in v['internal']:
        X, y = (Xj, yj) if c == 'Jamalpur' else (Xd, yd)
        R[name]['internal'][c] = internal(X[FEATS_ALL], y, w)
        log(f"{name} internal {c}: " + ', '.join(f"{m} {r['auc'][0]:.3f}" for m, r in R[name]['internal'][c].items()))
    json.dump(R, open(f'{res}/sensitivity2_results.json', 'w'), indent=1)
log('ALL DONE')
