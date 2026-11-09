"""Experiment 1-3: internal validation, cross-hospital external validation
(with sensitivity analyses) and cohort-discrimination test.

Usage: python3 -I src/exp_main.py <jamalpur.csv> <dhaka.csv> <out_dir>
"""
import sys, os, json, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load_jamalpur, load_dhaka, make_model, youden_threshold, threshold_metrics, duplicate_groups,
                    calibration, boot_ci, auc_dir_free, FEATS_ALL, FEATSETS, MODEL_NAMES, SEED)

warnings.filterwarnings('ignore')
jam, dha, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True); os.makedirs(f'{out}/preds', exist_ok=True)
t0 = time.time()
log = lambda m: print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

XJ, yJ, _ = load_jamalpur(jam)
XD, yD, _, cleaning = load_dhaka(dha)
COH = {'Jamalpur': (XJ, yJ), 'Dhaka': (XD, yD)}
R = {'cleaning_dhaka': cleaning}

# ------------------------------------------------ cohort description
desc = {}
for k, (X, y) in COH.items():
    d = {'n': int(len(y)), 'pos': int(y.sum()), 'neg': int((1 - y).sum()), 'prev': float(y.mean()),
         'male_frac': float(X.Male.mean()), 'missing_cells': int(X.isna().sum().sum())}
    for c in FEATS_ALL:
        if c == 'Male':
            continue
        for lab, m in [('pos', y == 1), ('neg', y == 0)]:
            s = X.loc[m, c]
            d[f'{c}_{lab}'] = [float(s.median()), float(s.quantile(.25)), float(s.quantile(.75))]
    d['univariable_auc'] = {c: auc_dir_free(y, X[c].values.astype(float)) for c in FEATS_ALL}
    desc[k] = d
R['cohorts'] = desc
log('description done')

# ------------------------------------------------ cohort discrimination
Xall = pd.concat([XJ, XD], ignore_index=True)
hosp = np.r_[np.zeros(len(XJ)), np.ones(len(XD))].astype(int)
R['cohort_discrimination_auc'] = {}
for fs, cols in FEATSETS.items():
    p = cross_val_predict(make_model('RF'), Xall[cols], hosp, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                          method='predict_proba')[:, 1]
    R['cohort_discrimination_auc'][fs] = float(roc_auc_score(hosp, p))
log('cohort discrimination done')

# ------------------------------------------------ internal validation
# Duplicate-aware repeated CV: records with identical feature vectors stay in the same fold
# (StratifiedGroupKFold, 5 folds x 3 repeats), both for evaluation and for threshold selection.
R['internal'] = {}
for cname, (X, y) in COH.items():
    R['internal'][cname] = {}
    g = duplicate_groups(X)
    splits = [s for r in range(3) for s in StratifiedGroupKFold(5, shuffle=True, random_state=SEED + r).split(X, y, g)]
    for mname in MODEL_NAMES:
        rows = []
        for tr, te in splits:
            thr, _ = youden_threshold(mname, X.iloc[tr], y[tr], folds=3, groups=g[tr])
            m = make_model(mname).fit(X.iloc[tr], y[tr])
            p = m.predict_proba(X.iloc[te])[:, 1]
            r = threshold_metrics(y[te], p, thr); r['auc'] = roc_auc_score(y[te], p); r['thr'] = thr
            r.update(calibration(y[te], p))
            rows.append(r)
        df = pd.DataFrame(rows)
        R['internal'][cname][mname] = {k: [float(df[k].mean()), float(df[k].std())] for k in df.columns}
        log(f'internal {cname} {mname}: AUC {df.auc.mean():.3f}')

# ------------------------------------------------ external validation + sensitivity
def external(Xs, ys, Xt, yt, cols, tag, save=False):
    res = {}
    for mname in MODEL_NAMES:
        thr, poof = youden_threshold(mname, Xs[cols], ys, folds=5, groups=duplicate_groups(Xs[cols]))
        m = make_model(mname).fit(Xs[cols], ys)
        p = m.predict_proba(Xt[cols])[:, 1]
        r = dict(auc=float(roc_auc_score(yt, p)), auc_ci=boot_ci(yt, p), thr=thr,
                 source_oof_auc=float(roc_auc_score(ys, poof)), n_target=int(len(yt)))
        r.update(threshold_metrics(yt, p, thr)); r.update(calibration(yt, p))
        res[mname] = r
        if save:
            np.save(f'{out}/preds/{tag}_{mname}.npy', p)
    return res


R['external'] = {}
for fs, cols in FEATSETS.items():
    for src, tgt in [('Jamalpur', 'Dhaka'), ('Dhaka', 'Jamalpur')]:
        key = f'{fs}|{src}->{tgt}'
        R['external'][key] = external(*COH[src], *COH[tgt], cols, f'{src}_to_{tgt}_{fs}', save=(fs == 'all8'))
        log(f'external {key}: ' + ', '.join(f"{m} {v['auc']:.3f}" for m, v in R['external'][key].items()))

# age-restricted sensitivity analysis (common age range 15-60)
AGE = (15, 60)
mJ = (XJ.Age >= AGE[0]) & (XJ.Age <= AGE[1]); mD = (XD.Age >= AGE[0]) & (XD.Age <= AGE[1])
R['age_restricted'] = {'range': AGE, 'n_Jamalpur': int(mJ.sum()), 'n_Dhaka': int(mD.sum()),
                       'median_age_J': float(XJ.Age[mJ].median()), 'median_age_D': float(XD.Age[mD].median())}
for fs in ['all8', 'noRBC']:
    cols = FEATSETS[fs]
    R['age_restricted'][f'{fs}|Jamalpur->Dhaka'] = external(XJ[mJ.values], yJ[mJ.values], XD[mD.values], yD[mD.values], cols, 'x')
    R['age_restricted'][f'{fs}|Dhaka->Jamalpur'] = external(XD[mD.values], yD[mD.values], XJ[mJ.values], yJ[mJ.values], cols, 'x')
    log(f'age-restricted {fs} done')

np.save(f'{out}/preds/yJ.npy', yJ); np.save(f'{out}/preds/yD.npy', yD)
json.dump(R, open(f'{out}/main_results.json', 'w'), indent=1)
log('ALL DONE')
