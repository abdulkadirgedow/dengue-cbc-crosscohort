"""Third external cohort: Riya et al. (IEEE Access 2024), 320 patients from two Dhaka hospitals,
August-September 2023, public at github.com/mritunjoychk17/Dengue-Prediction-in-Bangladesh-Using-Machine-Learning.

The dataset has no haematocrit, so models use the seven features shared by all three cohorts
(age, sex, platelets, WBC, RBC, lymphocyte %, neutrophil %), plus a six-feature version without RBC and
the platelet + WBC model. Models are trained on the full Jamalpur or Dhaka cohort and applied unchanged to
the Riya cohort, and vice versa; duplicate-aware internal CV gives each cohort's own ceiling.

Usage: python3 -I src/exp_third_cohort.py <jamalpur.csv> <dhaka.csv> <riya.csv> <results_dir>
"""
import sys, os, json, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load_jamalpur, load_dhaka, make_model, youden_threshold, threshold_metrics, duplicate_groups,
                    calibration, boot_ci, auc_dir_free, MODEL_NAMES, SEED)

warnings.filterwarnings('ignore')
jam, dha, riy, res = sys.argv[1:5]
os.makedirs(f'{res}/preds', exist_ok=True)
t0 = time.time()
log = lambda m: print(f'[{time.time() - t0:6.0f}s] {m}', flush=True)


def load_riya(path):
    r = pd.read_csv(path)
    X = pd.DataFrame({'Age': r.Age.astype(float), 'Male': (r.Gender == 'Male').astype(int),
                      'PLT': r.Platelets.astype(float), 'WBC': r.WBC.astype(float), 'RBC': r.RBC.astype(float),
                      'Lymph': r.Lymphocyte.astype(float), 'Neut': r.Neutrophil.astype(float)})
    flags = {'RBC<=0': int((X.RBC <= 0).sum()), 'WBC<1': int((X.WBC < 1).sum()), 'PLT<5': int((X.PLT < 5).sum())}
    X.loc[X.RBC <= 0, 'RBC'] = np.nan          # a recorded RBC of 0 is a missing-value code
    X.loc[X.WBC < 1, 'WBC'] = np.nan           # same rule as for Dhaka
    X.loc[X.PLT < 5, 'PLT'] = np.nan
    y = (r.Result == 'Positive').astype(int).values
    return X, y, r, flags


XJ, yJ, _ = load_jamalpur(jam)
XD, yD, _, _ = load_dhaka(dha)
XR, yR, rawR, flagsR = load_riya(riy)
COH = {'Jamalpur': (XJ, yJ), 'Dhaka': (XD, yD), 'Riya': (XR, yR)}
FS = {'shared7': ['Age', 'Male', 'PLT', 'WBC', 'RBC', 'Lymph', 'Neut'],
      'shared6_noRBC': ['Age', 'Male', 'PLT', 'WBC', 'Lymph', 'Neut'],
      'PLT+WBC': ['PLT', 'WBC']}

R = {'riya': {'n': int(len(yR)), 'pos': int(yR.sum()), 'neg': int((1 - yR).sum()), 'prev': float(yR.mean()),
              'male_frac': float(XR.Male.mean()), 'flags_set_missing': flagsR,
              'missing_cells': int(XR.isna().sum().sum()),
              'dates': [str(pd.to_datetime(rawR.Date, dayfirst=True, format='mixed').min().date()),
                        str(pd.to_datetime(rawR.Date, dayfirst=True, format='mixed').max().date())],
              'exact_duplicate_rows': int(rawR.drop(columns=['Serial', 'Date']).duplicated().sum()),
              'lymph_plus_neut_gt100': int(((rawR.Lymphocyte + rawR.Neutrophil) > 100).sum()),
              'univariable_auc': {c: auc_dir_free(yR, XR[c].values.astype(float)) for c in FS['shared7'] if c != 'Male'},
              'plt_lt100_frac_pos': float((XR.PLT[yR == 1] < 100).mean()),
              'plt_lt100_frac_neg': float((XR.PLT[yR == 0] < 100).mean()),
              'age_median': float(XR.Age.median())}}

# cohort discrimination (shared 7 features)
R['cohort_discrimination_auc'] = {}
for a, b in [('Riya', 'Dhaka'), ('Riya', 'Jamalpur')]:
    Xa, Xb = COH[a][0][FS['shared7']], COH[b][0][FS['shared7']]
    Xall = pd.concat([Xa, Xb], ignore_index=True); h = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))].astype(int)
    p = cross_val_predict(make_model('RF'), Xall, h, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                          method='predict_proba')[:, 1]
    R['cohort_discrimination_auc'][f'{a}_vs_{b}'] = float(roc_auc_score(h, p))
log(f"cohort discrimination {R['cohort_discrimination_auc']}")

# internal (duplicate-aware 5x3 CV) on each cohort with the shared feature sets
R['internal'] = {}
for fs in ['shared7', 'shared6_noRBC']:
    R['internal'][fs] = {}
    for c, (X, y) in COH.items():
        Xc = X[FS[fs]]; g = duplicate_groups(Xc)
        splits = [s for r in range(3) for s in StratifiedGroupKFold(5, shuffle=True, random_state=SEED + r).split(Xc, y, g)]
        R['internal'][fs][c] = {}
        for m in ['LR', 'RF', 'GBM']:
            aucs = []
            for tr, te in splits:
                p = make_model(m).fit(Xc.iloc[tr], y[tr]).predict_proba(Xc.iloc[te])[:, 1]
                aucs.append(roc_auc_score(y[te], p))
            R['internal'][fs][c][m] = [float(np.mean(aucs)), float(np.std(aucs))]
        log(f"internal {fs} {c}: " + ', '.join(f"{m} {v[0]:.3f}" for m, v in R['internal'][fs][c].items()))

# external: every ordered pair involving Riya
R['external'] = {}
for fs, cols in FS.items():
    R['external'][fs] = {}
    for s, t in [('Jamalpur', 'Riya'), ('Dhaka', 'Riya'), ('Riya', 'Jamalpur'), ('Riya', 'Dhaka')]:
        (Xs, ys), (Xt, yt) = COH[s], COH[t]
        out = {}
        for m in MODEL_NAMES:
            thr, _ = youden_threshold(m, Xs[cols], ys, folds=5, groups=duplicate_groups(Xs[cols]))
            p = make_model(m).fit(Xs[cols], ys).predict_proba(Xt[cols])[:, 1]
            r = dict(auc=float(roc_auc_score(yt, p)), auc_ci=boot_ci(yt, p), n_target=int(len(yt)))
            r.update(threshold_metrics(yt, p, thr)); r.update(calibration(yt, p))
            out[m] = r
            if fs == 'shared7':
                np.save(f'{res}/preds/{s}_to_{t}_shared7_{m}.npy', p)
        R['external'][fs][f'{s}->{t}'] = out
        log(f"external {fs} {s}->{t}: " + ', '.join(f"{m} {r['auc']:.3f}" for m, r in out.items()))
np.save(f'{res}/preds/yR.npy', yR)
json.dump(R, open(f'{res}/third_cohort_results.json', 'w'), indent=1)
log('ALL DONE')
