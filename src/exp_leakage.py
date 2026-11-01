"""Experiment 5: reproduce leakage-prone pipelines reported in the literature and
compare with their leakage-free counterparts, on the full feature set of each cohort.

Pipelines (10-fold stratified CV unless stated):
  correct_weighted : no resampling, class-weighted model
  correct_smoteenn : SMOTEENN applied inside each training fold only
  leaky_smoteenn   : SMOTEENN applied to the whole dataset, then CV   (cf. Tuhin et al. 2025)
  leaky_smote_split: SMOTE on whole dataset, then 80/20 split          (common pattern)
  leaky_ros_split  : random oversampling on whole dataset, then 80/20 split
Usage: python3 -I src/exp_leakage.py <jamalpur.csv> <dhaka.csv> <out_dir>
"""
import sys, os, json, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings('ignore')
SEED = 42
jam, dha, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
t0 = time.time()


def full_features(path, cohort):
    df = pd.read_csv(path)
    if cohort == 'Jamalpur':
        y = (df['Result'] == 'positive').astype(int).values
        X = df.drop(columns='Result').copy(); X['Gender'] = (X.Gender == 'Male').astype(int)
    else:
        y = (df['Dengue NS1'] == 'Positive').astype(int).values
        X = df[['Age', 'Sex', 'PLT', 'WBC', 'HCT', 'RBC', 'Lymph %', 'Neut %']].copy()
        X['Sex'] = (X.Sex == 'Male').astype(int)
    return X.astype(float), y


def clf(name):
    if name == 'RF':
        return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    if name == 'ET':
        return ExtraTreesClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    raise ValueError


def scores(y, pred, proba):
    return dict(acc=accuracy_score(y, pred), f1=f1_score(y, pred), bal_acc=balanced_accuracy_score(y, pred),
                auc=roc_auc_score(y, proba))


def cv_eval(X, y, make_pipe, resample_whole=None, k=10):
    if resample_whole is not None:
        X, y = resample_whole.fit_resample(SimpleImputer(strategy='median').fit_transform(X), y)
        X = pd.DataFrame(X)
    rows = []
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=SEED).split(X, y):
        m = make_pipe().fit(X.iloc[tr], y[tr])
        rows.append(scores(y[te], m.predict(X.iloc[te]), m.predict_proba(X.iloc[te])[:, 1]))
    d = pd.DataFrame(rows)
    return {c: [float(d[c].mean()), float(d[c].std())] for c in d.columns} | {'n_eval_total': int(len(y))}


def split_eval(X, y, make_pipe, sampler, before_split, reps=10):
    rows, dup = [], []
    for r in range(reps):
        Xi = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(X))
        if before_split:
            Xr, yr = sampler(r).fit_resample(Xi, y)
            Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, stratify=yr, random_state=r)
            keys = set(map(tuple, np.round(np.asarray(Xtr), 6)))
            dup.append(np.mean([tuple(v) in keys for v in np.round(np.asarray(Xte), 6)]))
        else:
            Xtr, Xte, ytr, yte = train_test_split(Xi, y, test_size=0.2, stratify=y, random_state=r)
            Xtr, ytr = sampler(r).fit_resample(Xtr, ytr)
        m = make_pipe().fit(Xtr, ytr)
        rows.append(scores(yte, m.predict(Xte), m.predict_proba(Xte)[:, 1]))
    d = pd.DataFrame(rows)
    res = {c: [float(d[c].mean()), float(d[c].std())] for c in d.columns}
    if dup:
        res['test_rows_identical_to_train'] = float(np.mean(dup))
    return res


class ROS:
    """Random oversampling of the minority class to parity (as in the authors' notebook)."""
    def __init__(self, seed): self.seed = seed
    def fit_resample(self, X, y):
        X = pd.DataFrame(X); y = np.asarray(y)
        maj = 1 if y.sum() >= len(y) - y.sum() else 0
        Xmin, Xmaj = X[y != maj], X[y == maj]
        Xup = resample(Xmin, replace=True, n_samples=len(Xmaj), random_state=self.seed)
        Xo = pd.concat([Xmaj, Xup]); yo = np.r_[np.full(len(Xmaj), maj), np.full(len(Xup), 1 - maj)]
        return Xo, yo


R = {}
for cohort, path in [('Jamalpur', jam), ('Dhaka', dha)]:
    X, y = full_features(path, cohort)
    R[cohort] = {'n_features': X.shape[1]}
    for mname in ['RF', 'ET']:
        pipe_plain = lambda: Pipeline([('imp', SimpleImputer(strategy='median')), ('m', clf(mname))])
        pipe_w = lambda: Pipeline([('imp', SimpleImputer(strategy='median')),
                                   ('m', clf(mname).set_params(class_weight='balanced'))])
        pipe_in = lambda: ImbPipeline([('imp', SimpleImputer(strategy='median')),
                                       ('s', SMOTEENN(random_state=SEED)), ('m', clf(mname))])
        R[cohort][mname] = {
            'correct_weighted': cv_eval(X, y, pipe_w),
            'correct_smoteenn': cv_eval(X, y, pipe_in),
            'leaky_smoteenn': cv_eval(X, y, pipe_plain, resample_whole=SMOTEENN(random_state=SEED)),
            'correct_smote_split': split_eval(X, y, pipe_plain, lambda r: SMOTE(random_state=r), before_split=False),
            'leaky_smote_split': split_eval(X, y, pipe_plain, lambda r: SMOTE(random_state=r), before_split=True),
            'correct_ros_split': split_eval(X, y, pipe_plain, lambda r: ROS(r), before_split=False),
            'leaky_ros_split': split_eval(X, y, pipe_plain, lambda r: ROS(r), before_split=True),
        }
        print(f'[{time.time()-t0:5.0f}s] {cohort} {mname} done', flush=True)

json.dump(R, open(f'{out}/leakage_results.json', 'w'), indent=1, default=float)
print('ALL DONE')
