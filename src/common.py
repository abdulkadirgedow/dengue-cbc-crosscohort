"""Shared data loading, harmonisation, models and metrics.

All preprocessing that learns from data (imputation, scaling, resampling,
threshold selection) is fitted on training data only.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, confusion_matrix

SEED = 42
FEATS_ALL = ['Age', 'Male', 'PLT', 'WBC', 'HCT', 'RBC', 'Lymph', 'Neut']
FEATSETS = {
    'all8': FEATS_ALL,
    'noRBC': [f for f in FEATS_ALL if f != 'RBC'],
    'PLT+WBC': ['PLT', 'WBC'],
}
MODEL_NAMES = ['LR', 'RF', 'GBM', 'Stacked']


# ----------------------------------------------------------------- data
def load_jamalpur(path):
    """Jamalpur 2024 (Mendeley 10.17632/6fsrsk3mb8.2). Returns harmonised X, y, raw df."""
    j = pd.read_csv(path)
    X = pd.DataFrame({
        'Age': j['Age'].astype(float), 'Male': (j['Gender'] == 'Male').astype(int),
        'PLT': j['Total Platelet Count(/cumm)'] / 1000.0,   # -> 10^3/uL
        'WBC': j['Total WBC count(/cumm)'] / 1000.0,         # -> 10^3/uL
        'HCT': j['HCT(%)'].astype(float), 'RBC': j['RBC'].astype(float),
        'Lymph': j['Lymphocytes(%)'].astype(float), 'Neut': j['Neutrophils(%)'].astype(float)})
    y = (j['Result'] == 'positive').astype(int).values
    return X, y, j


def load_dhaka(path):
    """Dhaka 2025 (Mendeley 10.17632/jsbmtk8hty.4). Implausible values set to NaN."""
    d = pd.read_csv(path)
    X = pd.DataFrame({
        'Age': d['Age'].astype(float), 'Male': (d['Sex'] == 'Male').astype(int),
        'PLT': d['PLT'].astype(float), 'WBC': d['WBC'].astype(float),
        'HCT': d['HCT'].astype(float), 'RBC': d['RBC'].astype(float),
        'Lymph': d['Lymph %'].astype(float), 'Neut': d['Neut %'].astype(float)})
    y = (d['Dengue NS1'] == 'Positive').astype(int).values
    cleaning = {'PLT<5': int((X.PLT < 5).sum()), 'WBC<1': int((X.WBC < 1).sum()),
                'HCT<15': int((X.HCT < 15).sum()), 'RBC>8': int((X.RBC > 8).sum())}
    X.loc[X.PLT < 5, 'PLT'] = np.nan
    X.loc[X.WBC < 1, 'WBC'] = np.nan
    X.loc[X.HCT < 15, 'HCT'] = np.nan
    X.loc[X.RBC > 8, 'RBC'] = np.nan
    return X, y, d, cleaning


# --------------------------------------------------------------- models
def make_model(name, seed=SEED):
    imp = SimpleImputer(strategy='median')
    if name == 'LR':
        return make_pipeline(imp, StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight='balanced'))
    if name == 'RF':
        return make_pipeline(imp, RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight='balanced', random_state=seed, n_jobs=-1))
    if name == 'GBM':
        return make_pipeline(imp, HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, class_weight='balanced', random_state=seed))
    if name == 'Stacked':
        base = [('rf', RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight='balanced',
                                              random_state=seed, n_jobs=-1)),
                ('gbm', HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, class_weight='balanced',
                                                       random_state=seed)),
                ('lr', make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced')))]
        return make_pipeline(imp, StackingClassifier(
            estimators=base, final_estimator=LogisticRegression(max_iter=2000, class_weight='balanced'), cv=3))
    raise ValueError(name)


def youden_threshold(model_name, X, y, seed=SEED, folds=3):
    """Decision threshold maximising Youden's J on out-of-fold training predictions."""
    p = cross_val_predict(make_model(model_name, seed), X, y,
                          cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
                          method='predict_proba')[:, 1]
    f, t, th = roc_curve(y, p)
    return float(th[np.argmax(t - f)]), p


# -------------------------------------------------------------- metrics
def threshold_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    return dict(sens=float(sens), spec=float(spec), bal_acc=float((sens + spec) / 2),
                acc=float((tp + tn) / len(y)), ppv=float(ppv), npv=float(npv))


def calibration(y, p):
    """Calibration slope (logistic recalibration) and calibration-in-the-large (intercept with slope=1)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    slope = LogisticRegression(C=1e8, max_iter=5000).fit(lp.reshape(-1, 1), y).coef_[0, 0]
    # intercept with offset: solve by Newton iterations
    a = 0.0
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(a + lp)))
        g = np.sum(y - mu); h = -np.sum(mu * (1 - mu))
        a -= g / h
    return dict(brier=float(brier_score_loss(y, p)), cal_slope=float(slope), cal_intercept=float(a),
                mean_pred=float(p.mean()), obs_rate=float(y.mean()))


def boot_ci(y, p, fn=roc_auc_score, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], p[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def net_benefit(y, p, thresholds):
    n = len(y); out = []
    for t in thresholds:
        pred = p >= t
        tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)


def auc_dir_free(y, x):
    m = ~np.isnan(x)
    a = roc_auc_score(y[m], x[m])
    return float(max(a, 1 - a))
