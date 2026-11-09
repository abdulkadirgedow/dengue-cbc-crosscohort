"""Data-quality audit of four public Bangladeshi dengue haematology datasets.

Usage: python3 -I src/data_audit.py <jamalpur> <dhaka> <multisite> <kalai> <out_dir>
"""
import sys, os, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

jam, dha, mul, kal, out = sys.argv[1:6]
os.makedirs(out, exist_ok=True)


def dfree(y, x):
    m = ~pd.isna(x); a = roc_auc_score(y[m], x[m]); return float(max(a, 1 - a))


A = {}
J = pd.read_csv(jam); D = pd.read_csv(dha); M = pd.read_csv(mul); K = pd.read_csv(kal)

# ---------------- Jamalpur
yJ = (J.Result == 'positive').astype(int).values
f5 = ['Total Platelet Count(/cumm)', 'Total WBC count(/cumm)', 'Neutrophils(%)', 'Age', 'PCT(%)']
num = J.drop(columns=['Result', 'Gender'])
grp = J.groupby(list(J.columns.drop('Result'))).Result.nunique()
A['Jamalpur'] = dict(
    rows=len(J), cols=J.shape[1], pos=int(yJ.sum()), neg=int((1 - yJ).sum()), missing=int(J.isna().sum().sum()),
    exact_duplicate_rows=int(J.duplicated().sum()),
    duplicate_feature_vectors_conflicting_label=int((grp > 1).sum()),
    pct_max=float(J['PCT(%)'].max()), pct_rows_gt_1=int((J['PCT(%)'] > 1).sum()),
    rbc_distinct_values=sorted(map(int, J.RBC.unique())),
    platelet_range=[float(J['Total Platelet Count(/cumm)'].min()), float(J['Total Platelet Count(/cumm)'].max())],
    platelet_lt_100k_frac_pos=float((J.loc[yJ == 1, 'Total Platelet Count(/cumm)'] < 100000).mean()),
    platelet_lt_100k_frac_neg=float((J.loc[yJ == 0, 'Total Platelet Count(/cumm)'] < 100000).mean()),
    univariable_auc={c: dfree(yJ, J[c].values) for c in num.columns})

# ---------------- Dhaka
yD = (D['Dengue NS1'] == 'Positive').astype(int).values
A['Dhaka'] = dict(
    rows=len(D), pos=int(yD.sum()), neg=int((1 - yD).sum()), label='NS1 antigen',
    missing_ALT=int(D.ALT.isna().sum()), missing_AST=int(D.AST.isna().sum()),
    empty_columns=[c for c in D.columns if D[c].isna().all()],
    exact_duplicate_rows=int(D.drop(columns=['SN']).duplicated().sum()),
    implausible={'PLT<5 (x10^3/uL)': int((D.PLT < 5).sum()), 'WBC<1 (x10^3/uL)': int((D.WBC < 1).sum()),
                 'HCT<15%': int((D.HCT < 15).sum()), 'RBC>8 (x10^6/uL)': int((D.RBC > 8).sum()),
                 'Lymph+Neut>100%': int(((D['Lymph %'] + D['Neut %']) > 100).sum())},
    platelet_lt_100_frac_pos=float((D.loc[yD == 1, 'PLT'] < 100).mean()),
    platelet_lt_100_frac_neg=float((D.loc[yD == 0, 'PLT'] < 100).mean()),
    univariable_auc={c: dfree(yD, D[c].values) for c in ['Age', 'PLT', 'WBC', 'HCT', 'RBC', 'Lymph %', 'Neut %']})

# ---------------- Multi-site (overlap with Jamalpur)
yM = (M.Result == 'positive').astype(int).values
m2 = M.rename(columns={'Hemoglobin(Hb%)': 'Hb', 'Total WBC count(/cumm)': 'WBC', 'Platelet Count': 'PLT', 'PCT(%)': 'PCT',
                       'MCV(fl)': 'MCV', 'RDW-CV(%)': 'RDW', 'MPV(fl)': 'MPV', 'PDW(%)': 'PDW', 'Neutrophils': 'Neut',
                       'Lymphocytes': 'Lymph'})
m4 = J.rename(columns={'Hemoglobin(g/dl)': 'Hb', 'Total WBC count(/cumm)': 'WBC', 'Total Platelet Count(/cumm)': 'PLT',
                       'PCT(%)': 'PCT', 'MCV(fl)': 'MCV', 'RDW-CV(%)': 'RDW', 'MPV(fl)': 'MPV', 'PDW(%)': 'PDW',
                       'Neutrophils(%)': 'Neut', 'Lymphocytes(%)': 'Lymph', 'Gender': 'Sex'})
key = ['Sex', 'Age', 'Hb', 'Neut', 'Lymph', 'MCV', 'RDW', 'PLT', 'MPV', 'PDW', 'PCT', 'WBC']
# Many-to-many matching on the 12 shared fields; values rounded to 4 decimals (all fields are recorded with
# at most 2 decimals, so rounding only guards against float representation differences).
mk = m2[key].round(4).copy(); mk['_i'] = range(len(mk)); mk['_lab'] = M.Result.values
jk = m4[key].round(4).copy(); jk['_j'] = range(len(jk)); jk['_lab'] = J.Result.values
pairs = mk.merge(jk, on=key, suffixes=('_m', '_j'))
per_m = pairs.groupby('_i').agg(n_j=('_j', 'nunique'), labs=('_lab_j', lambda v: set(v)), lab_m=('_lab_m', 'first'))
A['Multisite'] = dict(
    rows=len(M), pos=int(yM.sum()), neg=int((1 - yM).sum()), missing=int(M.isna().sum().sum()),
    match_fields=key, match_rounding_decimals=4,
    rows_identical_to_jamalpur_on_12_fields=int(len(per_m)),
    jamalpur_rows_matched=int(pairs._j.nunique()), unique_matching_keys=int(pairs[key].drop_duplicates().shape[0]),
    multisite_rows_matching_more_than_one_jamalpur_row=int((per_m.n_j > 1).sum()),
    of_which_same_label=int(sum(r.lab_m in r.labs and len(r.labs) == 1 for r in per_m.itertuples())),
    of_which_conflicting_jamalpur_labels=int(sum(len(r.labs) > 1 for r in per_m.itertuples())),
    of_which_different_label=int(sum(r.lab_m not in r.labs for r in per_m.itertuples())),
    pct_max=float(M['PCT(%)'].max()),
    shared_extremes_with_jamalpur={c: [float(m2[c].min()), float(m2[c].max()), float(m4[c].min()), float(m4[c].max())]
                                   for c in ['Hb', 'MCV', 'RDW', 'PCT']},
    univariable_auc_PLT=dfree(yM, M['Platelet Count'].values), univariable_auc_WBC=dfree(yM, M['Total WBC count(/cumm)'].values))

# ---------------- Kalai
Kd = K.dropna(subset=['Final Output']); yK = Kd['Final Output'].astype(int).values
w = Kd['WBC Count'].values; mw = ~np.isnan(w)
best = max(((float(t), float(((w[mw] < t).astype(int) == yK[mw]).mean())) for t in np.arange(2000, 11000, 100)), key=lambda z: z[1])
vc = K['Platelet Count'].value_counts()
A['Kalai'] = dict(
    rows=len(K), pos=int(yK.sum()), neg=int((1 - yK).sum()), missing_label=int(K['Final Output'].isna().sum()),
    missing=int(K.isna().sum().sum()), sex_categories=K.Sex.value_counts().to_dict(),
    binary_coded_columns=['Differential Count', 'RBC PANEL'],
    wbc_range_pos=[float(Kd.loc[Kd['Final Output'] == 1, 'WBC Count'].min()), float(Kd.loc[Kd['Final Output'] == 1, 'WBC Count'].max())],
    wbc_range_neg=[float(Kd.loc[Kd['Final Output'] == 0, 'WBC Count'].min()), float(Kd.loc[Kd['Final Output'] == 0, 'WBC Count'].max())],
    best_single_wbc_cutoff=best,
    univariable_auc={c: dfree(yK, Kd[c].values) for c in ['Age', 'Haemoglobin', 'WBC Count', 'Platelet Count', 'PDW']},
    implausible={'Age>100': int((K.Age > 100).sum()), 'Hb>20': int((K.Haemoglobin > 20).sum()), 'PDW>100': int((K.PDW > 100).sum())},
    platelet_values_repeated_ge5=int((vc >= 5).sum()), top_repeated_platelet=vc.head(5).to_dict())

# ---------------- Jamalpur vs Dhaka (the two development/validation cohorts)
# Seven shared fields; Jamalpur platelets and WBC converted from cells/uL to 10^3/uL. Exact matches only.
jj = pd.DataFrame({'Age': J.Age, 'Sex': (J.Gender == 'Male').astype(int), 'PLT': J['Total Platelet Count(/cumm)'] / 1000,
                   'WBC': J['Total WBC count(/cumm)'] / 1000, 'HCT': J['HCT(%)'], 'Lymph': J['Lymphocytes(%)'],
                   'Neut': J['Neutrophils(%)']}).astype(float).round(2)
dd = pd.DataFrame({'Age': D.Age, 'Sex': (D.Sex == 'Male').astype(int), 'PLT': D.PLT, 'WBC': D.WBC, 'HCT': D.HCT,
                   'Lymph': D['Lymph %'], 'Neut': D['Neut %']}).astype(float).round(2)
A['Jamalpur_vs_Dhaka'] = dict(fields=list(jj.columns), rounding_decimals=2,
                              records_matching_on_all_fields=int(dd.merge(jj.drop_duplicates(), on=list(jj.columns)).shape[0]),
                              records_matching_on_PLT_WBC_HCT=int(dd[['PLT', 'WBC', 'HCT']].merge(
                                  jj[['PLT', 'WBC', 'HCT']].drop_duplicates(), on=['PLT', 'WBC', 'HCT']).shape[0]))

json.dump(A, open(f'{out}/data_audit.json', 'w'), indent=1, default=lambda o: int(o) if isinstance(o, np.integer) else float(o))
print(json.dumps(A, indent=1, default=str)[:4000])
