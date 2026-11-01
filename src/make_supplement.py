"""Generate the supplementary tables (LaTeX) directly from the result files.
Usage: python3 -I src/make_supplement.py <results_dir> <out_tex>
"""
import sys, json
import pandas as pd

res, out = sys.argv[1:3]
M = json.load(open(f'{res}/main_results.json'))
B = json.load(open(f'{res}/external_bootstrap_ci.json'))
L = json.load(open(f'{res}/leakage_results.json'))
D = json.load(open(f'{res}/dedup_results.json'))
U = pd.read_csv(f'{res}/local_update_raw.csv')

ci = lambda v, d=3: f"{v[0]:.{d}f} ({v[1]:.{d}f}--{v[2]:.{d}f})"
T = []
T.append(r"""\documentclass[10pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{mathptmx}
\usepackage[margin=0.8in]{geometry}
\usepackage{booktabs,multirow,array,tabularx}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage{xr-hyper}
\usepackage[hidelinks]{hyperref}
\externaldocument[M-]{main}
\renewcommand{\thetable}{S\arabic{table}}
\title{Supplementary Material\\[4pt]\large Do Blood-Count-Based Dengue Classifiers Generalize Across Clinical Cohorts? A Leakage-Free External Validation and Data-Integrity Audit of Public Bangladeshi Datasets}
\author{Abdulkadir Salad Gedow, Ahmed Hassan Mohamed, Md.\ Muksit Ul Islam}
\date{}
\begin{document}
\maketitle
All values in this supplement are generated automatically from the result files in the accompanying code repository (\texttt{src/make\_supplement.py}).
""")

# S1 settings
T.append(r"""
\begin{table}[htbp]\centering\caption{Model settings and software environment. All random seeds were 42 unless stated.}\small
\begin{tabularx}{\linewidth}{@{}p{3.6cm}X@{}}\toprule
Component & Setting \\\midrule
Imputation & Median of the training data (\texttt{SimpleImputer}) \\
Logistic regression (LR) & Standardised inputs; L2 penalty, $C=1$; \texttt{class\_weight='balanced'}; max.\ 2,000 iterations \\
Random forest (RF) & 300 trees; minimum leaf size 3; Gini criterion; \texttt{class\_weight='balanced'} \\
Gradient boosting (GBM) & \texttt{HistGradientBoostingClassifier}; 200 iterations; learning rate 0.05; \texttt{class\_weight='balanced'} \\
Stacked ensemble & Base learners: RF (200 trees, min.\ leaf 3, balanced), GBM (150 iterations, learning rate 0.05, balanced), LR (standardised, balanced); meta-learner LR (balanced); inner 3-fold CV for base-learner probabilities \\
Threshold & Youden's $J$ maximised on out-of-fold training predictions (3-fold within internal CV folds; 5-fold for external validation) \\
Internal validation & Stratified 5-fold CV, 3 repeats (15 folds) \\
External CIs & 1,000 bootstrap resamples of the target cohort; percentile intervals; threshold fixed \\
Local updating & RF; local fractions 5, 10, 20, 30\%; 10 stratified splits (seeds 100--109) \\
Leakage experiments & RF and Extra Trees with 300 trees (other settings default); SMOTE $k=5$; SMOTEENN = SMOTE ($k=5$) + ENN ($k=3$); 80/20 stratified splits with seeds 0--9; stratified 10-fold CV (seed 42) \\
SHAP & TreeExplainer on RF trained on each full cohort; random sample of 600 patients (seed 0) \\
Software & Python 3.13.18; scikit-learn 1.9.1; imbalanced-learn 0.14.2; SHAP 0.54.1; NumPy 2.5.3; pandas 3.0.5; matplotlib 3.11.2 \\
\bottomrule\end{tabularx}\end{table}
""")

# S2 cohort characteristics
lab = {'Age': 'Age (years)', 'PLT': r'Platelets ($10^3$/\textmu L)', 'WBC': r'WBC ($10^3$/\textmu L)', 'HCT': r'Haematocrit (\%)',
       'RBC': r'RBC ($10^6$/\textmu L)', 'Lymph': r'Lymphocytes (\%)', 'Neut': r'Neutrophils (\%)'}
miss = {'Jamalpur': {k: '0' for k in lab}, 'Dhaka': {'Age': '0', 'PLT': '10 (0.8)', 'WBC': '24 (1.8)', 'HCT': '2 (0.2)',
                                                       'RBC': '3 (0.2)', 'Lymph': '0', 'Neut': '0'}}
rows = []
for f, l in lab.items():
    cells = []
    for c in ['Jamalpur', 'Dhaka']:
        d = M['cohorts'][c]
        for g in ['pos', 'neg']:
            m, q1, q3 = d[f'{f}_{g}']
            cells.append(f"{m:.4g} ({q1:.4g}--{q3:.4g})")
        cells.append(f"{d['univariable_auc'][f]:.2f}")
        cells.append(miss[c][f])
    rows.append(l + ' & ' + ' & '.join(cells) + r' \\')
cj, cd = M['cohorts']['Jamalpur'], M['cohorts']['Dhaka']
T.append(r"""
\begin{table}[htbp]\centering\caption{Cohort characteristics after harmonisation: median (IQR) by dengue status, direction-free univariable AUC (Eq.~1 of the main text) and number (\%) of missing values after implausible values were set to missing.}\scriptsize
\begin{tabular}{@{}l cccc cccc@{}}\toprule
 & \multicolumn{4}{c}{Jamalpur 2024 (""" + f"{cj['pos']} pos., {cj['neg']} neg.)" + r"""} & \multicolumn{4}{c}{Dhaka 2025 (""" + f"{cd['pos']} pos., {cd['neg']} neg.)" + r"""}\\
\cmidrule(lr){2-5}\cmidrule(l){6-9}
Feature & Positive & Negative & AUC & Missing & Positive & Negative & AUC & Missing \\\midrule
""" + '\n'.join(rows) + r"""
Male sex (\%) & \multicolumn{2}{c}{""" + f"{cj['male_frac']*100:.1f}" + r"""} & """ + f"{cj['univariable_auc']['Male']:.2f}" + r""" & 0 & \multicolumn{2}{c}{""" + f"{cd['male_frac']*100:.1f}" + r"""} & """ + f"{cd['univariable_auc']['Male']:.2f}" + r""" & 0 \\
\bottomrule\end{tabular}\end{table}
""")

# S3 external with CIs
r3 = []
for d in ['Jamalpur->Dhaka', 'Dhaka->Jamalpur']:
    for i, m in enumerate(['LR', 'RF', 'GBM', 'Stacked']):
        v = B[f'{d}|{m}']
        lead = (r'\multirow{4}{*}{' + d.replace('->', r' $\to$ ') + '}') if i == 0 else ''
        r3.append(f"{lead} & {m} & {ci(v['auc'])} & {ci(v['bal_acc'])} & {ci(v['sens'])} & {ci(v['spec'])} \\\\")
    r3.append(r'\midrule')
r3b = []
for d in ['Jamalpur->Dhaka', 'Dhaka->Jamalpur']:
    for i, m in enumerate(['LR', 'RF', 'GBM', 'Stacked']):
        v = B[f'{d}|{m}']
        lead = (r'\multirow{4}{*}{' + d.replace('->', r' $\to$ ') + '}') if i == 0 else ''
        r3b.append(f"{lead} & {m} & {ci(v['ppv'])} & {ci(v['npv'])} & {ci(v['brier'])} & {ci(v['cal_slope'],2)} & {ci(v['cal_intercept'],2)} \\\\")
    r3b.append(r'\midrule')
T.append(r"""
\begin{table}[htbp]\centering\caption{External validation (all eight features) with 95\% bootstrap confidence intervals (1,000 resamples of the target cohort; fitted source model and training-selected threshold held fixed, so the intervals are conditional on the fitted source model; Table S8 gives intervals that also reflect source-model uncertainty). Part A: discrimination and threshold measures. Part B: predictive values and calibration.}\scriptsize
\begin{tabular}{@{}llcccc@{}}\toprule
\multicolumn{6}{@{}l}{\textit{A.}}\\
Direction & Model & AUC & Balanced accuracy & Sensitivity & Specificity \\\midrule
""" + '\n'.join(r3[:-1]) + r"""
\bottomrule\end{tabular}

\vspace{6pt}
\begin{tabular}{@{}llccccc@{}}\toprule
\multicolumn{7}{@{}l}{\textit{B.}}\\
Direction & Model & PPV & NPV & Brier score & Calibration slope & Calibration intercept \\\midrule
""" + '\n'.join(r3b[:-1]) + r"""
\bottomrule\end{tabular}\end{table}
""")

# S4 sensitivity with CIs
rows4 = []
sens_rows = [('All 8 features', 'external', 'all8'), ('Without RBC', 'external', 'noRBC'), ('Platelets + WBC only', 'external', 'PLT+WBC'),
             ('Ages 15--60', 'age_restricted', 'all8'), ('Ages 15--60, without RBC', 'age_restricted', 'noRBC')]
for d in ['Jamalpur->Dhaka', 'Dhaka->Jamalpur']:
    rows4.append(r'\multicolumn{5}{@{}l}{\textit{' + d.replace('->', r' $\to$ ') + r'}}\\')
    for lab_, sec, fs in sens_rows:
        v = M[sec][f'{fs}|{d}']
        rows4.append(lab_ + ' & ' + ' & '.join(f"{v[m]['auc']:.3f} ({v[m]['auc_ci'][0]:.3f}--{v[m]['auc_ci'][1]:.3f})" for m in ['LR', 'RF', 'GBM', 'Stacked']) + r' \\')
    rows4.append(r'\midrule')
T.append(r"""
\begin{table}[htbp]\centering\caption{External AUC (95\% bootstrap CI) under sensitivity analyses. The age-restricted analyses use ages 15--60 in both cohorts (Jamalpur $n=1{,}300$; Dhaka $n=1{,}200$).}\scriptsize
\begin{tabular}{@{}lcccc@{}}\toprule
Analysis & LR & RF & GBM & Stacked \\\midrule
""" + '\n'.join(rows4[:-1]) + r"""
\bottomrule\end{tabular}\end{table}
""")

# S5 local updating
g = U.groupby(['direction', 'frac', 'strategy'])
names = {'source_only': 'Source model only', 'recalibrated': 'Source model, recalibrated', 'target_only': 'Local data only', 'pooled': 'Source + local data'}
rows5 = []
for d in ['Jamalpur->Dhaka', 'Dhaka->Jamalpur']:
    rows5.append(r'\multicolumn{7}{@{}l}{\textit{' + d.replace('->', r' $\to$ ') + r'}}\\')
    for f in [0.05, 0.10, 0.20, 0.30]:
        for k, (s, nm) in enumerate(names.items()):
            x = g.get_group((d, f, s))
            n = int(x.n_local.iloc[0])
            lead = f"{int(f*100)}\\% ($n={n}$)" if k == 0 else ''
            rows5.append(f"{lead} & {nm} & {x.auc.mean():.3f} $\\pm$ {x.auc.std():.3f} & {x.bal_acc.mean():.3f} $\\pm$ {x.bal_acc.std():.3f} & {x.brier.mean():.3f} & {x.cal_slope.mean():.2f} $\\pm$ {x.cal_slope.std():.2f} \\\\")
    rows5.append(r'\midrule')
T.append(r"""
\begin{table}[htbp]\centering\caption{Local updating with random forest: mean $\pm$ SD over 10 random splits, evaluated on the remaining target data. $n$ is the size of the local sample.}\scriptsize
\begin{tabular}{@{}llcccc@{}}\toprule
Local share & Strategy & AUC & Balanced accuracy & Brier & Calibration slope \\\midrule
""" + '\n'.join(r for r in rows5[:-1]).replace(r'\multicolumn{7}', r'\multicolumn{6}') + r"""
\bottomrule\end{tabular}\end{table}
""")

# S6 leakage full
rows6 = []
schemes = [('Random oversampling + split', 'correct_ros_split', 'leaky_ros_split', 'training only', 'before split'),
           ('SMOTE + split', 'correct_smote_split', 'leaky_smote_split', 'training only', 'before split'),
           ('SMOTEENN + 10-fold CV', 'correct_smoteenn', 'leaky_smoteenn', 'within folds', 'before CV'),
           ('No resampling (class weights) + 10-fold CV', 'correct_weighted', None, 'n/a', None)]
for c in ['Jamalpur', 'Dhaka']:
    rows6.append(r'\multicolumn{7}{@{}l}{\textit{' + c + f" ({L[c]['n_features']} features)" + r'}}\\')
    for nm, good, bad, gl, bl in schemes:
        for key, how in [(good, gl), (bad, bl)]:
            if key is None:
                continue
            cells = []
            for m in ['RF', 'ET']:
                v = L[c][m][key]
                cells.append(f"{v['acc'][0]*100:.1f} & {v['bal_acc'][0]:.3f} & {v['auc'][0]:.3f}")
            rows6.append(f"{nm if how in (gl,) else ''} & {how} & " + ' & '.join(cells) + r' \\')
    rows6.append(r'\midrule')
T.append(r"""
\begin{table}[htbp]\centering\caption{Leakage-free versus leaky resampling: accuracy (\%), balanced accuracy and AUC for random forest (RF) and Extra Trees (ET), mean over 10 splits or 10 folds. SMOTEENN applied before cross-validation: SMOTE ($k=5$) first oversampled the minority class to parity (Jamalpur 1,042 + 1,042 = 2,084 records; Dhaka 972 + 972 = 1,944); edited nearest neighbours ($k=3$, applied to both classes) then removed every record, real or synthetic, whose label disagreed with the majority of its three nearest neighbours. The remaining data, on which 10-fold CV was run, contained 648 records in Jamalpur (365 negative, 283 positive) and 1,569 in Dhaka (835 negative, 734 positive). In the correct version the same procedure was applied to each training fold only, and test folds kept their original records.}\scriptsize
\begin{tabular}{@{}llcccccc@{}}\toprule
 & & \multicolumn{3}{c}{RF} & \multicolumn{3}{c}{ET}\\\cmidrule(lr){3-5}\cmidrule(l){6-8}
Scheme & Resampling applied & Acc. & Bal.\ acc. & AUC & Acc. & Bal.\ acc. & AUC \\\midrule
""" + '\n'.join(rows6[:-1]).replace(r'\multicolumn{7}', r'\multicolumn{8}') + r"""
\bottomrule\end{tabular}\end{table}
""")

# S7 dedup
n = D['n_after_dedup']
T.append(r"""
\begin{table}[htbp]\centering\caption{Sensitivity analysis after removing exact duplicate records (Jamalpur $n=""" + f"{n['Jamalpur']:,}".replace(',', '{,}') + r"""$; Dhaka $n=""" + f"{n['Dhaka']:,}".replace(',', '{,}') + r"""$). Internal: mean $\pm$ SD over 15 CV folds. External: AUC (95\% bootstrap CI).}\small
\begin{tabular}{@{}lcc@{}}\toprule
Analysis & RF & GBM \\\midrule
Internal, Jamalpur & """ + f"{D['internal']['Jamalpur|RF'][0]:.3f} $\\pm$ {D['internal']['Jamalpur|RF'][1]:.3f} & {D['internal']['Jamalpur|GBM'][0]:.3f} $\\pm$ {D['internal']['Jamalpur|GBM'][1]:.3f}" + r""" \\
Internal, Dhaka & """ + f"{D['internal']['Dhaka|RF'][0]:.3f} $\\pm$ {D['internal']['Dhaka|RF'][1]:.3f} & {D['internal']['Dhaka|GBM'][0]:.3f} $\\pm$ {D['internal']['Dhaka|GBM'][1]:.3f}" + r""" \\
External, Jamalpur $\to$ Dhaka & """ + f"{ci(D['external']['Jamalpur->Dhaka|RF'])} & {ci(D['external']['Jamalpur->Dhaka|GBM'])}" + r""" \\
External, Dhaka $\to$ Jamalpur & """ + f"{ci(D['external']['Dhaka->Jamalpur|RF'])} & {ci(D['external']['Dhaka->Jamalpur|GBM'])}" + r""" \\
\bottomrule\end{tabular}\end{table}

\end{document}
""")
import os
tp = f'{res}/external_twosample_bootstrap.json'
if os.path.exists(tp):
    S = json.load(open(tp)); r8 = []
    for d in ['Jamalpur->Dhaka', 'Dhaka->Jamalpur']:
        for i, m in enumerate(['LR', 'RF', 'GBM']):
            v1 = B[f'{d}|{m}']; v2 = S[f'{d}|{m}']
            lead = (r'\multirow{3}{*}{' + d.replace('->', r' $\to$ ') + '}') if i == 0 else ''
            r8.append(f"{lead} & {m} & {v1['auc'][1]:.3f}--{v1['auc'][2]:.3f} & {v2['auc'][0]:.3f} ({v2['auc'][1]:.3f}--{v2['auc'][2]:.3f}) & "
                      f"{v1['cal_slope'][1]:.2f}--{v1['cal_slope'][2]:.2f} & {v2['cal_slope'][0]:.2f} ({v2['cal_slope'][1]:.2f}--{v2['cal_slope'][2]:.2f}) " + r'\\')
        r8.append(r'\midrule')
    s8 = (r"""
\begin{table}[htbp]\centering\caption{Conditional versus two-sample bootstrap 95\% intervals for external AUC and calibration slope. Conditional: target cohort resampled 1,000 times with the fitted source model held fixed (as in Table S3). Two-sample: in each of """ + f"{S['B']:,}" + r""" replicates the source cohort was resampled and the model refitted, and the target cohort was resampled independently; values are mean (2.5th--97.5th percentile). The stacked ensemble was omitted for computational reasons.}\small
\begin{tabular}{@{}llcccc@{}}\toprule
 & & \multicolumn{2}{c}{AUC} & \multicolumn{2}{c}{Calibration slope}\\\cmidrule(lr){3-4}\cmidrule(l){5-6}
Direction & Model & Conditional 95\% CI & Two-sample & Conditional 95\% CI & Two-sample \\\midrule
""" + '\n'.join(r8[:-1]) + r"""
\bottomrule\end{tabular}\end{table}
""")
    T[-1] = T[-1].replace(r'\end{document}', s8 + r'\end{document}')


# ---------------------------------------------------------------- S9 subgroups
sp = f'{res}/subgroup_results.json'
extra = ''
if os.path.exists(sp):
    G = json.load(open(sp)); r9 = []
    for d, rows in G.items():
        first = True
        for g, r in rows.items():
            lead = (r'\multirow{%d}{*}{%s}' % (len(rows), d.replace('->', r' $\to$ '))) if first else ''
            first = False
            cells = []
            for m in ['LR', 'RF', 'GBM', 'Stacked']:
                v = r.get(m)
                cells.append('--' if not v else f"{v['auc']:.2f} ({v['auc_ci'][0]:.2f}--{v['auc_ci'][1]:.2f})")
            cit = r.get('RF')
            r9.append(f"{lead} & {g.replace('15-60', '15--60').replace('<', '$<$').replace('>', '$>$')} & {r['n']:,} ({r['n_pos']:,}) & " + ' & '.join(cells)
                      + f" & {'--' if not cit else format(cit['citl'], '.2f')}" + r' \\')
        r9.append(r'\midrule')
    extra += r"""
\begin{table}[htbp]\centering\caption{External AUC (95\% conditional bootstrap CI, 1,000 resamples) by sex and age band in the target cohort, using the predictions of the full external models (all eight features). $n$ (pos.) is the number of patients (dengue-positive) in the subgroup. CITL: calibration-in-the-large of the RF model. Subgroups with fewer than 10 patients in either class are not analysed (--).}\scriptsize
\begin{tabular}{@{}llcccccc@{}}\toprule
Direction & Subgroup & $n$ (pos.) & LR & RF & GBM & Stacked & RF CITL \\\midrule
""" + '\n'.join(r9[:-1]) + r"""
\bottomrule\end{tabular}\end{table}
"""

# ---------------------------------------------------------------- S10 TRIPOD+AI
R = lambda k: r'\ref{M-' + k + '}'
tri = [
 ('Title', [('1', 'Title', 'Title page')]),
 ('Abstract', [('2', 'Abstract', 'Structured abstract')]),
 ('Introduction', [
  ('3a', 'Healthcare context and rationale', f'Section {R("sec:intro")}; Table 1'),
  ('3b', 'Target population, intended use and users', f'Section {R("sec:intro")} (final paragraph)'),
  ('3c', 'Known health inequalities', f'Not reported: no sociodemographic data beyond sex and age (Section {R("sec:limitations")})'),
  ('4', 'Objectives (development and/or evaluation)', f'Section {R("sec:intro")}; Section {R("sec:design")}')]),
 ('Methods', [
  ('5a', 'Data sources and representativeness', f'Section {R("sec:sources")}; Table 2; Section {R("sec:limitations")}'),
  ('5b', 'Dates of data collection', f'Section {R("sec:sources")}'),
  ('6a', 'Setting and centres', f'Section {R("sec:sources")}'),
  ('6b', 'Eligibility criteria', f'Section {R("sec:eligibility")}'),
  ('6c', 'Treatments', f'Section {R("sec:sources")} (not recorded; not relevant at presentation)'),
  ('7', 'Data preparation and quality checks', f'Sections {R("sec:audit-methods")}--{R("sec:preprocessing")}; Table 3'),
  ('8a', 'Outcome definition and assessment', f'Section {R("sec:outcome")}'),
  ('8b', 'Outcome assessors', f'Section {R("sec:outcome")} (not documented in the source data)'),
  ('8c', 'Blinding of outcome assessment', f'Section {R("sec:outcome")} (not documented in the source data)'),
  ('9a', 'Choice of initial predictors', f'Section {R("sec:harmonisation")}'),
  ('9b', 'Predictor definitions and timing', f'Section {R("sec:harmonisation")}; Table S2'),
  ('9c', 'Predictor assessors', 'Not applicable (automated laboratory measurements)'),
  ('10', 'Sample size', f'Section {R("sec:samplesize")}'),
  ('11', 'Missing data', f'Section {R("sec:preprocessing")}'),
  ('12a', 'Data use and partitioning', f'Section {R("sec:validation")}; Figure 1'),
  ('12b', 'Predictor handling', f'Sections {R("sec:harmonisation")}--{R("sec:preprocessing")}'),
  ('12c', 'Model type, building and internal validation', f'Sections {R("sec:models")}--{R("sec:validation")}; Table S1'),
  ('12d', 'Heterogeneity across clusters', f'Not possible: hospitals not identified in Dhaka (Sections {R("sec:eligibility")}, {R("sec:limitations")})'),
  ('12e', 'Performance measures', f'Section {R("sec:metrics")}'),
  ('12f', 'Model updating', f'Section {R("sec:updating")}'),
  ('12g', 'How predictions were calculated', f'Sections {R("sec:validation")}, {R("sec:software")} (released code)'),
  ('13', 'Class imbalance', f'Section {R("sec:models")} (class weights; no resampling)'),
  ('14', 'Fairness', f'Section {R("sec:validation")} (subgroups); Section {R("sec:limitations")}'),
  ('15', 'Model output and thresholds', f'Section {R("sec:models")}'),
  ('16', 'Differences between development and evaluation data', f'Sections {R("sec:outcome")}, {R("sec:cohorts")}, {R("sec:disc-comparability")}; Table S2'),
  ('17', 'Ethics', 'Declarations')]),
 ('Open science', [
  ('18a', 'Funding', 'Declarations'),
  ('18b', 'Conflicts of interest', 'Declarations'),
  ('18c', 'Protocol', f'Section {R("sec:design")}; Declarations (none prepared)'),
  ('18d', 'Registration', f'Section {R("sec:design")}; Declarations (not registered)'),
  ('18e', 'Data sharing', 'Declarations'),
  ('18f', 'Code sharing', f'Section {R("sec:software")}; Declarations')]),
 ('Patient and public involvement', [('19', 'Patient and public involvement', 'Declarations (none)')]),
 ('Results', [
  ('20a', 'Participant flow and outcome numbers', f'Section {R("sec:eligibility")}; Figure 1; Table 2'),
  ('20b', 'Participant characteristics', f'Section {R("sec:cohorts")}; Figure 2; Table S2'),
  ('20c', 'Comparison with development data', f'Section {R("sec:cohorts")}; Figure 2; Table S2'),
  ('21', 'Participants and events per analysis', 'Tables 2, 4--7; Tables S3--S9'),
  ('22', 'Full model specification', f'Section {R("sec:software")} (released code); Table S1'),
  ('23a', 'Performance with confidence intervals, incl.\\ subgroups', f'Sections {R("sec:res-internal")}--{R("sec:res-cal")}; Tables 4--6; Tables S3, S4, S8, S9'),
  ('23b', 'Heterogeneity across clusters', 'Not possible (see 12d)'),
  ('24', 'Model updating results', f'Section {R("sec:res-update")}; Figure 7; Table S5')]),
 ('Discussion', [
  ('25', 'Interpretation', f'Sections {R("sec:principal")}--{R("sec:disc-lit")}'),
  ('26', 'Limitations', f'Section {R("sec:limitations")}'),
  ('27a', 'Handling poor or missing input data', f'Sections {R("sec:preprocessing")}, {R("sec:disc-practice")}'),
  ('27b', 'User requirements', f'Not applicable: no model is proposed for deployment (Section {R("sec:disc-practice")})'),
  ('27c', 'Future research', f'Section {R("sec:conclusion")} (final paragraph)')]),
]
r10 = []
for sec, items in tri:
    r10.append(r'\multicolumn{3}{@{}l}{\textit{' + sec + r'}}\\')
    for it, topic, where in items:
        r10.append(f'{it} & {topic} & {where} ' + r'\\')
extra += r"""
\begin{table}[p]\centering\caption{TRIPOD+AI checklist (Collins et al., BMJ 2024;385:e078378). Section, table and figure numbers refer to the main manuscript unless prefixed with S.}\scriptsize
\begin{tabularx}{\linewidth}{@{}p{0.6cm}p{5.4cm}X@{}}\toprule
Item & Topic & Where reported \\\midrule
""" + '\n'.join(r10) + r"""
\bottomrule\end{tabularx}\end{table}
"""
T[-1] = T[-1].replace(r'\end{document}', extra + r'\end{document}')
open(out, 'w').write(''.join(T).replace('-0.00', '0.00'))
print('written', out)
