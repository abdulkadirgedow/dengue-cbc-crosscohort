"""Write the main-text results tables (LaTeX) directly from the result files.

Usage: python3 -I src/make_main_tables.py <results_dir> <out_dir>
Produces tab_internal.tex, tab_external.tex, tab_sens.tex and tab_third.tex.
"""
import sys, os, json

res, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
M = json.load(open(f'{res}/main_results.json'))
S2 = json.load(open(f'{res}/sensitivity2_results.json'))
T3 = json.load(open(f'{res}/third_cohort_results.json'))
MODELS = ['LR', 'RF', 'GBM', 'Stacked']
ARROW = lambda d: d.replace('->', r' $\to$ ')
f3 = lambda x: f'{x:.3f}'
f2 = lambda x: f'{x:.2f}'.replace('-0.00', '0.00')
ci = lambda r: f"{r['auc']:.3f} ({r['auc_ci'][0]:.3f}--{r['auc_ci'][1]:.3f})"


def write(name, body):
    open(f'{out}/{name}', 'w').write(body.replace('-0.000', '0.000'))


# ---------------- Table: internal validation
rows = []
for c in ['Jamalpur', 'Dhaka']:
    for i, m in enumerate(MODELS):
        r = M['internal'][c][m]
        lead = rf'\multirow{{4}}{{*}}{{{c}}}' if i == 0 else ''
        cells = ' & '.join(f'{r[k][0]:.3f} $\\pm$ {r[k][1]:.3f}' for k in ['auc', 'bal_acc', 'sens', 'spec'])
        rows.append(f'{lead} & {m} & {cells} \\\\')
    rows.append(r'\midrule')
write('tab_internal.tex', r"""\begin{tabular}{@{}llcccc@{}}
\toprule
Cohort & Model & AUC & Balanced acc. & Sensitivity & Specificity \\
\midrule
""" + '\n'.join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
""")

# ---------------- Table: external validation (all eight features)
rows = []
for d in ['Jamalpur->Dhaka', 'Dhaka->Jamalpur']:
    rows.append(rf'\multicolumn{{10}}{{@{{}}l}}{{\textit{{{ARROW(d)}}}}}\\')
    for m in MODELS:
        r = M['external'][f'all8|{d}'][m]
        rows.append(f"{m} & {ci(r)} & {f3(r['bal_acc'])} & {f3(r['sens'])} & {f3(r['spec'])} & {f3(r['ppv'])} & "
                    f"{f3(r['npv'])} & {f3(r['brier'])} & {f2(r['cal_slope'])} & {f2(r['cal_intercept'])} \\\\")
write('tab_external.tex', r"""\begin{tabular}{@{}lccccccccc@{}}
\toprule
Model & AUC (95\% CI) & Bal.\ acc. & Sens. & Spec. & PPV & NPV & Brier & Slope & CITL \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
""")

# ---------------- Table: sensitivity analyses (external AUC)
def ext_row(label, getter):
    return f"{label} & " + ' & '.join(f3(getter(m)) for m in MODELS) + r' \\'


rows = []
for d in ['Jamalpur->Dhaka', 'Dhaka->Jamalpur']:
    rows.append(rf'\multicolumn{{5}}{{@{{}}l}}{{\textit{{{ARROW(d)}}}}}\\')
    rows.append(ext_row('All 8 features (main analysis)', lambda m: M['external'][f'all8|{d}'][m]['auc']))
    rows.append(ext_row('Without RBC', lambda m: M['external'][f'noRBC|{d}'][m]['auc']))
    rows.append(ext_row('Ages 15--60', lambda m: M['age_restricted'][f'all8|{d}'][m]['auc']))
    rows.append(ext_row('Ages 15--60, without RBC', lambda m: M['age_restricted'][f'noRBC|{d}'][m]['auc']))
    rows.append(ext_row('Platelets + WBC only', lambda m: M['external'][f'PLT+WBC|{d}'][m]['auc']))
    rows.append(ext_row(r'Differential sum $\le$100\%', lambda m: S2['diff_le100']['external'][d][m]['auc']))
    rows.append(ext_row('Flagged Dhaka values kept', lambda m: S2['raw_extremes']['external'][d][m]['auc']))
    rows.append(ext_row('Without class weights', lambda m: S2['unweighted']['external'][d][m]['auc']))
write('tab_sens.tex', r"""\begin{tabular}{@{}lcccc@{}}
\toprule
Analysis & LR & RF & GBM & Stacked \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
""")

# ---------------- Table: third cohort (seven shared features)
E = T3['external']['shared7']; I = T3['internal']['shared7']
rows = [r'\multicolumn{5}{@{}l}{\textit{Internal validation (seven shared features; mean over 15 folds)}}\\']
NM = {'Jamalpur': 'Jamalpur 2024', 'Dhaka': 'Dhaka 2025', 'Riya': 'Dhaka 2023'}
for c in ['Jamalpur', 'Dhaka', 'Riya']:
    rows.append(f'{NM[c]} & ' + ' & '.join(f3(I[c][m][0]) for m in ['LR', 'RF', 'GBM']) + r' & -- \\')
rows.append(r'\midrule')
rows.append(r'\multicolumn{5}{@{}l}{\textit{External validation (AUC, 95\% CI)}}\\')
for d in ['Dhaka->Riya', 'Riya->Dhaka', 'Jamalpur->Riya', 'Riya->Jamalpur']:
    rows.append(' $\\to$ '.join(NM[x] for x in d.split('->')) + ' & ' + ' & '.join(
        f"{E[d][m]['auc']:.3f} ({E[d][m]['auc_ci'][0]:.2f}--{E[d][m]['auc_ci'][1]:.2f})" for m in MODELS) + r' \\')
write('tab_third.tex', r"""\begin{tabular}{@{}lcccc@{}}
\toprule
 & LR & RF & GBM & Stacked \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
""")
print('tables written to', out)
