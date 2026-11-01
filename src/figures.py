"""All manuscript figures. Usage: python3 -I src/figures.py <jamalpur.csv> <dhaka.csv> <results_dir> <fig_dir>"""
import sys, os, json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_jamalpur, load_dhaka, make_model, net_benefit, FEATSETS

warnings.filterwarnings('ignore')
jam, dha, res_dir, fig = sys.argv[1:5]
os.makedirs(fig, exist_ok=True)

# validated categorical palette (fixed order) + text tokens
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
COH_C = {'Jamalpur': C[0], 'Dhaka': C[1]}
MOD_C = {'LR': C[0], 'RF': C[1], 'GBM': C[2], 'Stacked': C[3]}
plt.rcParams.update({'font.size': 10.5, 'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'legend.frameon': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'font.family': 'DejaVu Sans'})

def panel_label(a, text, y=-0.30):
    """Bold '(a) ...' label centred underneath a panel, below its x-axis label."""
    a.text(0.5, y, text, transform=a.transAxes, ha='center', va='top', fontsize=11, fontweight='bold', color=INK)


XJ, yJ, _ = load_jamalpur(jam)
XD, yD, _, _ = load_dhaka(dha)
M = json.load(open(f'{res_dir}/main_results.json'))
L = json.load(open(f'{res_dir}/leakage_results.json'))
U = pd.read_csv(f'{res_dir}/local_update_raw.csv')
P = f'{res_dir}/preds'

# ---------------------------------------------------------------- Fig 2 distributions
fig2, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
for i, (feat, lab, lim) in enumerate([('PLT', 'Platelet count (×10³/µL)', (0, 500)), ('WBC', 'WBC count (×10³/µL)', (0, 16))]):
    data, pos, cols = [], [], []
    for j, (cn, X, y) in enumerate([('Jamalpur', XJ, yJ), ('Dhaka', XD, yD)]):
        for k, (cls, nm) in enumerate([(0, 'Negative'), (1, 'Positive')]):
            data.append(X.loc[y == cls, feat].dropna().values); pos.append(j * 2.6 + k); cols.append(C[k])
    bp = ax[i].boxplot(data, positions=pos, widths=0.75, patch_artist=True, showfliers=False,
                       medianprops=dict(color=INK, linewidth=1.2), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    for b, c in zip(bp['boxes'], cols):
        b.set_facecolor(c); b.set_alpha(0.85); b.set_edgecolor('white'); b.set_linewidth(1.5)
    ax[i].set_xticks([0.5, 3.1]); ax[i].set_xticklabels(['Jamalpur 2024', 'Dhaka 2025'])
    ax[i].set_ylabel(lab); ax[i].set_ylim(*lim); ax[i].grid(axis='x', visible=False)
    a_j = M['cohorts']['Jamalpur']['univariable_auc'][feat]; a_d = M['cohorts']['Dhaka']['univariable_auc'][feat]
    ax[i].text(0.5, lim[1] * 0.95, f'AUC {a_j:.2f}', ha='center', color=INK2, fontsize=9.5)
    ax[i].text(3.1, lim[1] * 0.95, f'AUC {a_d:.2f}', ha='center', color=INK2, fontsize=9.5)
    panel_label(ax[i], ['(a) Platelet count', '(b) White blood cell count'][i], y=-0.16)
h = [plt.Rectangle((0, 0), 1, 1, color=C[0]), plt.Rectangle((0, 0), 1, 1, color=C[1])]
fig2.legend(h, ['Dengue negative', 'Dengue positive'], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.06))
fig2.tight_layout(); fig2.savefig(f'{fig}/fig2_distributions.pdf'); fig2.savefig(f'{fig}/fig2_distributions.png'); plt.close(fig2)

# ---------------------------------------------------------------- Fig 3 internal vs external AUC
fig3, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
models = ['LR', 'RF', 'GBM', 'Stacked']
for i, (tgt, src) in enumerate([('Dhaka', 'Jamalpur'), ('Jamalpur', 'Dhaka')]):
    a = ax[i]
    for k, mname in enumerate(models):
        intr = M['internal'][tgt][mname]['auc']; ext = M['external'][f'all8|{src}->{tgt}'][mname]
        a.errorbar(intr[0], k + 0.15, xerr=intr[1], fmt='o', ms=6,  # internal: mean ± SD over 15 CV folds
                   color=C[0], ecolor=C[0], elinewidth=1.5, capsize=0)
        a.errorbar(ext['auc'], k - 0.15, xerr=[[ext['auc'] - ext['auc_ci'][0]], [ext['auc_ci'][1] - ext['auc']]],
                   fmt='s', ms=6, color=C[1], ecolor=C[1], elinewidth=1.5, capsize=0)
    a.axvline(0.5, color=INK2, lw=0.8, ls=':')
    a.set_yticks(range(len(models))); a.set_yticklabels(models); a.set_xlim(0.45, 1.0)
    panel_label(a, ['(a) Tested on Dhaka', '(b) Tested on Jamalpur'][i], y=-0.24)
    a.set_xlabel('AUC'); a.grid(axis='y', visible=False)
ax[0].plot([], [], 'o', color=C[0], label='Internal (trained on same cohort)')
ax[0].plot([], [], 's', color=C[1], label='External (trained on other cohort)')
fig3.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.07))
fig3.tight_layout(); fig3.savefig(f'{fig}/fig3_internal_vs_external.pdf'); fig3.savefig(f'{fig}/fig3_internal_vs_external.png'); plt.close(fig3)

# ---------------------------------------------------------------- Fig 4 calibration (external)
def reliability(y, p, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); idx = np.clip(np.searchsorted(q, p, side='right') - 1, 0, bins - 1)
    return np.array([p[idx == b].mean() for b in range(bins) if np.any(idx == b)]), \
        np.array([y[idx == b].mean() for b in range(bins) if np.any(idx == b)])


fig4, ax = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
for i, (src, tgt, yt) in enumerate([('Jamalpur', 'Dhaka', yD), ('Dhaka', 'Jamalpur', yJ)]):
    a = ax[i]; a.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=':')
    for mname in ['LR', 'RF', 'GBM']:
        p = np.load(f'{P}/{src}_to_{tgt}_all8_{mname}.npy'); mp, ob = reliability(yt, p)
        e = M['external'][f'all8|{src}->{tgt}'][mname]
        a.plot(mp, ob, '-o', color=MOD_C[mname], lw=2, ms=4, label=f"{mname} (slope {e['cal_slope']:.2f})")
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_aspect('equal')
    a.set_xlabel('Predicted probability'); panel_label(a, ['(a) Jamalpur → Dhaka', '(b) Dhaka → Jamalpur'][i], y=-0.22)
    a.legend(loc='lower right', fontsize=9)
ax[0].set_ylabel('Observed proportion positive')
fig4.tight_layout(); fig4.savefig(f'{fig}/fig4_calibration.pdf'); fig4.savefig(f'{fig}/fig4_calibration.png'); plt.close(fig4)

# ---------------------------------------------------------------- Fig 5 decision curves (external)
th = np.linspace(0.05, 0.90, 86)
fig5, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
for i, (src, tgt, yt) in enumerate([('Jamalpur', 'Dhaka', yD), ('Dhaka', 'Jamalpur', yJ)]):
    a = ax[i]; prev = yt.mean()
    a.plot(th, prev - (1 - prev) * th / (1 - th), color=INK2, lw=1, ls='--', label='Test all')
    a.axhline(0, color=INK2, lw=1, ls=':', label='Test none')
    for mname in ['LR', 'RF', 'GBM']:
        p = np.load(f'{P}/{src}_to_{tgt}_all8_{mname}.npy')
        a.plot(th, net_benefit(yt, p, th), color=MOD_C[mname], lw=2, label=mname)
    a.set_ylim(-0.1, max(prev, 0.8)); a.set_xlabel('Threshold probability')
    panel_label(a, ['(a) Jamalpur → Dhaka', '(b) Dhaka → Jamalpur'][i], y=-0.28)
ax[0].set_ylabel('Net benefit'); ax[1].legend(fontsize=9, loc='upper right')
fig5.tight_layout(); fig5.savefig(f'{fig}/fig5_decision_curves.pdf'); fig5.savefig(f'{fig}/fig5_decision_curves.png'); plt.close(fig5)

# ---------------------------------------------------------------- Fig 6 SHAP comparison
import shap
cols = FEATSETS['all8']
shap_imp = {}
for cn, X, y in [('Jamalpur', XJ, yJ), ('Dhaka', XD, yD)]:
    m = make_model('RF').fit(X[cols], y)
    Xi = pd.DataFrame(m[0].transform(X[cols]), columns=cols)
    rng = np.random.default_rng(0); idx = rng.choice(len(Xi), size=min(600, len(Xi)), replace=False)
    sv = shap.TreeExplainer(m[-1]).shap_values(Xi.iloc[idx])
    sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if np.ndim(sv) == 3 else sv)
    shap_imp[cn] = np.abs(sv).mean(axis=0)
json.dump({k: dict(zip(cols, map(float, v))) for k, v in shap_imp.items()}, open(f'{res_dir}/shap_importance.json', 'w'), indent=1)
labels = {'Age': 'Age', 'Male': 'Male sex', 'PLT': 'Platelets', 'WBC': 'WBC', 'HCT': 'Haematocrit', 'RBC': 'RBC',
          'Lymph': 'Lymphocytes %', 'Neut': 'Neutrophils %'}
order = np.argsort(shap_imp['Dhaka'] + shap_imp['Jamalpur'])
fig6, ax = plt.subplots(figsize=(5.2, 3.4))
yy = np.arange(len(cols))
ax.barh(yy + 0.2, shap_imp['Jamalpur'][order], height=0.38, color=COH_C['Jamalpur'], label='Jamalpur 2024', edgecolor='white', linewidth=1)
ax.barh(yy - 0.2, shap_imp['Dhaka'][order], height=0.38, color=COH_C['Dhaka'], label='Dhaka 2025', edgecolor='white', linewidth=1)
ax.set_yticks(yy); ax.set_yticklabels([labels[cols[i]] for i in order]); ax.grid(axis='y', visible=False)
ax.set_xlabel('Mean |SHAP value| (random forest, positive class)'); ax.legend(loc='lower right')
fig6.tight_layout(); fig6.savefig(f'{fig}/fig6_shap.pdf'); fig6.savefig(f'{fig}/fig6_shap.png'); plt.close(fig6)

# ---------------------------------------------------------------- Fig 7 local updating
STRAT = [('source_only', 'Source model only'), ('recalibrated', 'Source model, recalibrated'),
         ('target_only', 'Local data only'), ('pooled', 'Source + local data')]
fig7, ax = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
for i, (d, tgt) in enumerate([('Jamalpur->Dhaka', 'Dhaka'), ('Dhaka->Jamalpur', 'Jamalpur')]):
    a = ax[i]; sub = U[U.direction == d]
    for k, (s, nm) in enumerate(STRAT):
        g = sub[sub.strategy == s].groupby('frac').auc.agg(['mean', 'std'])
        x = g.index.values * 100
        a.plot(x, g['mean'], '-o', color=C[k], lw=2, ms=5, label=nm)
        a.fill_between(x, g['mean'] - g['std'], g['mean'] + g['std'], color=C[k], alpha=0.12, lw=0)
    ref = M['internal'][tgt]['RF']['auc'][0]
    a.axhline(ref, color=INK2, lw=0.9, ls='--'); a.text(x[-1], ref + 0.006, 'internal CV', ha='right', fontsize=9, color=INK2)
    a.set_xticks(x); a.set_xlabel(f'Share of {tgt} data used locally (%)')
    panel_label(a, ['(a) Jamalpur → Dhaka', '(b) Dhaka → Jamalpur'][i], y=-0.27)
ax[0].set_ylabel('AUC on remaining target data')
h, l = ax[0].get_legend_handles_labels(); fig7.legend(h, l, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.07), fontsize=9)
fig7.tight_layout(); fig7.savefig(f'{fig}/fig7_local_updating.pdf'); fig7.savefig(f'{fig}/fig7_local_updating.png'); plt.close(fig7)

# ---------------------------------------------------------------- Fig 8 leakage inflation (Jamalpur)
pairs = [('Oversampling\n(random), split', 'correct_ros_split', 'leaky_ros_split'),
         ('SMOTE, split', 'correct_smote_split', 'leaky_smote_split'),
         ('SMOTEENN, 10-fold CV', 'correct_smoteenn', 'leaky_smoteenn')]
fig8, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
for i, cn in enumerate(['Jamalpur', 'Dhaka']):
    a = ax[i]
    for k, (nm, good, bad) in enumerate(pairs):
        g = L[cn]['RF'][good]['acc'][0] * 100; b = L[cn]['RF'][bad]['acc'][0] * 100
        a.plot([g, b], [k, k], color=GRID, lw=3, zorder=1)
        a.scatter([g], [k], s=60, color=C[0], zorder=3, edgecolor='white', linewidth=1.5)
        a.scatter([b], [k], s=60, color=C[1], zorder=3, edgecolor='white', linewidth=1.5)
        a.text(b + 0.8, k, f'+{b - g:.1f}', va='center', fontsize=9.5, color=INK2)
    a.set_yticks(range(len(pairs))); a.set_yticklabels([p[0] for p in pairs]); a.grid(axis='y', visible=False)
    a.set_xlabel('Accuracy (%) — random forest'); panel_label(a, ['(a) Jamalpur', '(b) Dhaka'][i], y=-0.27)
    a.set_xlim(60, 102)
ax[0].scatter([], [], color=C[0], label='Resampling on training data only (correct)')
ax[0].scatter([], [], color=C[1], label='Resampling before split / CV (leaky)')
fig8.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.08), fontsize=9.5)
fig8.tight_layout(); fig8.savefig(f'{fig}/fig8_leakage.pdf'); fig8.savefig(f'{fig}/fig8_leakage.png'); plt.close(fig8)
print('figures done')
