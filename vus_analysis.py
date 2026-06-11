import numpy as np, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import stats as sc_stats
import warnings; warnings.filterwarnings('ignore')

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)          # one level up from src/
_OUT  = _os.path.join(_ROOT, 'outputs')
_os.makedirs(_OUT, exist_ok=True)
R   = json.load(open(_os.path.join(_OUT, 'results.json')))
sg  = np.array(R['sev_grid'])
fg  = np.array(R['fpr_grid'])

BLUE='#1F4E79'; RED='#C00000'; GREEN='#70AD47'
LTBLUE='#BDD7EE'; LTRED='#FCE4D6'
norm_v = Normalize(vmin=0.3, vmax=1.0)

def slogit(p): return np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
def slogistic(x): return 1/(1+np.exp(-np.clip(x,-15,15)))

def binormal_surface(auc_s_arr, fg, sg):
    auc_s_arr = np.array(auc_s_arr, dtype=float)
    """Fit binormal model to AUC(s) profile and return TPR surface Z."""
    valid = ~np.isnan(auc_s_arr)
    if valid.sum() < 3:
        return np.zeros((len(sg), len(fg)))
    s_v = sg[valid]; a_v = auc_s_arr[valid]
    # alpha from binormal: AUC = Phi(a/sqrt(2)) => a = sqrt(2)*Phi^{-1}(AUC)
    a_b = np.sqrt(2) * sc_stats.norm.ppf(np.clip(a_v, 0.501, 0.999))
    # Linear fit: a(s) = a0 + a1*s_norm
    s_n = (s_v - sg[0]) / (sg[-1] - sg[0])
    coef = np.polyfit(s_n, a_b, 1)
    Z = np.zeros((len(sg), len(fg)))
    for i, s in enumerate(sg):
        sn = (s - sg[0]) / (sg[-1] - sg[0])
        a_s = np.polyval(coef, sn)
        Z[i,:] = sc_stats.norm.cdf(a_s - sc_stats.norm.ppf(1-fg))
        for j in range(1, len(fg)):
            if Z[i,j] < Z[i,j-1]: Z[i,j] = Z[i,j-1]
    return Z

def dome_plot(ax, Z, sg, fg, col, vus, vus_lo, vus_hi, label):
    X, Y = np.meshgrid(fg, sg)
    fc = cm.viridis(norm_v(Z)); fc[...,3] = 0.78
    ax.plot_surface(X, Y, Z, facecolors=fc, linewidth=0,
                    antialiased=True, shade=True)
    ax.plot_surface(X, Y, X.copy(), color='lightgrey', alpha=0.12, linewidth=0)
    for q in [0.2, 0.5, 0.8]:
        idx = int(q * len(sg))
        vx = np.concatenate([[0], fg, [1], [0]])
        vz = np.concatenate([[0], Z[idx,:], [0], [0]])
        vy = np.full_like(vx, sg[idx])
        poly = Poly3DCollection([list(zip(vx,vy,vz))],
                                 alpha=0.18, facecolor=col, edgecolor='none')
        ax.add_collection3d(poly)
        ax.plot(fg, np.full_like(fg, sg[idx]), Z[idx,:], color=col, lw=1.8, zorder=5)
    # VUS waterline
    ax.plot_surface(X, Y, np.full_like(Z, vus),
                    color=col, alpha=0.12, linewidth=0)
    ax.text2D(0.05, 0.93,
              f'VUS = {vus:.3f}\n[{vus_lo:.3f}, {vus_hi:.3f}]' if vus is not None else 'VUS = n/e',
              transform=ax.transAxes, fontsize=9, fontweight='bold', color=col,
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        alpha=0.90, edgecolor=col, lw=1.5))
    ax.set_xlabel('1\u2212Specificity', fontsize=7, labelpad=1)
    ax.set_ylabel('SYNTAX Score', fontsize=7, labelpad=1)
    ax.set_zlabel('Sensitivity', fontsize=7, labelpad=1)
    ax.set_xlim(0,1); ax.set_ylim(sg[0],sg[-1]); ax.set_zlim(0,1)
    ax.set_title(label, fontsize=10, fontweight='bold', color=col, pad=5)
    ax.view_init(elev=26, azim=-52); ax.tick_params(labelsize=6)

def savefig(fig, name):
    fig.savefig(_os.path.join(_OUT, name),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  {name} saved.')

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Simulation 1 VUS domes (side by side)
# ════════════════════════════════════════════════════════════════════════════
s1 = R['sim1']
Z1 = binormal_surface(np.array(s1['auc_s1'], dtype=float), fg, sg)
Z2 = binormal_surface(np.array(s1['auc_s2'], dtype=float), fg, sg)

fig = plt.figure(figsize=(17, 7)); fig.patch.set_facecolor('white')
gs_f = GridSpec(1, 2, figure=fig, wspace=0.08, top=0.87,
                bottom=0.04, left=0.02, right=0.96)

ax1 = fig.add_subplot(gs_f[0,0], projection='3d')
dome_plot(ax1, Z1, sg, fg, BLUE, s1['vus1'], s1['vus1_lo'], s1['vus1_hi'],
          f'Test 1  (Naive AUC = {s1["nauc1"]:.3f})')

ax2 = fig.add_subplot(gs_f[0,1], projection='3d')
dome_plot(ax2, Z2, sg, fg, RED, s1['vus2'], s1['vus2_lo'], s1['vus2_hi'],
          f'Test 2  (Naive AUC = {s1["nauc2"]:.3f})')

sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm_v); sm.set_array([])
cbar_ax = fig.add_axes([0.963, 0.15, 0.012, 0.65])
fig.colorbar(sm, cax=cbar_ax).set_label('Sensitivity (TPR)', fontsize=9)

fig.suptitle(
    f'Simulation 1: Paired Comparison \u2014 VUS\u2081={s1["vus1"]:.3f} vs '
    f'VUS\u2082={s1["vus2"]:.3f}, \u0394VUS={s1["dvus"]:.3f} '
    f'[{s1["dvus_lo"]:.3f}, {s1["dvus_hi"]:.3f}], '
    f'P(VUS\u2081>VUS\u2082)={s1["p_vus1_gt"]:.3f}',
    fontsize=11, fontweight='bold', y=1.00)
savefig(fig, 'fig1_sim1_domes.png')

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Simulation 2 naive ROC curves
# ════════════════════════════════════════════════════════════════════════════
s2 = R['sim2']
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.patch.set_facecolor('white')

for ax, fpr_r, tpr_r, nauc, col, lbl, n, pop in [
    (axes[0], s2['fpr_nA'], s2['tpr_nA'], s2['naucA'],
     BLUE, 'Test A', 1500, 'Full spectrum (SYNTAX 1\u201360)'),
    (axes[1], s2['fpr_nB'], s2['tpr_nB'], s2['naucB'],
     RED,  'Test B', 500,  'Severe only (SYNTAX \u226533)'),
]:
    ax.plot(fpr_r, tpr_r, color=col, lw=2.5, label=f'AUC = {nauc:.3f}')
    ax.plot([0,1],[0,1], 'k--', lw=1.0, alpha=0.5)
    ax.fill_between(fpr_r, tpr_r, alpha=0.15, color=col)
    ax.set_xlabel('1 \u2212 Specificity (FPR)', fontsize=11)
    ax.set_ylabel('Sensitivity (TPR)', fontsize=11)
    ax.set_title(f'{lbl}  \u2014  n={n}\n{pop}',
                 fontsize=11, fontweight='bold', color=col)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.25); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.text(0.55, 0.10, f'Naive AUC = {nauc:.3f}',
            transform=ax.transAxes, fontsize=13, fontweight='bold', color=col)

fig.suptitle(
    'Simulation 2: Na\u00efve ROC Curves \u2014 Similar AUC Despite Different Populations\n'
    f'Test A AUC={s2["naucA"]:.3f} (full spectrum) vs '
    f'Test B AUC={s2["naucB"]:.3f} (severe only, spectrum-inflated)',
    fontsize=11, fontweight='bold')
plt.tight_layout()
savefig(fig, 'fig2_sim2_naive_roc.png')

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Simulation 2 VUS domes — redesigned
#   Left  (Test A): full dome, three slices at mild/intermediate/severe
#   Right (Test B): only severe region coloured; mild+intermediate shown as
#                   a translucent grey ghost; hard floor cut at SYNTAX 33;
#                   single labelled slice at PVUS_severe midpoint (~45)
# ════════════════════════════════════════════════════════════════════════════
ZA = binormal_surface(np.array(s2['auc_sA'], dtype=float), fg, sg)
ZB = binormal_surface(np.array(s2['auc_sB'], dtype=float), fg, sg)

# PVUS_severe midpoint index  (midpoint of SYNTAX 33-60 on the grid)
sev_start_idx = int(np.searchsorted(sg, 33.0))
pvus_mid_idx  = sev_start_idx + (len(sg) - sev_start_idx) // 2

fig = plt.figure(figsize=(17, 8)); fig.patch.set_facecolor('white')
gs_f = GridSpec(1, 2, figure=fig, wspace=0.06, top=0.88,
                bottom=0.04, left=0.02, right=0.96)

# ── LEFT: Test A — standard full dome ────────────────────────────────────────
ax1 = fig.add_subplot(gs_f[0, 0], projection='3d')
X, Y = np.meshgrid(fg, sg)
fc = cm.viridis(norm_v(ZA)); fc[..., 3] = 0.78
ax1.plot_surface(X, Y, ZA, facecolors=fc, linewidth=0, antialiased=True, shade=True)
ax1.plot_surface(X, Y, X.copy(), color='lightgrey', alpha=0.12, linewidth=0)

# Three slices at mild (~SYNTAX 11), intermediate (~28), severe (~47)
slice_targets = [11, 28, 47]
slice_labels  = ['Mild', 'Intermed.', 'Severe']
slice_cols    = ['#1F4E79', '#7D5A00', '#1D6B2E']
for s_tgt, slbl, scol in zip(slice_targets, slice_labels, slice_cols):
    idx = int(np.argmin(np.abs(sg - s_tgt)))
    vx = np.concatenate([[0], fg, [1], [0]])
    vz = np.concatenate([[0], ZA[idx,:], [0], [0]])
    vy = np.full_like(vx, sg[idx])
    poly = Poly3DCollection([list(zip(vx, vy, vz))],
                             alpha=0.22, facecolor=scol, edgecolor='none')
    ax1.add_collection3d(poly)
    ax1.plot(fg, np.full_like(fg, sg[idx]), ZA[idx,:], color=scol, lw=2.2, zorder=5)
    ax1.text(0.0, sg[idx], ZA[idx, 0]+0.04, slbl, fontsize=7, color=scol, fontweight='bold')

vusA = s2['vusA']
ax1.plot_surface(X, Y, np.full_like(ZA, vusA), color=BLUE, alpha=0.10, linewidth=0)
ax1.text2D(0.05, 0.93,
           f'VUS = {vusA:.3f}\n[{s2["vusA_lo"]:.3f}, {s2["vusA_hi"]:.3f}]',
           transform=ax1.transAxes, fontsize=9, fontweight='bold', color=BLUE,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.92,
                     edgecolor=BLUE, lw=1.5))
ax1.set_xlabel('1\u2212Specificity', fontsize=7, labelpad=1)
ax1.set_ylabel('SYNTAX Score', fontsize=7, labelpad=1)
ax1.set_zlabel('Sensitivity', fontsize=7, labelpad=1)
ax1.set_xlim(0,1); ax1.set_ylim(sg[0], sg[-1]); ax1.set_zlim(0,1)
ax1.set_title('Test A \u2014 Full Spectrum  (n=1,500)\nAll three severity regions characterised',
              fontsize=10, fontweight='bold', color=BLUE, pad=5)
ax1.view_init(elev=26, azim=-52); ax1.tick_params(labelsize=6)

# ── RIGHT: Test B — truncated dome with ghost and single PVUS slice ──────────
ax2 = fig.add_subplot(gs_f[0, 1], projection='3d')

# 1. Ghost surface for the MISSING region (SYNTAX < 33): pale grey wireframe
Z_ghost = ZB.copy()
for i in range(sev_start_idx):      # mild + intermediate rows — show as ghost
    ax2.plot_surface(
        np.meshgrid(fg, sg[max(0,i):i+2])[0],
        np.meshgrid(fg, sg[max(0,i):i+2])[1],
        Z_ghost[max(0,i):i+2, :],
        color='#CCCCCC', alpha=0.08, linewidth=0)

# 2. Hard vertical "cut wall" at SYNTAX = 33 — makes truncation explicit
cut_s = sg[sev_start_idx]
Z_cut = ZB[sev_start_idx, :]        # ROC curve at the cut boundary
wall_x = np.concatenate([[0], fg, [1], [1], [0], [0]])
wall_y = np.full(len(wall_x), cut_s)
wall_z = np.concatenate([[0], Z_cut, [0], [1], [1], [0]])
poly_wall = Poly3DCollection([list(zip(wall_x, wall_y, wall_z))],
                              alpha=0.30, facecolor='#C00000', edgecolor='none')
ax2.add_collection3d(poly_wall)
# Label the cut
ax2.text(0.5, cut_s - 1.5, 0.50, 'No data\nbelow\nSYNTAX 33',
         fontsize=7, color='#C00000', fontweight='bold', ha='center', va='top')

# 3. Coloured dome surface only for SEVERE region (SYNTAX >= 33)
X_sev = np.meshgrid(fg, sg[sev_start_idx:])[0]
Y_sev = np.meshgrid(fg, sg[sev_start_idx:])[1]
Z_sev = ZB[sev_start_idx:, :]
fc_sev = cm.viridis(norm_v(Z_sev)); fc_sev[..., 3] = 0.80
ax2.plot_surface(X_sev, Y_sev, Z_sev, facecolors=fc_sev,
                 linewidth=0, antialiased=True, shade=True)

# 4. Diagonal floor (full range, for reference)
ax2.plot_surface(X, Y, X.copy(), color='lightgrey', alpha=0.10, linewidth=0)

# 5. Single PVUS_severe slice at midpoint
pvus_sev_val = s2['pvusB'][2][0]    # PVUS_severe point estimate for Test B
idx_pvus = pvus_mid_idx
vx = np.concatenate([[0], fg, [1], [0]])
vz = np.concatenate([[0], ZB[idx_pvus,:], [0], [0]])
vy = np.full_like(vx, sg[idx_pvus])
poly_pvus = Poly3DCollection([list(zip(vx, vy, vz))],
                              alpha=0.35, facecolor='#C00000', edgecolor='none')
ax2.add_collection3d(poly_pvus)
ax2.plot(fg, np.full_like(fg, sg[idx_pvus]), ZB[idx_pvus,:],
         color='#C00000', lw=2.5, zorder=5)
ax2.text(0.0, sg[idx_pvus], ZB[idx_pvus, 0]+0.04,
         f'PVUS\u2091\u2091\u1d5b = {pvus_sev_val:.3f}',
         fontsize=7.5, color='#C00000', fontweight='bold')

# 6. Annotation box
ax2.text2D(0.05, 0.93,
           f'Global VUS = n/e\n(coverage 43% < 50%)\n'
           f'PVUS\u2091\u2091\u1d5b = {pvus_sev_val:.3f}\n'
           f'[{s2["pvusB"][2][1]:.3f}, {s2["pvusB"][2][2]:.3f}]',
           transform=ax2.transAxes, fontsize=8.5, fontweight='bold', color=RED,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.92,
                     edgecolor=RED, lw=1.5))
ax2.set_xlabel('1\u2212Specificity', fontsize=7, labelpad=1)
ax2.set_ylabel('SYNTAX Score', fontsize=7, labelpad=1)
ax2.set_zlabel('Sensitivity', fontsize=7, labelpad=1)
ax2.set_xlim(0,1); ax2.set_ylim(sg[0], sg[-1]); ax2.set_zlim(0,1)
ax2.set_title('Test B \u2014 Severe Only  (n=500)\nGrey ghost = missing data region; red wall = cut at SYNTAX 33',
              fontsize=10, fontweight='bold', color=RED, pad=5)
ax2.view_init(elev=26, azim=-52); ax2.tick_params(labelsize=6)

sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm_v); sm.set_array([])
cbar_ax = fig.add_axes([0.963, 0.15, 0.012, 0.65])
fig.colorbar(sm, cax=cbar_ax).set_label('Sensitivity (TPR)', fontsize=9)

pvus_sev_B = s2['pvusB'][2]
pvus_sev_A = s2['pvusA'][2]
dpvus_sev  = s2['dpvusAB'][2]
p_AgtB_sev = s2['p_pvusA_gt'][2]
fig.suptitle(
    f'Simulation 2: Spectrum Bias Revealed by Coverage-Gated VUS Analysis\n'
    f'Test A (full spectrum): Global VUS = {s2["vusA"]:.3f} [{s2["vusA_lo"]:.3f}, {s2["vusA_hi"]:.3f}]  '
    f'| Test B (severe only): Global VUS = n/e\n'
    f'Only PVUS\u2091\u2091\u1d5b is comparable: A = {pvus_sev_A[0]:.3f} vs B = {pvus_sev_B[0]:.3f}, '
    f'\u0394PVUS = {dpvus_sev[0]:.3f} [{dpvus_sev[1]:.3f}, {dpvus_sev[2]:.3f}], '
    f'P(A>B) = {p_AgtB_sev:.3f}',
    fontsize=10, fontweight='bold', y=1.01)
savefig(fig, 'fig3_sim2_domes.png')

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4: AUC(s) profiles for both simulations (supplementary / for doc)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(15, 6)); fig.patch.set_facecolor('white')

for ax, auc_s1_, auc_s2_, lo1, hi1, lo2, hi2, v1, v2, col1, col2, title, lbl1, lbl2 in [
    (axes[0],
     np.array(s1['auc_s1'], dtype=float), np.array(s1['auc_s2'], dtype=float),
     np.array(s1['auc_s1_lo'], dtype=float), np.array(s1['auc_s1_hi'], dtype=float),
     np.array(s1['auc_s2_lo'], dtype=float), np.array(s1['auc_s2_hi'], dtype=float),
     s1['vus1'], s1['vus2'], BLUE, RED,
     'Simulation 1: AUC(s) Profiles (Paired)',
     f'Test 1  VUS={s1["vus1"]:.3f}', f'Test 2  VUS={s1["vus2"]:.3f}'),
    (axes[1],
     np.array(s2['auc_sA'], dtype=float), np.array(s2['auc_sB'], dtype=float),
     np.array(s2['auc_sA_lo'], dtype=float), np.array(s2['auc_sA_hi'], dtype=float),
     np.array([np.nan]*50), np.array([np.nan]*50),
     s2['vusA'], s2['vusB'], BLUE, RED,
     'Simulation 2: AUC(s) Profiles (Unpaired)',
     f'Test A  VUS={s2["vusA"]:.3f}', f'Test B  VUS=n/e'),
]:
    valid1 = ~np.isnan(auc_s1_)
    valid2 = ~np.isnan(auc_s2_)
    ax.fill_between(sg[valid1], lo1[valid1], hi1[valid1], alpha=0.18, color=col1)
    ax.plot(sg[valid1], auc_s1_[valid1], color=col1, lw=2.5, label=lbl1)
    ax.axhline(v1, color=col1, lw=1.2, ls=':', alpha=0.7)
    if valid2.any():
        ax.fill_between(sg[valid2], lo2[valid2], hi2[valid2], alpha=0.15, color=col2)
        ax.plot(sg[valid2], auc_s2_[valid2], color=col2, lw=2.5, label=lbl2)
        if v2 is not None and not (v2!=v2): ax.axhline(v2, color=col2, lw=1.2, ls=':', alpha=0.7)
    else:
        # only plot what is there for test B in sim2
        ax.plot(sg[valid2], auc_s2_[valid2], color=col2, lw=2.5, label=lbl2)

    # PVUS region shading
    for (lo_s, hi_s), shade in [(( 0,22),'#EBF3FB'),((23,32),'#FFF2CC'),((33,60),'#E2EFDA')]:
        ax.axvspan(lo_s, hi_s, alpha=0.25, color=shade, zorder=0)
        ax.axvline(lo_s, color='#AAAAAA', lw=0.7, ls='--', alpha=0.6)

    ax.axhline(0.5, color='grey', lw=0.9, ls='--', alpha=0.5, label='Uninformative (0.5)')
    ax.set_xlabel('SYNTAX Score (disease severity)', fontsize=11)
    ax.set_ylabel('AUC at severity level', fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.2); ax.set_ylim(0.35, 1.0)
    ax.text(4, 0.37, 'Mild', fontsize=8, color='#1F4E79', alpha=0.8)
    ax.text(24, 0.37, 'Interm.', fontsize=8, color='#7D5A00', alpha=0.8)
    ax.text(37, 0.37, 'Severe', fontsize=8, color='#1D6B2E', alpha=0.8)

plt.tight_layout()
savefig(fig, 'fig4_auc_profiles.png')

print('\nAll figures saved.')
