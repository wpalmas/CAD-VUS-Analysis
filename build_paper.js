"""
run_simulations_v3.py
---------------------
Coverage-first VUS framework:
  - Global VUS requires ≥50% severity coverage; otherwise n/e
  - PVUS computed per region independently
  - Global ΔVUS requires both tests to have global VUS
  - Region ΔVUS computed only where both tests have data in that region
  - P(A > B) reported for every estimable comparison (global and per-region)
"""

import numpy as np
import pandas as pd
import json
import math
import warnings
from scipy import stats as sc_stats
from sklearn.metrics import roc_curve, auc as sk_auc

warnings.filterwarnings('ignore')

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_OUT  = _os.path.join(_ROOT, 'outputs')
_os.makedirs(_OUT, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
S_MIN, S_MAX  = 1.0, 60.0
PREV          = 0.30
N_BOOT        = 1000
N_GRID        = 50
N_STAR        = 15
WINDOW        = 5.0
CWR_THRESH    = 0.25
SGI_MIN_COV   = 0.50
VUS_MIN_COV   = 0.50   # global VUS requires this fraction of full range covered

fpr_grid = np.linspace(0.001, 0.999, 80)
sev_grid = np.linspace(S_MIN, S_MAX, N_GRID)

PVUS_BOUNDS = [(S_MIN, 23.0), (23.0, 33.0), (33.0, S_MAX + 1)]
PVUS_LABELS = ['Mild (0\u201322)', 'Intermediate (23\u201332)', 'Severe (\u226533)']


# ── Region helpers ────────────────────────────────────────────────────────────

def region_of(s):
    for i, (lo, hi) in enumerate(PVUS_BOUNDS):
        if lo <= s < hi:
            return i
    return len(PVUS_BOUNDS) - 1


def verify_tiling(sg=sev_grid):
    for s in sg:
        n = sum(1 for lo, hi in PVUS_BOUNDS if lo <= s < hi)
        assert n == 1, f"Grid point {s:.3f} in {n} regions"
    return True


# ── Data simulation ───────────────────────────────────────────────────────────

def sim_pop(n, s_mean, s_sd, s_lo, s_hi, slope_dis,
            int_dis=-1.2, int_nd=-1.5, sd=1.2, seed=42):
    rng     = np.random.default_rng(seed)
    syntax  = np.clip(rng.normal(s_mean, s_sd, n), s_lo, s_hi)
    disease = rng.binomial(1, PREV, n)
    mu      = np.where(disease == 1, int_dis + slope_dis * syntax, int_nd)
    score   = rng.normal(mu, sd)
    return pd.DataFrame({'syntax': syntax, 'disease': disease, 'score': score})


def make_full(slope, seed=42):
    rng = np.random.default_rng(seed)
    return pd.concat([
        sim_pop(500, 15, 6,  1,  35, slope, seed=int(rng.integers(int(1e6)))),
        sim_pop(500, 23, 7,  5,  50, slope, seed=int(rng.integers(int(1e6)))),
        sim_pop(500, 32, 7, 10,  60, slope, seed=int(rng.integers(int(1e6)))),
    ], ignore_index=True)


# ── Core: AUC(s) with region-clipped window ───────────────────────────────────

def compute_auc_s(dis_df, nd, sg=sev_grid, win=WINDOW, ns=N_STAR):
    a = np.full(len(sg), np.nan)
    for i, s in enumerate(sg):
        ri = region_of(s)
        lo_r, hi_r = PVUS_BOUNDS[ri]
        lo_w = max(s - win, lo_r)
        hi_w = min(s + win, hi_r)
        mask = (dis_df['syntax'] >= lo_w) & (dis_df['syntax'] < hi_w)
        d = dis_df[mask]['score'].values
        if len(d) < ns:
            continue
        labels = np.array([1] * len(d) + [0] * len(nd))
        fp, tp, _ = roc_curve(labels, np.concatenate([d, nd]))
        a[i] = float(sk_auc(fp, tp))
    return a


# ── Coverage check ────────────────────────────────────────────────────────────

def severity_coverage(auc_s, sg=sev_grid):
    """Return fractional coverage of the full [S_MIN, S_MAX] range."""
    vi = np.where(~np.isnan(auc_s))[0]
    if len(vi) < 2:
        return 0.0
    return float((sg[vi[-1]] - sg[vi[0]]) / (S_MAX - S_MIN))


# ── VUS (coverage-gated) ──────────────────────────────────────────────────────

def compute_vus(auc_s, sg=sev_grid, min_cov=VUS_MIN_COV):
    """
    Global VUS. Returns (value_or_nan, message).
    nan when coverage < min_cov.
    """
    cov = severity_coverage(auc_s, sg)
    if cov < min_cov:
        vi = np.where(~np.isnan(auc_s))[0]
        s_lo = sg[vi[0]] if len(vi) > 0 else S_MIN
        s_hi = sg[vi[-1]] if len(vi) > 0 else S_MIN
        msg = (f'Global VUS cannot be estimated: observed range '
               f'[{s_lo:.0f}, {s_hi:.0f}] covers only {cov*100:.0f}% '
               f'of the full range [{S_MIN:.0f}, {S_MAX:.0f}] '
               f'(minimum required: {min_cov*100:.0f}%)')
        return np.nan, msg
    v = auc_s[~np.isnan(auc_s)]
    return float(np.mean(v)), 'estimable'


# ── PVUS (per region, independently gated) ────────────────────────────────────

def compute_pvus(auc_s, sg=sev_grid):
    """
    PVUS per region. Each region estimated independently.
    Returns dict: label -> (value_or_nan, n_grid_points_with_data).
    """
    result = {}
    for (lo, hi), lbl in zip(PVUS_BOUNDS, PVUS_LABELS):
        mask = (sg >= lo) & (sg < hi) & (~np.isnan(auc_s))
        k = mask.sum()
        result[lbl] = (float(np.mean(auc_s[mask])) if k > 0 else np.nan, int(k))
    return result


def pvus_values(pvus_dict):
    """Return just the scalar values from compute_pvus output."""
    return {lbl: v for lbl, (v, k) in pvus_dict.items()}


# ── SGI (parametric, coverage-gated) ─────────────────────────────────────────

def fit_binormal_surface(auc_s, sg=sev_grid):
    valid = ~np.isnan(auc_s)
    if valid.sum() < 3:
        return auc_s.copy()
    s_v  = sg[valid]
    a_v  = np.clip(auc_s[valid], 0.501, 0.999)
    sn_v = (s_v - S_MIN) / (S_MAX - S_MIN)
    a_b  = np.sqrt(2) * sc_stats.norm.ppf(a_v)
    coef = np.polyfit(sn_v, a_b, 1)
    sn_all  = (sg - S_MIN) / (S_MAX - S_MIN)
    return sc_stats.norm.cdf(np.polyval(coef, sn_all) / np.sqrt(2))


def compute_sgi(auc_s, sg=sev_grid, min_cov=SGI_MIN_COV):
    vi = np.where(~np.isnan(auc_s))[0]
    if len(vi) < 3:
        return np.nan, 'SGI: fewer than 3 characterised bins'
    s_lo = sg[vi[0]]; s_hi = sg[vi[-1]]
    cov  = (s_hi - s_lo) / (S_MAX - S_MIN)
    if cov < min_cov:
        return np.nan, (f'SGI cannot be estimated: observed range '
                        f'[{s_lo:.0f}, {s_hi:.0f}] covers only {cov*100:.0f}% '
                        f'of the full range (minimum: {min_cov*100:.0f}%)')
    fit  = fit_binormal_surface(auc_s, sg)
    a_lo = fit[vi[0]]; a_hi = fit[vi[-1]]
    if a_hi <= a_lo:
        return np.nan, 'SGI: fitted surface not monotone at extremes'
    return float((a_hi - a_lo) / a_hi * 100.0), 'estimable'


# ── Quality measures ──────────────────────────────────────────────────────────

def compute_mvf(auc_s):
    return float(np.isnan(auc_s).sum() / len(auc_s))


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n; d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1-p) / n + z**2 / (4 * n**2)) / d
    return max(0.0, c - h), min(1.0, c + h)


def icv_from_boot(auc_s_pt, boot_auc_s):
    has  = ~np.isnan(auc_s_pt)
    boot = np.array(boot_auc_s, dtype=float)
    cwr_arr = np.full(len(auc_s_pt), np.nan)
    for k in range(len(auc_s_pt)):
        if not has[k]: continue
        v = boot[:, k]; v = v[~np.isnan(v)]
        if len(v) < 10: continue
        lo, hi = np.percentile(v, [2.5, 97.5])
        pt = auc_s_pt[k]
        if pt > 0: cwr_arr[k] = (hi - lo) / pt
    exc = has & (cwr_arr > CWR_THRESH)
    return float(exc.sum() / max(has.sum(), 1)), cwr_arr


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def boot_stats(dis_df, nd, n_boot=N_BOOT, seed=0,
               dis_df2=None, nd2=None, paired=False):
    """
    Returns B (and B2 if paired) with keys:
      vus, pvus (list of 3), sgi, auc_s
    vus entries are nan when coverage insufficient.
    """
    rng2 = np.random.default_rng(seed)
    n_d  = len(dis_df); n_nd = len(nd)
    B    = {'vus': [], 'pvus': [[], [], []], 'sgi': [], 'auc_s': []}
    B2   = {'vus': [], 'pvus': [[], [], []], 'sgi': [], 'auc_s': []} if paired else None

    for draw in range(n_boot):
        if draw % 200 == 0:
            print(f'    draw {draw}/{n_boot}...', flush=True)
        idx_d  = rng2.integers(0, n_d,  n_d)
        idx_nd = rng2.integers(0, n_nd, n_nd)
        bd = dis_df.iloc[idx_d].reset_index(drop=True)
        bn = nd[idx_nd]
        a  = compute_auc_s(bd, bn)

        vus_val, _ = compute_vus(a)
        B['vus'].append(vus_val)
        pv = compute_pvus(a)
        for j, lbl in enumerate(PVUS_LABELS):
            B['pvus'][j].append(pv[lbl][0])
        B['sgi'].append(compute_sgi(a)[0])
        B['auc_s'].append(a)

        if paired:
            bd2 = dis_df2.iloc[idx_d].reset_index(drop=True)
            bn2 = nd2[idx_nd] if nd2 is not None else bn
            a2  = compute_auc_s(bd2, bn2)
            vus2, _ = compute_vus(a2)
            B2['vus'].append(vus2)
            pv2 = compute_pvus(a2)
            for j, lbl in enumerate(PVUS_LABELS):
                B2['pvus'][j].append(pv2[lbl][0])
            B2['sgi'].append(compute_sgi(a2)[0])
            B2['auc_s'].append(a2)

    return (B, B2) if paired else B


def pci(arr):
    """Posterior CI from bootstrap array. Returns (mean, lo, hi) or (nan,nan,nan)."""
    a = np.array([x for x in arr
                  if x is not None and not np.isnan(float(x))], dtype=float)
    if len(a) < 10:
        return np.nan, np.nan, np.nan
    return float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def p_greater(arr1, arr2):
    """
    Bayesian P(arr1 > arr2) from paired bootstrap draws.
    Both arrays must be same length (one draw each).
    Handles nan draws by excluding them pairwise.
    """
    a1 = np.array(arr1, dtype=float)
    a2 = np.array(arr2, dtype=float)
    ok = ~np.isnan(a1) & ~np.isnan(a2)
    if ok.sum() < 10:
        return np.nan
    return float(np.mean(a1[ok] > a2[ok]))


def delta_pci(arr1, arr2):
    """
    ΔVUS posterior: arr1 - arr2, pairwise, excluding nan draws.
    Returns (mean, lo, hi, p_gt) or (nan,nan,nan,nan).
    """
    a1 = np.array(arr1, dtype=float)
    a2 = np.array(arr2, dtype=float)
    ok = ~np.isnan(a1) & ~np.isnan(a2)
    if ok.sum() < 10:
        return np.nan, np.nan, np.nan, np.nan
    d = a1[ok] - a2[ok]
    return (float(np.mean(d)),
            float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)),
            float(np.mean(d > 0)))


def fmt(m, lo, hi, pct=False, dec=3):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return 'n/e\u2020'
    if pct:
        return f'{m:.1f}% [{lo:.1f}%, {hi:.1f}%]'
    return f'{m:.{dec}f} [{lo:.{dec}f}, {hi:.{dec}f}]'


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return 'n/e\u2020'
    return f'{p:.3f}'


def clean(obj):
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, list):
        return [clean(x) for x in obj]
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    return obj


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

verify_tiling()
print('Region tiling: OK', flush=True)

# ── Datasets ──────────────────────────────────────────────────────────────────
df1 = make_full(0.055, seed=42)
df2 = make_full(0.042, seed=42)
rng_b = np.random.default_rng(99)
dfB   = sim_pop(500, 45, 8, 33, 60, 0.042,
                seed=int(rng_b.integers(int(1e6))))

fp_n1, tp_n1, _ = roc_curve(df1['disease'], df1['score'])
fp_n2, tp_n2, _ = roc_curve(df2['disease'], df2['score'])
fp_nB, tp_nB, _ = roc_curve(dfB['disease'], dfB['score'])
nauc1 = float(sk_auc(fp_n1, tp_n1))
nauc2 = float(sk_auc(fp_n2, tp_n2))
naucB = float(sk_auc(fp_nB, tp_nB))

dis1 = df1[df1['disease'] == 1].reset_index(drop=True)
nd1  = df1[df1['disease'] == 0]['score'].values
dis2 = df2[df2['disease'] == 1].reset_index(drop=True)
nd2  = df2[df2['disease'] == 0]['score'].values
disA = dis1.copy(); ndA = nd1.copy()
disB = dfB[dfB['disease'] == 1].reset_index(drop=True)
ndB  = dfB[dfB['disease'] == 0]['score'].values
naucA = nauc1

fp_nA, tp_nA = fp_n1, tp_n1

# ── Point estimates ────────────────────────────────────────────────────────────
auc_s1 = compute_auc_s(dis1, nd1)
auc_s2 = compute_auc_s(dis2, nd2)
auc_sA = auc_s1.copy()
auc_sB = compute_auc_s(disB, ndB)

vus1_pt, vus1_msg = compute_vus(auc_s1)
vus2_pt, vus2_msg = compute_vus(auc_s2)
vusA_pt, vusA_msg = compute_vus(auc_sA)
vusB_pt, vusB_msg = compute_vus(auc_sB)

pvus1_pt = compute_pvus(auc_s1)
pvus2_pt = compute_pvus(auc_s2)
pvusA_pt = compute_pvus(auc_sA)
pvusB_pt = compute_pvus(auc_sB)

sgi1_pt, sgi1_msg = compute_sgi(auc_s1)
sgi2_pt, sgi2_msg = compute_sgi(auc_s2)
sgiA_pt, sgiA_msg = compute_sgi(auc_sA)
sgiB_pt, sgiB_msg = compute_sgi(auc_sB)

mvf1 = compute_mvf(auc_s1); mvf2 = compute_mvf(auc_s2)
mvfA = compute_mvf(auc_sA); mvfB = compute_mvf(auc_sB)

print(f'\nPoint estimates:')
print(f'  Test1 VUS={vus1_pt:.4f} ({vus1_msg[:9]})')
print(f'  Test2 VUS={vus2_pt:.4f} ({vus2_msg[:9]})')
print(f'  TestA VUS={vusA_pt:.4f} ({vusA_msg[:9]})')
print(f'  TestB VUS: {vusB_msg[:60]}')
for lbl in PVUS_LABELS:
    v, k = pvusB_pt[lbl]
    vs = f'{v:.4f}' if not np.isnan(v) else 'n/e'
    print(f'    TestB PVUS {lbl}: {vs} ({k} grid pts)')


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION 1: PAIRED
# ═══════════════════════════════════════════════════════════════════════════════
print('\nBootstrapping Simulation 1 (paired)...', flush=True)
B1, B2 = boot_stats(dis1, nd1, dis_df2=dis2, nd2=nd2, paired=True, seed=1)

# Global VUS
v1m, v1lo, v1hi = pci(B1['vus'])
v2m, v2lo, v2hi = pci(B2['vus'])
dvm, dvlo, dvhi, p_dvus = delta_pci(B1['vus'], B2['vus'])
p_1gt2 = p_greater(B1['vus'], B2['vus'])

# PVUS + region ΔVUS
pvus1_ci  = [pci(B1['pvus'][j]) for j in range(3)]
pvus2_ci  = [pci(B2['pvus'][j]) for j in range(3)]
dpvus_ci  = [delta_pci(B1['pvus'][j], B2['pvus'][j]) for j in range(3)]
p_pvus    = [p_greater(B1['pvus'][j], B2['pvus'][j]) for j in range(3)]

# SGI
sgi1m, sgi1lo, sgi1hi = pci(B1['sgi'])
sgi2m, sgi2lo, sgi2hi = pci(B2['sgi'])

# MVF / ICV
icv1, cwr1 = icv_from_boot(auc_s1, B1['auc_s'])
icv2, cwr2 = icv_from_boot(auc_s2, B2['auc_s'])
n_exc1 = int((~np.isnan(auc_s1) & (cwr1 > CWR_THRESH)).sum())
n_chr1 = int((~np.isnan(auc_s1)).sum())
n_exc2 = int((~np.isnan(auc_s2) & (cwr2 > CWR_THRESH)).sum())
n_chr2 = int((~np.isnan(auc_s2)).sum())
mvf1_lo, mvf1_hi = wilson_ci(int(np.isnan(auc_s1).sum()), N_GRID)
mvf2_lo, mvf2_hi = wilson_ci(int(np.isnan(auc_s2).sum()), N_GRID)
icv1_lo, icv1_hi = wilson_ci(n_exc1, n_chr1)
icv2_lo, icv2_hi = wilson_ci(n_exc2, n_chr2)

print('\n=== Simulation 1: Paired ===')
print(f'  VUS:    T1={fmt(v1m,v1lo,v1hi)}  T2={fmt(v2m,v2lo,v2hi)}')
print(f'  dVUS:   {fmt(dvm,dvlo,dvhi)}  P(T1>T2)={fmt_p(p_1gt2)}')
print(f'  SGI:    T1={fmt(sgi1m,sgi1lo,sgi1hi,pct=True)}  T2={fmt(sgi2m,sgi2lo,sgi2hi,pct=True)}')
for j, lbl in enumerate(PVUS_LABELS):
    m1,l1,h1 = pvus1_ci[j]; m2,l2,h2 = pvus2_ci[j]
    dm,dl,dh,_ = dpvus_ci[j]; pp = p_pvus[j]
    print(f'  {lbl}:')
    print(f'    T1={fmt(m1,l1,h1)}  T2={fmt(m2,l2,h2)}')
    print(f'    dPVUS={fmt(dm,dl,dh)}  P(T1>T2)={fmt_p(pp)}')


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION 2: UNPAIRED  (coverage-gated)
# ═══════════════════════════════════════════════════════════════════════════════
print('\nBootstrapping Simulation 2 — Test A...', flush=True)
BA = boot_stats(disA, ndA, seed=2)
print('Bootstrapping Simulation 2 — Test B...', flush=True)
BB = boot_stats(disB, ndB, seed=3)

# Global VUS — Test B is n/e
vAm, vAlo, vAhi = pci(BA['vus'])
vBm, vBlo, vBhi = pci(BB['vus'])   # will be nan (coverage < 50%)

# Global ΔVUS — n/e because TestB global VUS is n/e
dvABm, dvABlo, dvABhi, _ = delta_pci(BA['vus'], BB['vus'])  # nan
p_AgB_global = p_greater(BA['vus'], BB['vus'])               # nan

# PVUS per region
pvusA_ci = [pci(BA['pvus'][j]) for j in range(3)]
pvusB_ci = [pci(BB['pvus'][j]) for j in range(3)]

# Region ΔVUS — only where both have data
dpvusAB_ci = [delta_pci(BA['pvus'][j], BB['pvus'][j]) for j in range(3)]
p_pvusAB   = [p_greater(BA['pvus'][j], BB['pvus'][j]) for j in range(3)]

# SGI
sgiAm, sgiAlo, sgiAhi = pci(BA['sgi'])

# MVF / ICV
icvA, cwrA = icv_from_boot(auc_sA, BA['auc_s'])
icvB, cwrB = icv_from_boot(auc_sB, BB['auc_s'])
n_excA = int((~np.isnan(auc_sA) & (cwrA > CWR_THRESH)).sum())
n_chrA = int((~np.isnan(auc_sA)).sum())
n_excB = int((~np.isnan(auc_sB) & (cwrB > CWR_THRESH)).sum())
n_chrB = int((~np.isnan(auc_sB)).sum())
mvfA_lo, mvfA_hi = wilson_ci(int(np.isnan(auc_sA).sum()), N_GRID)
mvfB_lo, mvfB_hi = wilson_ci(int(np.isnan(auc_sB).sum()), N_GRID)
icvA_lo, icvA_hi = wilson_ci(n_excA, n_chrA)
icvB_lo, icvB_hi = wilson_ci(n_excB, n_chrB)

print('\n=== Simulation 2: Unpaired (coverage-gated) ===')
print(f'  Naive AUC: TA={naucA:.3f}  TB={naucB:.3f}')
print(f'  Global VUS TA: {fmt(vAm,vAlo,vAhi)}')
print(f'  Global VUS TB: {vusB_msg[:80]}')
print(f'  Global dVUS:   n/e (TestB global VUS not estimable)')
for j, lbl in enumerate(PVUS_LABELS):
    mA,lA,hA = pvusA_ci[j]; mB,lB,hB = pvusB_ci[j]
    dm,dl,dh,_ = dpvusAB_ci[j]; pp = p_pvusAB[j]
    print(f'  {lbl}:')
    print(f'    TA={fmt(mA,lA,hA)}  TB={fmt(mB,lB,hB)}')
    print(f'    dPVUS={fmt(dm,dl,dh)}  P(TA>TB)={fmt_p(pp)}')
print(f'  SGI TA: {fmt(sgiAm,sgiAlo,sgiAhi,pct=True)}')
print(f'  SGI TB: {sgiB_msg[:80]}')


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════════

R = clean({
    'sim1': {
        'nauc1': nauc1, 'nauc2': nauc2,
        'vus1': v1m, 'vus1_lo': v1lo, 'vus1_hi': v1hi,
        'vus2': v2m, 'vus2_lo': v2lo, 'vus2_hi': v2hi,
        'dvus': dvm, 'dvus_lo': dvlo, 'dvus_hi': dvhi,
        'p_vus1_gt': p_1gt2,
        'mvf1': mvf1, 'mvf1_lo': mvf1_lo, 'mvf1_hi': mvf1_hi,
        'mvf2': mvf2, 'mvf2_lo': mvf2_lo, 'mvf2_hi': mvf2_hi,
        'icv1': icv1, 'icv1_lo': icv1_lo, 'icv1_hi': icv1_hi,
        'icv2': icv2, 'icv2_lo': icv2_lo, 'icv2_hi': icv2_hi,
        'sgi1': sgi1m, 'sgi1_lo': sgi1lo, 'sgi1_hi': sgi1hi,
        'sgi2': sgi2m, 'sgi2_lo': sgi2lo, 'sgi2_hi': sgi2hi,
        'sgi1_msg': sgi1_msg, 'sgi2_msg': sgi2_msg,
        'pvus1': [list(x) for x in pvus1_ci],
        'pvus2': [list(x) for x in pvus2_ci],
        'dpvus': [list(x[:3]) for x in dpvus_ci],
        'p_pvus1_gt': p_pvus,
        'auc_s1': auc_s1.tolist(), 'auc_s2': auc_s2.tolist(),
        'auc_s1_lo': np.nanpercentile(np.array(B1['auc_s'], dtype=float), 2.5,  axis=0).tolist(),
        'auc_s1_hi': np.nanpercentile(np.array(B1['auc_s'], dtype=float), 97.5, axis=0).tolist(),
        'auc_s2_lo': np.nanpercentile(np.array(B2['auc_s'], dtype=float), 2.5,  axis=0).tolist(),
        'auc_s2_hi': np.nanpercentile(np.array(B2['auc_s'], dtype=float), 97.5, axis=0).tolist(),
        'fpr_n1': fp_n1.tolist(), 'tpr_n1': tp_n1.tolist(),
        'fpr_n2': fp_n2.tolist(), 'tpr_n2': tp_n2.tolist(),
    },
    'sim2': {
        'naucA': naucA, 'naucB': naucB,
        'vusA': vAm, 'vusA_lo': vAlo, 'vusA_hi': vAhi,
        'vusB': vBm, 'vusB_lo': vBlo, 'vusB_hi': vBhi,
        'vusB_msg': vusB_msg,
        'dvus': dvABm, 'dvus_lo': dvABlo, 'dvus_hi': dvABhi,
        'p_A_gt_B': p_AgB_global,
        'mvfA': mvfA, 'mvfA_lo': mvfA_lo, 'mvfA_hi': mvfA_hi,
        'mvfB': mvfB, 'mvfB_lo': mvfB_lo, 'mvfB_hi': mvfB_hi,
        'icvA': icvA, 'icvA_lo': icvA_lo, 'icvA_hi': icvA_hi,
        'icvB': icvB, 'icvB_lo': icvB_lo, 'icvB_hi': icvB_hi,
        'sgiA': sgiAm, 'sgiA_lo': sgiAlo, 'sgiA_hi': sgiAhi,
        'sgiA_msg': sgiA_msg, 'sgiB_msg': sgiB_msg,
        'pvusA': [list(x) for x in pvusA_ci],
        'pvusB': [list(x) for x in pvusB_ci],
        'dpvusAB': [list(x[:3]) for x in dpvusAB_ci],
        'p_pvusA_gt': p_pvusAB,
        'auc_sA': auc_sA.tolist(), 'auc_sB': auc_sB.tolist(),
        'auc_sA_lo': np.nanpercentile(np.array(BA['auc_s'], dtype=float), 2.5,  axis=0).tolist(),
        'auc_sA_hi': np.nanpercentile(np.array(BA['auc_s'], dtype=float), 97.5, axis=0).tolist(),
        'fpr_nA': fp_nA.tolist(), 'tpr_nA': tp_nA.tolist(),
        'fpr_nB': fp_nB.tolist(), 'tpr_nB': tp_nB.tolist(),
    },
    'sev_grid': sev_grid.tolist(),
    'fpr_grid': fpr_grid.tolist(),
})

with open(_os.path.join(_OUT, 'results.json'), 'w') as f:
    json.dump(R, f, indent=2)

print('\nAll results saved.')
print('Done.')
