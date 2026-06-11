"""
vus_analysis.py
===============
Complete VUS analysis pipeline for a single study.

Implements:
  - Pooled non-diseased FPR (non-diseased SYNTAX = 0 by definition)
  - Region-clipped windows (half-open boundaries, no gaps, no cross-region leakage)
  - Coverage-first global VUS (requires ≥50% severity coverage)
  - PVUS per region, independently estimated
  - SGI: parametric binormal, denominator = AUC_severe, bounded [0,100%]
  - MVF: Missing Volume Fraction with Wilson 95% CI
  - ICV: Imprecision of Covered Volume with Wilson 95% CI
  - Naive AUC with Hanley-McNeil 95% CI
  - Bootstrap credible intervals (1000 draws) for all VUS statistics
  - P(A>B) Bayesian posterior probability for every comparison

Usage
-----
    from vus_analysis import VUSAnalysis
    result = VUSAnalysis(
        data         = df,            # DataFrame with disease, syntax, score columns
        score_col    = 'test_score',
        disease_col  = 'disease_status',
        severity_col = 'syntax_score',
        severity_min = 0.0,
        severity_max = 60.0,
    ).run()
    result.summary()

License: MIT
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as sc_stats
from sklearn.metrics import roc_curve, auc as sk_auc
import warnings
import math

warnings.filterwarnings('ignore')

# ── Default configuration ─────────────────────────────────────────────────────
DEFAULT_N_GRID      = 50
DEFAULT_N_FPR       = 80
DEFAULT_WINDOW      = 5.0
DEFAULT_N_STAR      = 15
DEFAULT_N_BOOT      = 1000
DEFAULT_CWR_THRESH  = 0.25
DEFAULT_VUS_MIN_COV = 0.50
DEFAULT_SGI_MIN_COV = 0.50
DEFAULT_PVUS_LABELS = ['Mild (0\u201322)', 'Intermediate (23\u201332)', 'Severe (\u226533)']


# ── PVUS region definitions (half-open: every grid point in exactly one) ─────

def make_pvus_bounds(severity_min: float, severity_max: float,
                     breaks=(23.0, 33.0)):
    """
    Build half-open PVUS region boundaries from a severity range and breaks.
    Default breaks at 23 and 33 match the SYNTAX trial tertiles.
    Returns list of (lo, hi) pairs covering [severity_min, severity_max].
    """
    lo_vals = [severity_min] + list(breaks)
    hi_vals = list(breaks) + [severity_max + 1e-9]
    return list(zip(lo_vals, hi_vals))


def region_of(s: float, bounds) -> int:
    for i, (lo, hi) in enumerate(bounds):
        if lo <= s < hi:
            return i
    return len(bounds) - 1


def verify_tiling(sg: np.ndarray, bounds) -> bool:
    for s in sg:
        n = sum(1 for lo, hi in bounds if lo <= s < hi)
        if n != 1:
            raise ValueError(f'Grid point {s:.3f} belongs to {n} regions')
    return True


# ── Core estimation ───────────────────────────────────────────────────────────

def compute_auc_s(dis_df: pd.DataFrame,
                  nd_scores: np.ndarray,
                  sg: np.ndarray,
                  bounds,
                  win: float  = DEFAULT_WINDOW,
                  ns:  int    = DEFAULT_N_STAR) -> np.ndarray:
    """
    AUC at each severity grid point using pooled non-diseased FPR.
    Patient window is clipped to the same PVUS region as the grid point,
    preventing cross-boundary leakage.
    """
    a = np.full(len(sg), np.nan)
    for i, s in enumerate(sg):
        ri = region_of(s, bounds)
        lo_r, hi_r = bounds[ri]
        lo_w = max(s - win, lo_r)
        hi_w = min(s + win, hi_r)
        mask = (dis_df['_sev'] >= lo_w) & (dis_df['_sev'] < hi_w)
        d = dis_df.loc[mask, '_score'].values
        if len(d) < ns:
            continue
        labels = np.concatenate([[1] * len(d), [0] * len(nd_scores)])
        scores = np.concatenate([d, nd_scores])
        fp, tp, _ = roc_curve(labels, scores)
        a[i] = float(sk_auc(fp, tp))
    return a


def fit_binormal_surface(auc_s: np.ndarray,
                         sg: np.ndarray,
                         sev_min: float,
                         sev_max: float) -> np.ndarray:
    """
    Fit linear binormal model: a_bin(s) = c0 + c1*s_norm
    AUC(s) = Phi(a_bin / sqrt(2)).  Returns fitted AUC array.
    """
    valid = ~np.isnan(auc_s)
    if valid.sum() < 3:
        return auc_s.copy()
    s_v  = sg[valid]
    a_v  = np.clip(auc_s[valid], 0.501, 0.999)
    sn_v = (s_v - sev_min) / (sev_max - sev_min)
    a_b  = np.sqrt(2) * sc_stats.norm.ppf(a_v)
    coef = np.polyfit(sn_v, a_b, 1)
    sn_all = (sg - sev_min) / (sev_max - sev_min)
    return sc_stats.norm.cdf(np.polyval(coef, sn_all) / np.sqrt(2))


def severity_coverage(auc_s: np.ndarray, sg: np.ndarray,
                      sev_min: float, sev_max: float) -> float:
    vi = np.where(~np.isnan(auc_s))[0]
    if len(vi) < 2:
        return 0.0
    return float((sg[vi[-1]] - sg[vi[0]]) / (sev_max - sev_min))


def compute_vus(auc_s: np.ndarray, sg: np.ndarray,
                sev_min: float, sev_max: float,
                min_cov: float = DEFAULT_VUS_MIN_COV):
    """
    Global VUS. Returns (value_or_nan, message).
    nan when coverage < min_cov.
    """
    cov = severity_coverage(auc_s, sg, sev_min, sev_max)
    if cov < min_cov:
        vi = np.where(~np.isnan(auc_s))[0]
        s_lo = sg[vi[0]] if len(vi) > 0 else sev_min
        s_hi = sg[vi[-1]] if len(vi) > 0 else sev_min
        msg = (f'Global VUS cannot be estimated: observed range '
               f'[{s_lo:.0f}, {s_hi:.0f}] covers only {cov*100:.0f}% '
               f'of the full range [{sev_min:.0f}, {sev_max:.0f}] '
               f'(minimum required: {min_cov*100:.0f}%)')
        return np.nan, msg
    v = auc_s[~np.isnan(auc_s)]
    return float(np.mean(v)), 'estimable'


def compute_pvus(auc_s: np.ndarray, sg: np.ndarray, bounds,
                 labels=None):
    """
    PVUS per region. Returns dict label -> (value_or_nan, n_grid_points).
    Identity VUS == weighted_mean(PVUS) holds exactly.
    """
    if labels is None:
        labels = DEFAULT_PVUS_LABELS
    result = {}
    for (lo, hi), lbl in zip(bounds, labels):
        mask = (sg >= lo) & (sg < hi) & (~np.isnan(auc_s))
        k = mask.sum()
        result[lbl] = (float(np.mean(auc_s[mask])) if k > 0 else np.nan, int(k))
    return result


def compute_sgi(auc_s: np.ndarray, sg: np.ndarray,
                sev_min: float, sev_max: float,
                min_cov: float = DEFAULT_SGI_MIN_COV):
    """
    SGI = [AUC_fit(s_max_obs) - AUC_fit(s_min_obs)] / AUC_fit(s_max_obs) * 100%.
    Uses fitted binormal surface; denominator = AUC_severe (stable, bounded [0,100%]).
    Returns (value_or_nan, message).
    """
    vi = np.where(~np.isnan(auc_s))[0]
    if len(vi) < 3:
        return np.nan, 'SGI: fewer than 3 characterised bins'
    s_lo = sg[vi[0]]; s_hi = sg[vi[-1]]
    cov  = (s_hi - s_lo) / (sev_max - sev_min)
    if cov < min_cov:
        return np.nan, (f'SGI cannot be estimated: observed range '
                        f'[{s_lo:.0f}, {s_hi:.0f}] covers only {cov*100:.0f}% '
                        f'of the full possible range [{sev_min:.0f}, {sev_max:.0f}] '
                        f'(minimum required: {min_cov*100:.0f}%)')
    fit  = fit_binormal_surface(auc_s, sg, sev_min, sev_max)
    a_lo = fit[vi[0]]; a_hi = fit[vi[-1]]
    if a_hi <= a_lo:
        return np.nan, 'SGI: fitted surface not monotone at observed extremes'
    return float((a_hi - a_lo) / a_hi * 100.0), 'estimable'


def compute_mvf(auc_s: np.ndarray):
    return float(np.isnan(auc_s).sum() / len(auc_s))


def compute_icv(auc_s_pt: np.ndarray, boot_auc_s: list,
                cwr_thresh: float = DEFAULT_CWR_THRESH):
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
    exc = has & (cwr_arr > cwr_thresh)
    return float(exc.sum() / max(has.sum(), 1)), cwr_arr


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0: return 0.0, 1.0
    p = k / n; d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    h = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / d
    return max(0.0, c - h), min(1.0, c + h)


def hanley_mcneil_ci(auc: float, n_pos: int, n_neg: int, z: float = 1.96):
    """
    Hanley & McNeil (1982) analytic 95% CI for the AUC.
    Standard method for reporting naive AUC confidence intervals.
    Returns (auc, lo, hi, se).
    """
    Q1  = auc / (2 - auc)
    Q2  = 2 * auc**2 / (1 + auc)
    var = (auc*(1-auc)
           + (n_pos-1)*(Q1 - auc**2)
           + (n_neg-1)*(Q2 - auc**2)) / (n_pos * n_neg)
    se  = float(np.sqrt(max(var, 0)))
    lo  = max(0.0, auc - z*se)
    hi  = min(1.0, auc + z*se)
    return float(auc), lo, hi, se


# ── Bootstrap CI helpers ──────────────────────────────────────────────────────

def pci(arr):
    a = np.array([x for x in arr
                  if x is not None and not np.isnan(float(x))], dtype=float)
    if len(a) < 10:
        return np.nan, np.nan, np.nan
    return float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def p_greater(arr1, arr2):
    """P(arr1 > arr2) from paired bootstrap draws, nan excluded pairwise."""
    a1 = np.array(arr1, dtype=float)
    a2 = np.array(arr2, dtype=float)
    ok = ~np.isnan(a1) & ~np.isnan(a2)
    return float(np.mean(a1[ok] > a2[ok])) if ok.sum() >= 10 else np.nan


def delta_pci(arr1, arr2):
    """(mean, lo, hi, P>0) for arr1-arr2, nan excluded pairwise."""
    a1 = np.array(arr1, dtype=float)
    a2 = np.array(arr2, dtype=float)
    ok = ~np.isnan(a1) & ~np.isnan(a2)
    if ok.sum() < 10:
        return np.nan, np.nan, np.nan, np.nan
    d = a1[ok] - a2[ok]
    return (float(np.mean(d)), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)), float(np.mean(d > 0)))


# ── Main analysis class ───────────────────────────────────────────────────────

class VUSAnalysis:
    """
    Full VUS analysis for one or two diagnostic tests.

    Parameters
    ----------
    data          : pd.DataFrame with score, disease, severity columns
    score_col     : continuous test score column name
    disease_col   : binary disease label column (1=diseased, 0=non-diseased)
    severity_col  : disease severity column name
    severity_min  : minimum POSSIBLE severity value (full biological range)
    severity_max  : maximum POSSIBLE severity value
    score_col_2   : second test score column (optional, for paired comparison)
    pvus_breaks   : tuple of PVUS region break points (default (23.0, 33.0))
    pvus_labels   : list of PVUS region labels
    n_grid        : number of severity grid points (default 50)
    n_fpr         : FPR grid resolution (default 80)
    window        : severity window half-width (default 5.0)
    n_star        : minimum diseased per bin (default 15)
    n_boot        : bootstrap draws (default 1000)
    cwr_thresh    : ICV CI width ratio threshold (default 0.25)
    vus_min_cov   : minimum coverage for global VUS (default 0.50)
    sgi_min_cov   : minimum coverage for SGI (default 0.50)
    random_seed   : random seed (default 42)
    verbose       : print progress (default True)
    """

    def __init__(self,
                 data:          pd.DataFrame,
                 score_col:     str   = 'test_score',
                 disease_col:   str   = 'disease_status',
                 severity_col:  str   = 'syntax_score',
                 severity_min:  float = 0.0,
                 severity_max:  float = 60.0,
                 score_col_2:   str   = None,
                 pvus_breaks:   tuple = (23.0, 33.0),
                 pvus_labels:   list  = None,
                 n_grid:        int   = DEFAULT_N_GRID,
                 n_fpr:         int   = DEFAULT_N_FPR,
                 window:        float = DEFAULT_WINDOW,
                 n_star:        int   = DEFAULT_N_STAR,
                 n_boot:        int   = DEFAULT_N_BOOT,
                 cwr_thresh:    float = DEFAULT_CWR_THRESH,
                 vus_min_cov:   float = DEFAULT_VUS_MIN_COV,
                 sgi_min_cov:   float = DEFAULT_SGI_MIN_COV,
                 random_seed:   int   = 42,
                 verbose:       bool  = True):

        self.data         = data.copy()
        self.score_col    = score_col
        self.disease_col  = disease_col
        self.severity_col = severity_col
        self.sev_min      = severity_min
        self.sev_max      = severity_max
        self.score_col_2  = score_col_2
        self.paired       = (score_col_2 is not None)
        self.n_grid       = n_grid
        self.n_fpr        = n_fpr
        self.window       = window
        self.n_star       = n_star
        self.n_boot       = n_boot
        self.cwr_thresh   = cwr_thresh
        self.vus_min_cov  = vus_min_cov
        self.sgi_min_cov  = sgi_min_cov
        self.seed         = random_seed
        self.verbose      = verbose

        self.fpr_grid = np.linspace(0.001, 0.999, n_fpr)
        self.sg       = np.linspace(severity_min, severity_max, n_grid)
        self.bounds   = make_pvus_bounds(severity_min, severity_max, pvus_breaks)
        self.plabels  = pvus_labels or DEFAULT_PVUS_LABELS
        verify_tiling(self.sg, self.bounds)

        # Prepare internal columns
        self.data['_score']   = self.data[score_col]
        self.data['_disease'] = self.data[disease_col].astype(int)
        self.data['_sev']     = self.data[severity_col]
        if self.paired:
            self.data['_score2'] = self.data[score_col_2]

        self._dis = self.data[self.data['_disease'] == 1].reset_index(drop=True)
        self._nd  = self.data[self.data['_disease'] == 0]['_score'].values
        if self.paired:
            self._nd2 = self.data[self.data['_disease'] == 0]['_score2'].values

    # ── Naive AUC (Hanley-McNeil) ────────────────────────────────────────────

    def _naive_auc(self, score_col='_score'):
        fp, tp, _ = roc_curve(self.data['_disease'], self.data[score_col])
        auc = float(sk_auc(fp, tp))
        n_pos = int(self.data['_disease'].sum())
        n_neg = int((self.data['_disease'] == 0).sum())
        auc_hm, lo, hi, se = hanley_mcneil_ci(auc, n_pos, n_neg)
        return {'auc': auc_hm, 'lo': lo, 'hi': hi, 'se': se,
                'n_pos': n_pos, 'n_neg': n_neg,
                'fpr': fp.tolist(), 'tpr': tp.tolist()}

    # ── Point estimates ──────────────────────────────────────────────────────

    def _point_estimates(self, dis_df, nd):
        auc_s = compute_auc_s(dis_df, nd, self.sg, self.bounds,
                               self.window, self.n_star)
        vus_v, vus_msg = compute_vus(auc_s, self.sg, self.sev_min,
                                      self.sev_max, self.vus_min_cov)
        pvus_d = compute_pvus(auc_s, self.sg, self.bounds, self.plabels)
        sgi_v, sgi_msg = compute_sgi(auc_s, self.sg, self.sev_min,
                                      self.sev_max, self.sgi_min_cov)
        mvf = compute_mvf(auc_s)
        return dict(auc_s=auc_s, vus=vus_v, vus_msg=vus_msg,
                    pvus=pvus_d, sgi=sgi_v, sgi_msg=sgi_msg, mvf=mvf)

    # ── Bootstrap ────────────────────────────────────────────────────────────

    def _bootstrap(self, dis_df, nd, dis_df2=None, nd2=None):
        rng   = np.random.default_rng(self.seed)
        n_d   = len(dis_df); n_nd = len(nd)
        B     = {'vus': [], 'pvus': [[] for _ in self.bounds], 'sgi': [], 'auc_s': []}
        B2    = {'vus': [], 'pvus': [[] for _ in self.bounds], 'sgi': [], 'auc_s': []} \
                if self.paired else None

        for draw in range(self.n_boot):
            if self.verbose and draw % 200 == 0:
                print(f'  draw {draw}/{self.n_boot}...', flush=True)
            idx_d  = rng.integers(0, n_d,  n_d)
            idx_nd = rng.integers(0, n_nd, n_nd)
            bd  = dis_df.iloc[idx_d].reset_index(drop=True)
            bn  = nd[idx_nd]
            a   = compute_auc_s(bd, bn, self.sg, self.bounds, self.window, self.n_star)
            v, _ = compute_vus(a, self.sg, self.sev_min, self.sev_max, self.vus_min_cov)
            B['vus'].append(v)
            pv = compute_pvus(a, self.sg, self.bounds, self.plabels)
            for j, lbl in enumerate(self.plabels):
                B['pvus'][j].append(pv[lbl][0])
            B['sgi'].append(compute_sgi(a, self.sg, self.sev_min, self.sev_max, self.sgi_min_cov)[0])
            B['auc_s'].append(a)

            if self.paired and dis_df2 is not None:
                bd2 = dis_df2.iloc[idx_d].reset_index(drop=True)
                bn2 = nd2[idx_nd] if nd2 is not None else bn
                a2  = compute_auc_s(bd2, bn2, self.sg, self.bounds, self.window, self.n_star)
                v2, _ = compute_vus(a2, self.sg, self.sev_min, self.sev_max, self.vus_min_cov)
                B2['vus'].append(v2)
                pv2 = compute_pvus(a2, self.sg, self.bounds, self.plabels)
                for j, lbl in enumerate(self.plabels):
                    B2['pvus'][j].append(pv2[lbl][0])
                B2['sgi'].append(compute_sgi(a2, self.sg, self.sev_min, self.sev_max, self.sgi_min_cov)[0])
                B2['auc_s'].append(a2)

        return (B, B2) if self.paired else B

    # ── Run ──────────────────────────────────────────────────────────────────

    def run(self) -> 'VUSResult':
        if self.verbose:
            print(f'VUS Analysis: n={len(self.data)}, '
                  f'severity [{self.sev_min}, {self.sev_max}]', flush=True)

        # Naive AUC
        naive1 = self._naive_auc('_score')
        naive2 = self._naive_auc('_score2') if self.paired else None

        # Point estimates
        if self.verbose: print('Computing point estimates...', flush=True)
        pt1 = self._point_estimates(self._dis, self._nd)
        if self.paired:
            dis2 = self.data[self.data['_disease'] == 1].reset_index(drop=True)
            dis2['_score'] = dis2['_score2']
            nd2 = self._nd2
            pt2 = self._point_estimates(dis2, nd2)
        else:
            pt2 = None

        # Bootstrap
        if self.verbose: print('Bootstrapping...', flush=True)
        if self.paired:
            dis2_boot = self.data[self.data['_disease'] == 1].reset_index(drop=True)
            dis2_boot['_score'] = dis2_boot['_score2']
            B, B2 = self._bootstrap(self._dis, self._nd, dis2_boot, self._nd2)
        else:
            B = self._bootstrap(self._dis, self._nd)
            B2 = None

        # ICV
        icv1, cwr1 = compute_icv(pt1['auc_s'], B['auc_s'], self.cwr_thresh)
        n_exc1 = int((~np.isnan(pt1['auc_s']) & (cwr1 > self.cwr_thresh)).sum())
        n_chr1 = int((~np.isnan(pt1['auc_s'])).sum())
        icv2 = cwr2 = n_exc2 = n_chr2 = None
        if B2:
            icv2, cwr2 = compute_icv(pt2['auc_s'], B2['auc_s'], self.cwr_thresh)
            n_exc2 = int((~np.isnan(pt2['auc_s']) & (cwr2 > self.cwr_thresh)).sum())
            n_chr2 = int((~np.isnan(pt2['auc_s'])).sum())

        # CIs
        v1m,v1lo,v1hi  = pci(B['vus'])
        dv = dp = None
        if B2:
            v2m,v2lo,v2hi = pci(B2['vus'])
            dv  = delta_pci(B['vus'], B2['vus'])
            p12 = p_greater(B['vus'], B2['vus'])
        else:
            v2m = v2lo = v2hi = p12 = None

        pvus1_ci = [pci(B['pvus'][j]) for j in range(len(self.bounds))]
        pvus2_ci = dpvus_ci = p_pvus = None
        if B2:
            pvus2_ci  = [pci(B2['pvus'][j]) for j in range(len(self.bounds))]
            dpvus_ci  = [delta_pci(B['pvus'][j], B2['pvus'][j])
                         for j in range(len(self.bounds))]
            p_pvus    = [p_greater(B['pvus'][j], B2['pvus'][j])
                         for j in range(len(self.bounds))]

        sgi1m,sgi1lo,sgi1hi = pci(B['sgi'])
        sgi2m = sgi2lo = sgi2hi = None
        if B2:
            sgi2m,sgi2lo,sgi2hi = pci(B2['sgi'])

        mvf1_lo,mvf1_hi = wilson_ci(int(np.isnan(pt1['auc_s']).sum()), self.n_grid)
        icv1_lo,icv1_hi = wilson_ci(n_exc1, n_chr1)
        mvf2_lo = mvf2_hi = icv2_lo = icv2_hi = None
        if B2:
            mvf2_lo,mvf2_hi = wilson_ci(int(np.isnan(pt2['auc_s']).sum()), self.n_grid)
            icv2_lo,icv2_hi = wilson_ci(n_exc2, n_chr2)

        return VUSResult(
            naive1=naive1, naive2=naive2,
            pt1=pt1, pt2=pt2,
            vus1=(v1m,v1lo,v1hi), vus2=(v2m,v2lo,v2hi) if B2 else None,
            dvus=dv, p_1gt2=p12,
            pvus1_ci=pvus1_ci, pvus2_ci=pvus2_ci,
            dpvus_ci=dpvus_ci, p_pvus=p_pvus,
            sgi1=(sgi1m,sgi1lo,sgi1hi),
            sgi2=(sgi2m,sgi2lo,sgi2hi) if B2 else None,
            mvf1=(pt1['mvf'],mvf1_lo,mvf1_hi),
            mvf2=(pt2['mvf'],mvf2_lo,mvf2_hi) if B2 else None,
            icv1=(icv1,icv1_lo,icv1_hi),
            icv2=(icv2,icv2_lo,icv2_hi) if B2 else None,
            sev_grid=self.sg, fpr_grid=self.fpr_grid,
            pvus_labels=self.plabels, bounds=self.bounds,
            vus_msg1=pt1['vus_msg'],
            vus_msg2=pt2['vus_msg'] if pt2 else None,
            sgi_msg1=pt1['sgi_msg'],
            sgi_msg2=pt2['sgi_msg'] if pt2 else None,
            boot1=B, boot2=B2,
        )


class VUSResult:
    """Container for VUSAnalysis results with summary printing."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def _fmt(m, lo, hi, pct=False, dec=3):
        if m is None or (isinstance(m, float) and math.isnan(m)):
            return 'n/e'
        if pct:
            return f'{m:.1f}% [{lo:.1f}%, {hi:.1f}%]'
        return f'{m:.{dec}f} [{lo:.{dec}f}, {hi:.{dec}f}]'

    @staticmethod
    def _fmtp(p):
        if p is None or (isinstance(p, float) and math.isnan(p)):
            return 'n/e'
        return f'{p:.3f}'

    def summary(self):
        F = self._fmt; P = self._fmtp
        print('\n' + '='*65)
        print(' VUS ANALYSIS SUMMARY')
        print('='*65)
        if self.naive1:
            n1 = self.naive1
            print(f' Naive AUC (Test 1): {n1["auc"]:.3f} '
                  f'[{n1["lo"]:.3f}, {n1["hi"]:.3f}]  '
                  f'(Hanley-McNeil, n+={n1["n_pos"]}, n-={n1["n_neg"]})')
        if self.naive2:
            n2 = self.naive2
            print(f' Naive AUC (Test 2): {n2["auc"]:.3f} '
                  f'[{n2["lo"]:.3f}, {n2["hi"]:.3f}]  '
                  f'(n+={n2["n_pos"]}, n-={n2["n_neg"]})')
        print()
        v1 = self.vus1; v2 = self.vus2
        print(f' Global VUS (Test 1): {F(*v1)}   {self.vus_msg1[:40]}')
        if v2 is not None:
            print(f' Global VUS (Test 2): {F(*v2)}   {self.vus_msg2[:40]}')
        if self.dvus:
            print(f' Global \u0394VUS:         {F(*self.dvus[:3])}   '
                  f'P(T1>T2) = {P(self.p_1gt2)}')
        print()
        for j, lbl in enumerate(self.pvus_labels):
            p1 = self.pvus1_ci[j]
            print(f' PVUS {lbl}:')
            print(f'   Test 1: {F(*p1)}', end='')
            if self.pvus2_ci:
                p2 = self.pvus2_ci[j]
                dp = self.dpvus_ci[j]
                pp = self.p_pvus[j]
                print(f'  |  Test 2: {F(*p2)}')
                print(f'   \u0394PVUS: {F(*dp[:3])}   P(T1>T2) = {P(pp)}')
            else:
                print()
        print()
        print(f' SGI (Test 1): {F(self.sgi1[0],self.sgi1[1],self.sgi1[2],pct=True)}')
        if self.sgi2 and self.sgi2[0] is not None:
            print(f' SGI (Test 2): {F(self.sgi2[0],self.sgi2[1],self.sgi2[2],pct=True)}')
        print()
        print(f' MVF (Test 1): {F(self.mvf1[0],self.mvf1[1],self.mvf1[2],pct=True)}')
        print(f' ICV (Test 1): {F(self.icv1[0],self.icv1[1],self.icv1[2],pct=True)}')
        print('='*65)
