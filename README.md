# CAD-VUS-Analysis

**Volume Under the ROC Surface (VUS) — Complete Analysis Pipeline**

A severity-stratified diagnostic accuracy framework addressing spectrum bias.

---

## What This Repository Provides

| Module | Description |
|--------|-------------|
| `src/vus_analysis.py` | Core VUS analysis class — import and use in your own code |
| `src/run_simulations.py` | Full simulation producing all paper tables and figures |
| `src/make_figures.py` | All publication figures (run after simulations) |
| `stan/vus_meta.stan` | Bayesian VUS meta-analysis Stan model |
| `data/` | Example datasets (simulated) |
| `outputs/` | Results saved here on first run |

---

## Quick Start

```bash
pip install numpy pandas scipy scikit-learn matplotlib
python run_example.py        # both simulations, all statistics
python src/make_figures.py   # all figures → outputs/
```

---

## Key Features

### Coverage-First VUS
Global VUS is estimated only when the observed severity range covers ≥ 50% of the full possible range. A severe-only study returns:

```
Global VUS cannot be estimated: observed range [34, 59] covers only
43% of the full range [0, 60] (minimum required: 50%)
```

### PVUS — Partial VUS by Severity Region
Estimated independently per region (Mild 0–22, Intermediate 23–32, Severe ≥ 33).  
Identity: **VUS = weighted mean(PVUS_mild, PVUS_intermediate, PVUS_severe)** holds exactly.

### ΔVUS and ΔPVUS with Bayesian P(A > B)
Every comparison reports:
- 95% bootstrap credible interval
- **P(A > B)**: fraction of bootstrap draws where A exceeds B

This resolves the limitation of CI-boundary checking when intervals overlap modestly.

### Naive AUC with Hanley-McNeil 95% CI
The standard analytic confidence interval for AUC comparisons.

### SGI — Spectrum Gradient Index
```
SGI = [AUC_fit(s_max) − AUC_fit(s_min)] / AUC_fit(s_max) × 100%
```
- Uses fitted binormal surface (stable)
- Denominator = AUC_severe (bounded [0%, 100%])
- Requires ≥ 50% severity coverage

### MVF, ICV Quality Measures
- **MVF**: Missing Volume Fraction with Wilson 95% CI
- **ICV**: Imprecision of Covered Volume with Wilson 95% CI

---

## Using Your Own Data

```python
from src.vus_analysis import VUSAnalysis

result = VUSAnalysis(
    data         = your_dataframe,
    score_col    = 'biomarker_value',
    disease_col  = 'has_disease',       # 1 = diseased, 0 = non-diseased
    severity_col = 'syntax_score',
    severity_min = 0.0,                  # FULL possible range
    severity_max = 60.0,
    score_col_2  = 'second_test',        # optional, for paired comparison
).run()

result.summary()
```

### For paired comparison (both tests in same patients):
Set `score_col_2` to the second test score column.

### For unpaired comparison (separate cohorts):
Run two `VUSAnalysis` instances separately, then compare PVUS values region by region.

---

## Input Data Format

One row per patient:

| Column | Type | Description |
|--------|------|-------------|
| `disease_status` | int | 1 = disease present, 0 = disease absent |
| `syntax_score` | float | Continuous severity variable |
| `test_score` | float | Continuous diagnostic test score |

Column names are fully configurable.

---

## PVUS Region Boundaries

Default: SYNTAX trial tertiles (0–22, 23–32, ≥ 33). Customise with:

```python
VUSAnalysis(..., pvus_breaks=(20.0, 35.0),
            pvus_labels=['Low', 'Medium', 'High'])
```

---

## Interpreting Results

| Statistic | Scale | Good | Fair | Poor |
|-----------|-------|------|------|------|
| VUS / PVUS | 0.5 → 1.0 | > 0.80 | 0.70–0.80 | < 0.70 |
| MVF | 0 → 100% | < 15% | 15–35% | > 35% |
| ICV | 0 → 100% | < 15% | 15–30% | > 30% |
| SGI | 0 → 100% | < 20% | 20–40% | > 40% |

---

## Bayesian Meta-Analysis (Stan)

For pooling multiple studies, see `stan/vus_meta.stan`.

Install Stan: `pip install cmdstanpy && python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"`

---
