# MAT / AcciMap — Chemical Terrorism Incident Analysis

Data and code for:

> **The Missing Upper Layers: A Quantitative Test of AcciMap on 203 Chemical
> Terrorism Incidents, 1970-2021**
> Cha, Shin, Yoo, Lee, Park, Jang & Kang. Submitted to *Safety Science*, 2026.
> Corresponding author: Ku Kang (bisu9082@gmail.com)

Everything reported in the manuscript and the Supplementary Material can be
recomputed from this repository. Released under the MIT License.

---

## Read this first: one coded variable was contaminated by the outcome

During preparation we found that two of the 24 MAT variables were derived
from the outcome they were used to predict. The original coding rule was:

```python
mc_intent = 1 if ((nkill + nwound) > 50 or (agent_class == 2 and tgt_sel <= 1)) else 0
lethality = 1 if nkill >= 1 else 0
```

`v22_mass_casualty_intent` is a function of `nkill`, and `nkill` is the
definition of the outcome. `v24_wmd_signaling` is in turn a function of
`v22`, so the contamination propagates.

Measured on the released data:

| | value |
|---|---|
| incidents with `mc_intent = 1` | 21 |
| …fired by the casualty rule | 21 (all of them) |
| …fired by the agent rule | 0 (never once) |
| corr(v22, lethality) | 0.4201 |
| corr(v24, lethality) | 0.3044 |
| corr(v14, lethality), uncontaminated reference | 0.4230 |

This is complete leakage, not partial: `v22` measures casualty magnitude, not
intent. **Both variables are excluded from every result reported in the
paper.** The predictive analysis uses 16 variables (24 − 6 collinear − 2
contaminated); the clustering uses 22 (24 − 2).

Recoding was not available: the agent branch never fires, so the data contain
no alternative definition, and returning to source documents to recode 203
incidents was not feasible at this stage.

The contaminated script is kept in this repository as
`code/analysis_main_original_CONTAMINATED.py` for transparency. **It did not
produce the published numbers.** Do not use it to reproduce the paper.

---

## Layout

```
data/
  mat_coded_public.csv      N = 203; v01–v24, lethality, year
code/
  rerun_clean.py            clean 16-variable re-run: clustering, ML, BN
  regen_final_v2.py         fig1-fig5, figS1, figS3, figS4 + reported values
  regen_s2_s5.py            figS2 correlation matrix, figS5 temporal validation
  finalize.py               earlier supplementary figures and tables
  verify_acciterror.py      independent recomputation of reported values
  verify_v2.py              second-pass checks
  requirements.txt
  analysis_main_original_CONTAMINATED.py   superseded; see warning above
```

### What is and is not in the released data

`mat_coded_public.csv` carries the 24 MAT-coded variables, the binary
lethality outcome, and the incident year. Fields that would redistribute the
Global Terrorism Database itself — GTD event ID, country, perpetrator name,
raw casualty counts, agent label — are withheld. The GTD is available to
researchers directly from START, University of Maryland
(<https://start.umd.edu/gtd/>) under its own terms.

The withheld fields are not needed for reproduction: every quantity in the
manuscript was recomputed from `mat_coded_public.csv` alone and matched the
internal dataset identically.

---

## Reproducing

```bash
pip install -r code/requirements.txt

python3 code/rerun_clean.py     --csv data/mat_coded_public.csv --out clean
python3 code/regen_final_v2.py  --csv data/mat_coded_public.csv --out final
python3 code/regen_s2_s5.py     --csv data/mat_coded_public.csv --out figs_s
```

**Use the pinned versions.** `requirements.txt` pins `scikit-learn==1.9.1`
and `numpy==2.5.3` because the tree-ensemble AUCs are not stable across
scikit-learn releases: on 1.7.2 the same code and the same data give M1
0.759 rather than 0.770 and M6 0.767 rather than 0.769, while the logistic
baseline, the clustering, the bootstrap, the correlation matrix and the
Bayesian network reproduce to four decimal places on both. The reported
environment is Python 3.13.2 with scikit-learn 1.9.1, numpy 2.5.3 and
pgmpy 1.1.0; `pgmpy` is pinned separately because its network API changed
across versions.

### Expected values

Clustering (Gower distance, Ward linkage, 22 variables):

| | |
|---|---|
| cluster sizes | 182 / 21 |
| silhouette | 0.644 |
| lethality rate by cluster | 0.220 / 0.381 |

Classification AUC, 5-fold cross-validated, 16 variables:

| model | AUC | AUC before removing v22/v24 |
|---|---|---|
| B0 logistic | **0.7784** | 0.8004 |
| M1 | 0.7702 | 0.8006 |
| M2 | 0.7588 | 0.7885 |
| M3 | 0.7648 | 0.7961 |
| M4 | **0.7794** | 0.8047 |
| M6 | 0.7685 | 0.7917 |

Removing the contamination costs 0.022–0.031 AUC, uniformly across models.
B0 and M4 then differ by 0.001: the linear model is not improved upon.

Out-of-fold permutation importance:

| variable | Δ AUC |
|---|---|
| v14_region_instability | **+0.1364** |
| v06_suicide_operation | +0.0351 |
| v18_international_links | +0.0223 |
| v12_civilian_targeting | +0.0176 |
| v10_target_selectivity | +0.0124 |

The single governance proxy outranks the second-placed adversary variable by
a factor of 3.9.

Out-of-period performance, reported as found: chronological holdout B0 0.583,
M4 0.689; rolling-origin 0.684 ± 0.097 over folds
[0.577, 0.667, 0.600, 0.846, 0.732]; within-period CV 0.779. Bootstrap
out-of-fold 0.759 [0.664, 0.841] against in-sample 0.853 [0.788, 0.910].

Bayesian network (BDeu, pgmpy 1.1.0), baseline P(lethal) = 0.242:

| node set high | Δ P(lethal) |
|---|---|
| agent_class | +0.374 |
| region_instability | +0.351 |
| campaign | +0.313 |
| conflict_zone | +0.291 |
| group_sophistication | +0.071 |
| actor_type | +0.064 |
| state_sponsorship | +0.040 |

---

## What was withdrawn

An interaction grid crossing regional stability with agent class appeared in
an earlier draft. It is non-monotonic (stable + high-hazard agent 0.716
against unstable + high-hazard agent 0.548), and with only 11 high-hazard
incidents each cell holds 2–4 events. The panel was removed rather than
interpreted; the figure caption states why.

A SHAP beeswarm figure present in an early draft could not be reproduced and
was deleted. It is not in this repository and should not be reinstated.

---

## Citing

Please cite the paper. If you use the MAT coding scheme, note the
contamination finding above: when building a coding scheme from an incident
database, check each predictor by its *definition*, not its name — a variable
called "intent" whose rule reads the outcome field is leakage. This one was
caught because the coding script was slated for release.

## License

MIT. See `LICENSE`.
