#!/usr/bin/env python3
"""
finalize.py — close out every remaining unverified number and stale figure
in the Safety Science submission.

Produces, all from mat_coded_dataset.csv:

  TABLE 1 / S2   AUC, F1, precision, recall for six models (18-var, 5 seeds)
  TABLE S1       the TRUE collinear partner and |r| for each excluded variable
  TABLE S3       class-imbalance base/balanced absolute AUC
  TABLE S4       BDeu equivalent-sample-size sensitivity of the five dP values
  figS1          dendrogram (24-var Gower/Ward, the input clustering used)
  figS3          bootstrap AUC, out-of-fold vs in-sample, side by side
  figS4          full out-of-fold permutation importance, all 18 variables

Usage: python3 finalize.py --csv mat_coded_dataset.csv --out figures
"""
import argparse, json, os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              StackingClassifier)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score)
from sklearn.inspection import permutation_importance
import gower

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="mat_coded_dataset.csv")
ap.add_argument("--out", default="figures")
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
LOG = []
def say(s):
    print(s); LOG.append(s)

PAL = {"c1": "#C94F4A", "c2": "#E8943A", "c3": "#4AACB0", "c4": "#5B8DB8",
       "c5": "#8B7355", "c6": "#D4A853", "grey": "#9B9B9B", "dark": "#2C2C2C"}
FS_LABEL, FS_AXIS, FS_TICK, FS_BAR, FS_PANEL = 26, 16, 15, 10, 28
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.facecolor": "#F8F6F2",
                     "figure.facecolor": "white", "savefig.facecolor": "white"})

LAYERS = [("M1", ["v01_agent_class", "v02_delivery_complexity",
                  "v03_acquisition_difficulty", "v04_weaponization_level"], "c1"),
          ("M2", ["v05_actor_type", "v06_suicide_operation",
                  "v07_team_coordination", "v08_claimed_responsibility"], "c2"),
          ("M3", ["v09_coordinated_attack", "v10_target_selectivity",
                  "v11_operational_complexity", "v12_civilian_targeting",
                  "v15_campaign_attacks"], "c6"),
          ("M4", ["v13_conflict_zone", "v14_region_instability"], "c3"),
          ("M5", ["v16_group_sophistication", "v17_training_evidence",
                  "v18_international_links"], "c5"),
          ("M6", ["v19_state_sponsorship", "v20_cross_border",
                  "v21_ideological_motivation"], "c4"),
          ("M7", ["v22_mass_casualty_intent", "v23_propaganda_intent",
                  "v24_wmd_signaling"], "grey")]
V2L = {v: c for c, vs, _ in LAYERS for v in vs}
L2C = {c: PAL[k] for c, _, k in LAYERS}

df = pd.read_csv(A.csv)
ALLV = [c for c in df.columns if c.startswith("v")]
DROP6 = ["v02_delivery_complexity", "v03_acquisition_difficulty",
         "v04_weaponization_level", "v07_team_coordination",
         "v17_training_evidence", "v23_propaganda_intent"]
FEAT = [c for c in ALLV if c not in DROP6]
X, X24 = df[FEAT].values.astype(float), df[ALLV].values.astype(float)
y = df["lethality"].values.astype(int)
SEEDS = [42, 123, 456, 789, 2025]
M4P = dict(learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.6,
           max_features=0.6, min_samples_leaf=3)
say(f"N={len(df)} lethal={y.sum()} analytical features={len(FEAT)}")

# ── TABLE S1 : true collinear partner for each excluded variable ────────
say("\n=== TABLE S1 : excluded variables, true strongest partner ===")
C = df[ALLV].corr().abs()
say(f"{'excluded':<30}{'strongest partner':<30}{'|r|':>7}   {'>=0.875?':>9}")
s1 = []
for v in DROP6:
    others = [u for u in ALLV if u != v]
    best = max(others, key=lambda u: C.loc[v, u])
    r = float(C.loc[v, best])
    s1.append(dict(excluded=v, partner=best, r=round(r, 4),
                   meets=bool(r >= 0.875)))
    say(f"{v:<30}{best:<30}{r:>7.3f}   {'YES' if r>=0.875 else 'NO':>9}")

# ── TABLE 1 / S2 : full metric set ─────────────────────────────────────
say("\n=== TABLE 1 / S2 : six models, 18-var, 5 seeds ===")
def mk(seed):
    return {"B0": LogisticRegression(class_weight="balanced", max_iter=500,
                                     random_state=seed),
            "M1": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                         random_state=seed, n_jobs=-1),
            "M2": GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                             learning_rate=0.05, subsample=0.8,
                                             random_state=seed),
            "M3": GradientBoostingClassifier(n_estimators=300, max_depth=5,
                                             learning_rate=0.03, subsample=0.8,
                                             min_samples_leaf=5, random_state=seed),
            "M4": GradientBoostingClassifier(random_state=seed, **M4P),
            "M6": StackingClassifier(
                [("lr", LogisticRegression(class_weight="balanced", max_iter=500)),
                 ("rf", RandomForestClassifier(n_estimators=200,
                                               class_weight="balanced",
                                               random_state=seed)),
                 ("gbt", GradientBoostingClassifier(random_state=seed))],
                final_estimator=LogisticRegression(max_iter=500), cv=5)}
tab1 = {}
say(f"{'model':<6}{'AUC':>16}{'F1':>16}{'Prec':>16}{'Recall':>16}")
for name in ["B0", "M1", "M2", "M3", "M4", "M6"]:
    a, f, p, r = [], [], [], []
    for s in SEEDS:
        cv = StratifiedKFold(5, shuffle=True, random_state=s)
        pr = cross_val_predict(mk(s)[name], X, y, cv=cv, method="predict_proba")[:, 1]
        pd_ = (pr >= 0.5).astype(int)
        a.append(roc_auc_score(y, pr)); f.append(f1_score(y, pd_, zero_division=0))
        p.append(precision_score(y, pd_, zero_division=0))
        r.append(recall_score(y, pd_, zero_division=0))
    tab1[name] = {k: [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)]
                  for k, v in (("auc", a), ("f1", f), ("prec", p), ("rec", r))}
    say(f"{name:<6}{np.mean(a):>8.3f}+-{np.std(a):<6.3f}"
        f"{np.mean(f):>8.3f}+-{np.std(f):<6.3f}"
        f"{np.mean(p):>8.3f}+-{np.std(p):<6.3f}"
        f"{np.mean(r):>8.3f}+-{np.std(r):<6.3f}")

# ── TABLE S3 : class imbalance, absolute values ────────────────────────
say("\n=== TABLE S3 : class-imbalance sensitivity (single seed 42) ===")
cv1 = StratifiedKFold(5, shuffle=True, random_state=42)
s3 = {}
for nm, base, bal in (
    ("B0", LogisticRegression(max_iter=500, random_state=42),
     LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)),
    ("M1", RandomForestClassifier(n_estimators=200, random_state=42),
     RandomForestClassifier(n_estimators=200, class_weight="balanced",
                            random_state=42))):
    ab = roc_auc_score(y, cross_val_predict(base, X, y, cv=cv1,
                                            method="predict_proba")[:, 1])
    bb = roc_auc_score(y, cross_val_predict(bal, X, y, cv=cv1,
                                            method="predict_proba")[:, 1])
    s3[nm] = dict(base=round(float(ab), 4), balanced=round(float(bb), 4),
                  delta=round(float(bb - ab), 4))
w = np.where(y == 1, (len(y) - y.sum()) / y.sum(), 1.0)
ab = roc_auc_score(y, cross_val_predict(
    GradientBoostingClassifier(random_state=42, **M4P), X, y, cv=cv1,
    method="predict_proba")[:, 1])
oofw = np.zeros(len(y))
for tri, tei in cv1.split(X, y):
    m = GradientBoostingClassifier(random_state=42, **M4P)
    m.fit(X[tri], y[tri], sample_weight=w[tri])
    oofw[tei] = m.predict_proba(X[tei])[:, 1]
bb = roc_auc_score(y, oofw)
s3["M4"] = dict(base=round(float(ab), 4), balanced=round(float(bb), 4),
                delta=round(float(bb - ab), 4))
for k, v in s3.items():
    say(f"  {k}: base={v['base']:.3f}  balanced={v['balanced']:.3f}  "
        f"delta={v['delta']:+.3f}")

# ── TABLE S4 : BDeu equivalent-sample-size sensitivity ─────────────────
say("\n=== TABLE S4 : BDeu ESS sensitivity ===")
s4 = {}
try:
    try:
        from pgmpy.models import DiscreteBayesianNetwork as BN
    except Exception:
        from pgmpy.models import BayesianNetwork as BN
    from pgmpy.estimators import BayesianEstimator
    from pgmpy.inference import VariableElimination
    BV = ["v01_agent_class", "v05_actor_type", "v13_conflict_zone",
          "v15_campaign_attacks", "v16_group_sophistication",
          "v19_state_sponsorship", "v22_mass_casualty_intent", "lethality"]
    d = df[BV].copy()
    for c in ["v01_agent_class", "v05_actor_type", "v16_group_sophistication"]:
        d[c] = (d[c] > d[c].median()).astype(int)
    d = d.astype(int)
    E = [("v13_conflict_zone", "v15_campaign_attacks"),
         ("v13_conflict_zone", "v01_agent_class"),
         ("v19_state_sponsorship", "v01_agent_class"),
         ("v19_state_sponsorship", "v16_group_sophistication"),
         ("v16_group_sophistication", "v05_actor_type"),
         ("v16_group_sophistication", "v01_agent_class"),
         ("v05_actor_type", "v22_mass_casualty_intent"),
         ("v01_agent_class", "v22_mass_casualty_intent"),
         ("v15_campaign_attacks", "v22_mass_casualty_intent"),
         ("v22_mass_casualty_intent", "lethality"),
         ("v01_agent_class", "lethality"), ("v05_actor_type", "lethality")]
    NODES = ["v22_mass_casualty_intent", "v01_agent_class",
             "v15_campaign_attacks", "v13_conflict_zone", "v19_state_sponsorship"]
    say(f"{'ESS':<6}{'baseline':>10}" + "".join(f"{n.split('_',1)[1][:14]:>16}"
                                                for n in NODES))
    for ess in (1, 5, 10, 20, 50):
        bn = BN(E)
        bn.fit(d, estimator=BayesianEstimator, prior_type="BDeu",
               equivalent_sample_size=ess)
        inf = VariableElimination(bn)
        q = lambda ev: float(inf.query(["lethality"], evidence=ev).values[1])
        row = {n: round(q({n: 1}) - q({n: 0}), 4) for n in NODES}
        s4[ess] = dict(baseline=round(q({}), 4), deltas=row)
        say(f"{ess:<6}{q({}):>10.4f}" + "".join(f"{row[n]:>+16.4f}" for n in NODES))
except Exception as e:
    say(f"  BN block failed: {e}")

# ── figS1 : dendrogram on the 24-variable Gower matrix ─────────────────
say("\n=== FIGURES ===")
G = gower.gower_matrix(X24)
Z = linkage(squareform(G, checks=False), method="ward")
lab2 = fcluster(Z, 2, criterion="maxclust")
cut = float(np.sort(Z[:, 2])[-1])
fig, ax = plt.subplots(figsize=(20, 10))
dendrogram(Z, ax=ax, color_threshold=cut, no_labels=True,
           above_threshold_color=PAL["grey"])
ax.axhline(cut, ls="--", lw=2, color=PAL["c1"],
           label=f"$k=2$ cut (height = {cut:.2f})")
ax.set_xlabel(f"Incidents ($n$ = {len(df)}), leaves suppressed", fontsize=FS_AXIS)
ax.set_ylabel("Ward linkage distance", fontsize=FS_AXIS)
ax.set_title("Hierarchical clustering of MAT-coded incidents\n"
             "Ward linkage on the 24-variable Gower distance matrix",
             fontsize=FS_LABEL - 4, pad=16)
ax.legend(fontsize=FS_TICK, loc="upper right", framealpha=0.92)
ax.tick_params(labelsize=FS_TICK)
fig.savefig(f"{A.out}/figS1_dendrogram.png", dpi=200, bbox_inches="tight")
plt.close()
sizes = sorted(np.bincount(lab2)[1:].tolist())
say(f"  figS1: cut height={cut:.4f}  cluster sizes={sizes}")

# ── figS3 : bootstrap, out-of-fold vs in-sample ────────────────────────
m4 = GradientBoostingClassifier(random_state=42, **M4P)
oof = cross_val_predict(m4, X, y, method="predict_proba",
                        cv=StratifiedKFold(5, shuffle=True, random_state=42))[:, 1]
ins = GradientBoostingClassifier(random_state=42, **M4P).fit(X, y).predict_proba(X)[:, 1]
rng = np.random.RandomState(42)
bo, bi = [], []
for _ in range(1000):
    idx = rng.choice(len(y), len(y), replace=True)
    if len(np.unique(y[idx])) < 2:
        continue
    bo.append(roc_auc_score(y[idx], oof[idx]))
    bi.append(roc_auc_score(y[idx], ins[idx]))
fig, ax = plt.subplots(figsize=(20, 10))
for vals, col, nm in ((bo, PAL["c4"], "out-of-fold predictions"),
                      (bi, PAL["c1"], "in-sample predictions")):
    ax.hist(vals, bins=45, alpha=0.62, color=col, edgecolor="white",
            label=f"{nm}: mean {np.mean(vals):.3f}, "
                  f"95% PI [{np.percentile(vals,2.5):.3f}, "
                  f"{np.percentile(vals,97.5):.3f}]")
    ax.axvline(np.mean(vals), ls="--", lw=2.2, color=col)
ax.set_xlabel("Bootstrap AUC ($n$ = 1,000 resamples)", fontsize=FS_AXIS)
ax.set_ylabel("Frequency", fontsize=FS_AXIS)
ax.set_title("Bootstrap AUC for the tuned GBT: the cost of resampling in sample",
             fontsize=FS_LABEL - 4, pad=16)
ax.legend(fontsize=FS_TICK - 1, loc="upper left", framealpha=0.92)
ax.tick_params(labelsize=FS_TICK)
fig.savefig(f"{A.out}/figS3_bootstrap_auc.png", dpi=200, bbox_inches="tight")
plt.close()
say(f"  figS3: OOF mean={np.mean(bo):.4f} CI=[{np.percentile(bo,2.5):.4f},"
    f"{np.percentile(bo,97.5):.4f}] | in-sample mean={np.mean(bi):.4f} "
    f"CI=[{np.percentile(bi,2.5):.4f},{np.percentile(bi,97.5):.4f}]")

# ── figS4 : full out-of-fold permutation importance ────────────────────
imp = np.zeros(len(FEAT)); n = 0
for s in SEEDS:
    for tri, tei in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
        f_ = GradientBoostingClassifier(random_state=s, **M4P).fit(X[tri], y[tri])
        imp += permutation_importance(f_, X[tei], y[tei], n_repeats=30,
                                      random_state=s,
                                      scoring="roc_auc").importances_mean
        n += 1
imp = pd.Series(imp / n, index=FEAT).sort_values()
fig, ax = plt.subplots(figsize=(20, 11))
ax.barh(range(len(imp)), imp.values,
        color=[L2C[V2L[t]] for t in imp.index], edgecolor="white", height=0.74)
ax.set_yticks(range(len(imp)))
ax.set_yticklabels([f"{t[4:].replace('_',' ').title()}  ({V2L[t]})"
                    for t in imp.index], fontsize=FS_BAR + 2)
ax.axvline(0, color=PAL["dark"], lw=1.2)
ax.set_xlabel("Permutation importance, out-of-fold (mean AUC decrease)",
              fontsize=FS_AXIS)
ax.set_title("Full importance profile, all 18 analytical variables",
             fontsize=FS_LABEL - 2, pad=16)
ax.tick_params(labelsize=FS_TICK)
ax.legend(handles=[mpatches.Patch(color=L2C[c], label=c) for c, _, _ in LAYERS],
          fontsize=FS_TICK - 2, loc="lower right", ncol=2, framealpha=0.92)
fig.savefig(f"{A.out}/figS4_full_importance.png", dpi=200, bbox_inches="tight")
plt.close()
say("  figS4 full ranking (low to high):")
for k, v in imp.items():
    say(f"    {k:<32} {V2L[k]}  {v:+.4f}")

json.dump(dict(table1=tab1, tableS1=s1, tableS3=s3, tableS4=s4),
          open(f"{A.out}/finalize.json", "w"), indent=2, default=float)
open(f"{A.out}/finalize_values.txt", "w").write("\n".join(LOG))
print(f"\n-> {A.out}/  figS1, figS3, figS4 + finalize.json + finalize_values.txt")
