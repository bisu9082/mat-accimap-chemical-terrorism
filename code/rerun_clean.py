#!/usr/bin/env python3
"""
rerun_clean.py — rebuild every result after removing the two variables
contaminated by the outcome.

v22_mass_casualty_intent fires entirely on (nkill + nwound) > 50, and
lethality is defined as nkill >= 1. The variable is therefore a function of
the outcome's own input, not a measure of intent. v24_wmd_signaling is
defined as f(agent_class, mc_intent) and inherits the contamination.

Analytical matrix: 24 - 6 collinear - 2 contaminated = 16 variables.

The script answers four questions before anything else:
  Q1  Is Cluster 2 simply the set of 21 incidents with mc_intent = 1?
  Q2  How much of the reported AUC was the contamination worth?
  Q3  Does regional instability remain the leading variable?
  Q4  What does the Bayesian network give once mc_intent is removed and
      regional instability is added in its place?

Usage: python3 rerun_clean.py --csv mat_coded_dataset.csv --out clean
"""
import argparse, json, os, warnings
import numpy as np, pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              StackingClassifier)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     TimeSeriesSplit, RandomizedSearchCV)
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, silhouette_score)
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
import gower

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="mat_coded_dataset.csv")
ap.add_argument("--out", default="clean")
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
LOG = []
def say(s=""):
    print(s); LOG.append(s)

df = pd.read_csv(A.csv)
ALLV = [c for c in df.columns if c.startswith("v")]
y = df["lethality"].values.astype(int)
COLLIN = ["v02_delivery_complexity", "v03_acquisition_difficulty",
          "v04_weaponization_level", "v07_team_coordination",
          "v17_training_evidence", "v23_propaganda_intent"]
LEAK = ["v22_mass_casualty_intent", "v24_wmd_signaling"]
OLD18 = [c for c in ALLV if c not in COLLIN]
CLEAN = [c for c in OLD18 if c not in LEAK]
CLEAN24 = [c for c in ALLV if c not in LEAK]
X_old, X_new = df[OLD18].values.astype(float), df[CLEAN].values.astype(float)
SEEDS = [42, 123, 456, 789, 2025]
R = {}
say(f"N={len(df)}  lethal={y.sum()}")
say(f"old analytical matrix : {len(OLD18)} variables")
say(f"new analytical matrix : {len(CLEAN)} variables  (removed {LEAK})")

# ── Q1 : is Cluster 2 just the mass-casualty set? ──────────────────────
say("\n" + "=" * 68)
say("Q1  Cluster 2 vs the 21 incidents flagged by the contaminated variable")
say("=" * 68)
mc = (df["v22_mass_casualty_intent"] == 1).values
def cluster_on(cols, tag):
    G = gower.gower_matrix(df[cols].values.astype(float))
    Z = linkage(squareform(G, checks=False), method="ward")
    sil = {k: float(silhouette_score(G, fcluster(Z, k, criterion="maxclust"),
                                     metric="precomputed")) for k in range(2, 7)}
    lab = fcluster(Z, 2, criterion="maxclust")
    small = np.argmin(np.bincount(lab)[1:]) + 1
    mem = lab == small
    inter = int((mem & mc).sum())
    say(f"  [{tag}] sil k=2..6 = {[round(sil[k],4) for k in range(2,7)]}")
    say(f"           sizes = {sorted(np.bincount(lab)[1:].tolist())}   "
        f"small-cluster lethality = {y[mem].mean():.3f}")
    say(f"           small cluster n={mem.sum()}, of which mc_intent=1 : "
        f"{inter}  (Jaccard {inter/max(1,(mem|mc).sum()):.3f})")
    return dict(silhouette=sil, sizes=sorted(np.bincount(lab)[1:].tolist()),
                small_n=int(mem.sum()), overlap_mc=inter,
                jaccard=round(inter / max(1, (mem | mc).sum()), 4),
                small_lethality=round(float(y[mem].mean()), 4),
                labels=lab.tolist())
R["cluster_all24"] = cluster_on(ALLV, "all 24, as published")
R["cluster_clean22"] = cluster_on(CLEAN24, "22, contaminated removed")

# ── Q2 / Q3 : models and importance, old vs new ────────────────────────
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
            "M4": GradientBoostingClassifier(random_state=seed, learning_rate=0.01,
                                             max_depth=3, n_estimators=100,
                                             subsample=0.6, max_features=0.6,
                                             min_samples_leaf=3),
            "M6": StackingClassifier(
                [("lr", LogisticRegression(class_weight="balanced", max_iter=500)),
                 ("rf", RandomForestClassifier(n_estimators=200,
                                               class_weight="balanced",
                                               random_state=seed)),
                 ("gbt", GradientBoostingClassifier(random_state=seed))],
                final_estimator=LogisticRegression(max_iter=500), cv=5)}

def metrics(Xi, tag):
    say(f"\n  [{tag}]  {'model':<5}{'AUC':>16}{'F1':>16}{'Prec':>16}{'Recall':>16}")
    out = {}
    for nm in ["B0", "M1", "M2", "M3", "M4", "M6"]:
        a, f, p, r = [], [], [], []
        for s in SEEDS:
            cv = StratifiedKFold(5, shuffle=True, random_state=s)
            pr = cross_val_predict(mk(s)[nm], Xi, y, cv=cv,
                                   method="predict_proba")[:, 1]
            pd_ = (pr >= 0.5).astype(int)
            a.append(roc_auc_score(y, pr)); f.append(f1_score(y, pd_, zero_division=0))
            p.append(precision_score(y, pd_, zero_division=0))
            r.append(recall_score(y, pd_, zero_division=0))
        out[nm] = {k: [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)]
                   for k, v in (("auc", a), ("f1", f), ("prec", p), ("rec", r))}
        say(f"         {nm:<5}{np.mean(a):>8.3f}+-{np.std(a):<6.3f}"
            f"{np.mean(f):>8.3f}+-{np.std(f):<6.3f}"
            f"{np.mean(p):>8.3f}+-{np.std(p):<6.3f}"
            f"{np.mean(r):>8.3f}+-{np.std(r):<6.3f}")
    return out

say("\n" + "=" * 68)
say("Q2  What the contamination was worth")
say("=" * 68)
R["models_old18"] = metrics(X_old, "18 var, as published")
R["models_clean16"] = metrics(X_new, "16 var, clean")
say("\n  AUC change by model:")
for nm in ["B0", "M1", "M2", "M3", "M4", "M6"]:
    o = R["models_old18"][nm]["auc"][0]; n = R["models_clean16"][nm]["auc"][0]
    say(f"    {nm:<5}{o:.4f} -> {n:.4f}   {n-o:+.4f}")

say("\n" + "=" * 68)
say("Q3  Importance ranking on the clean matrix (out-of-fold)")
say("=" * 68)
imp = np.zeros(len(CLEAN)); k = 0
for s in SEEDS:
    for tri, tei in StratifiedKFold(5, shuffle=True, random_state=s).split(X_new, y):
        f_ = GradientBoostingClassifier(random_state=s, learning_rate=0.01,
                                        max_depth=3, n_estimators=100,
                                        subsample=0.6, max_features=0.6,
                                        min_samples_leaf=3).fit(X_new[tri], y[tri])
        imp += permutation_importance(f_, X_new[tei], y[tei], n_repeats=30,
                                      random_state=s,
                                      scoring="roc_auc").importances_mean
        k += 1
imp = pd.Series(imp / k, index=CLEAN).sort_values(ascending=False)
R["importance_clean"] = {a: round(float(b), 4) for a, b in imp.items()}
for a, b in imp.items():
    say(f"    {a:<34}{b:+.4f}")

# temporal, clean matrix
tr, te = (df["year"] < 2000).values, (df["year"] >= 2000).values
R["holdout_clean"] = {}
say("\n  chronological hold-out (clean matrix):")
for nm in ["B0", "M4"]:
    m = mk(42)[nm].fit(X_new[tr], y[tr])
    v = float(roc_auc_score(y[te], m.predict_proba(X_new[te])[:, 1]))
    R["holdout_clean"][nm] = round(v, 4)
    say(f"    {nm}: {v:.4f}")
o = np.argsort(df["year"].values, kind="stable")
Xo, yo = X_new[o], y[o]
folds = []
for a_, b_ in TimeSeriesSplit(n_splits=5, test_size=15).split(Xo):
    if len(np.unique(yo[b_])) < 2:
        continue
    m = mk(42)["M4"].fit(Xo[a_], yo[a_])
    folds.append(float(roc_auc_score(yo[b_], m.predict_proba(Xo[b_])[:, 1])))
R["rolling_clean"] = dict(folds=[round(f, 4) for f in folds],
                          mean=round(float(np.mean(folds)), 4),
                          sd=round(float(np.std(folds)), 4))
say(f"    rolling: {np.mean(folds):.4f} +- {np.std(folds):.4f}  "
    f"{[round(f,3) for f in folds]}")

# ── Q4 : Bayesian network without mc_intent, with region instability ───
say("\n" + "=" * 68)
say("Q4  Bayesian network, contaminated node removed")
say("=" * 68)
try:
    try:
        from pgmpy.models import DiscreteBayesianNetwork as BN
    except Exception:
        from pgmpy.models import BayesianNetwork as BN
    from pgmpy.estimators import BayesianEstimator
    from pgmpy.inference import VariableElimination
    BV = ["v14_region_instability", "v13_conflict_zone", "v19_state_sponsorship",
          "v16_group_sophistication", "v05_actor_type", "v01_agent_class",
          "v15_campaign_attacks", "lethality"]
    d = df[BV].copy()
    for c in ["v01_agent_class", "v05_actor_type", "v16_group_sophistication"]:
        d[c] = (d[c] > d[c].median()).astype(int)
    d = d.astype(int)
    E = [("v14_region_instability", "v13_conflict_zone"),
         ("v14_region_instability", "v15_campaign_attacks"),
         ("v14_region_instability", "lethality"),
         ("v13_conflict_zone", "v15_campaign_attacks"),
         ("v13_conflict_zone", "v01_agent_class"),
         ("v19_state_sponsorship", "v01_agent_class"),
         ("v19_state_sponsorship", "v16_group_sophistication"),
         ("v16_group_sophistication", "v05_actor_type"),
         ("v16_group_sophistication", "v01_agent_class"),
         ("v01_agent_class", "lethality"),
         ("v05_actor_type", "lethality"),
         ("v15_campaign_attacks", "lethality")]
    bn = BN(E)
    bn.fit(d, estimator=BayesianEstimator, prior_type="BDeu",
           equivalent_sample_size=5)
    inf = VariableElimination(bn)
    q = lambda ev: float(inf.query(["lethality"], evidence=ev).values[1])
    base = q({})
    say(f"  nodes={len(BV)}  edges={len(E)}   baseline P(lethal) = {base:.4f}")
    sens = {}
    for v in ["v14_region_instability", "v01_agent_class", "v15_campaign_attacks",
              "v13_conflict_zone", "v19_state_sponsorship", "v05_actor_type"]:
        p0, p1 = q({v: 0}), q({v: 1})
        sens[v] = dict(p0=round(p0, 4), p1=round(p1, 4), delta=round(p1 - p0, 4))
    for v, s in sorted(sens.items(), key=lambda kv: -kv[1]["delta"]):
        say(f"    {v:<34}P1={s['p1']:.4f}  dP={s['delta']:+.4f}")
    R["bn_clean"] = dict(baseline=round(base, 4), n_edges=len(E), sensitivity=sens)
    grid = {f"inst={i},agent={a}": round(q({"v14_region_instability": i,
                                            "v01_agent_class": a}), 4)
            for i in (0, 1) for a in (0, 1)}
    R["bn_clean"]["interaction"] = grid
    say(f"  interaction grid: {grid}")
    scen = {"S0": round(base, 4),
            "no advanced agent": round(q({"v01_agent_class": 0}), 4),
            "no campaign": round(q({"v15_campaign_attacks": 0}), 4),
            "stable region": round(q({"v14_region_instability": 0}), 4),
            "all three": round(q({"v01_agent_class": 0, "v15_campaign_attacks": 0,
                                  "v14_region_instability": 0}), 4)}
    R["bn_clean"]["scenarios"] = scen
    say(f"  scenarios: {scen}")
except Exception as e:
    say(f"  BN failed: {e}")

json.dump(R, open(f"{A.out}/rerun_clean.json", "w"), indent=2, default=float)
open(f"{A.out}/rerun_clean.txt", "w").write("\n".join(LOG))
print(f"\n-> {A.out}/rerun_clean.json, rerun_clean.txt")
