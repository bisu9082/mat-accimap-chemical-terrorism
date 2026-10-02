#!/usr/bin/env python3
"""
verify_v2.py — replicate the ORIGINAL pipeline configuration exactly, so that
a mismatch means the manuscript is wrong rather than that this script differs.

v1 used its own feature filter and its own hyperparameters, which made several
diffs uninterpretable. v2 pins everything to what analysis_main.py and
experiment_summary.json actually specify.

Usage: python3 verify_v2.py --csv mat_coded_dataset.csv
"""
import argparse, json, warnings
import numpy as np, pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              StackingClassifier)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.cluster import KMeans
import gower

warnings.filterwarnings("ignore")
SEEDS = [42, 123, 456, 789, 2025]
ap = argparse.ArgumentParser(); ap.add_argument("--csv", default="mat_coded_dataset.csv")
a = ap.parse_args()
df = pd.read_csv(a.csv)
ALLV = [c for c in df.columns if c.startswith("v")]
y = df["lethality"].values.astype(int)
OUT = {}

# the six the manuscript/JSON says were dropped
CLAIMED_DROP = ["v02_delivery_complexity", "v03_acquisition_difficulty",
                "v04_weaponization_level", "v07_team_coordination",
                "v17_training_evidence", "v23_propaganda_intent"]
DROP18 = [c for c in ALLV if c not in CLAIMED_DROP]

print("=" * 70)
print("Q1  collinearity: which variables does the r>=0.875 rule remove?")
print("=" * 70)
C = df[ALLV].corr().abs()
pairs = [(ALLV[i], ALLV[j], float(C.iloc[i, j]))
         for i in range(len(ALLV)) for j in range(i + 1, len(ALLV))
         if C.iloc[i, j] >= 0.875]
print(f"pairs with |r| >= 0.875 in the full 24-variable set: {len(pairs)}")
for x, z, r in sorted(pairs, key=lambda t: -t[2]):
    print(f"   {r:.3f}  {x}  x  {z}")
OUT["pairs_ge_875"] = pairs
print(f"\nmanuscript says these 6 were dropped: {CLAIMED_DROP}")
print(f"-> leaves {len(DROP18)} variables")

print("\nQ2  the Fig. S2 problem pair, measured directly:")
for x, z in [("v16_group_sophistication", "v19_state_sponsorship"),
             ("v01_agent_class", "v24_wmd_signaling"),
             ("v01_agent_class", "v15_campaign_attacks")]:
    r = float(C.loc[x, z])
    OUT[f"corr_{x}__{z}"] = r
    flag = "  <-- ABOVE the stated 0.875 cutoff, yet retained" if r >= 0.875 else ""
    print(f"   r({x}, {z}) = {r:.4f}{flag}")

# ── Q3 clustering, on each candidate feature set ────────────────────────
print("\n" + "=" * 70)
print("Q3  clustering — manuscript claims sil(k=2)=0.647, n=182/21, kmeans=0.482")
print("=" * 70)
OUT["clustering"] = {}
for tag, feats in (("all_24", ALLV), ("drop6_18", DROP18)):
    X = df[feats].values.astype(float)
    G = gower.gower_matrix(X)
    Z = linkage(squareform(G, checks=False), method="ward")
    sil = {k: float(silhouette_score(G, fcluster(Z, k, criterion="maxclust"),
                                     metric="precomputed")) for k in range(2, 7)}
    lab = fcluster(Z, 2, criterion="maxclust")
    sizes = sorted(np.bincount(lab)[1:].tolist())
    rates = {int(c): float(y[lab == c].mean()) for c in sorted(set(lab))}
    km = {k: float(silhouette_score(X, KMeans(n_clusters=k, random_state=42,
                                              n_init=20).fit_predict(X)))
          for k in range(2, 7)}
    OUT["clustering"][tag] = dict(silhouette=sil, sizes=sizes, rates=rates,
                                  kmeans=km)
    print(f"  [{tag}]  sil k2..k6 = "
          f"{[round(sil[k],4) for k in range(2,7)]}")
    print(f"           cluster sizes = {sizes}   rates = "
          f"{ {k: round(v,3) for k,v in rates.items()} }")
    print(f"           kmeans k2 = {km[2]:.4f}")

# ── Q4/Q5 models with the ORIGINAL configs ──────────────────────────────
print("\n" + "=" * 70)
print("Q4/Q5  models, using analysis_main.py configs and the JSON's M4 params")
print("=" * 70)
M4P = dict(learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.6,
           max_features=0.6, min_samples_leaf=3)          # from experiment_summary.json


def build(seed):
    return {
        "B0_LR": LogisticRegression(class_weight="balanced", max_iter=500,
                                    random_state=seed),
        "M1_RF": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                        random_state=seed, n_jobs=-1),
        "M2_GBT": GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                             learning_rate=0.05, subsample=0.8,
                                             random_state=seed),
        "M3_GBT_full": GradientBoostingClassifier(n_estimators=300, max_depth=5,
                                                  learning_rate=0.03, subsample=0.8,
                                                  min_samples_leaf=5,
                                                  random_state=seed),
        "M4_GBT_tuned": GradientBoostingClassifier(random_state=seed, **M4P),
        "M6_Stacking": StackingClassifier(
            [("lr", LogisticRegression(class_weight="balanced", max_iter=500)),
             ("rf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                           random_state=seed)),
             ("gbt", GradientBoostingClassifier(random_state=seed))],
            final_estimator=LogisticRegression(max_iter=500), cv=5),
    }


CLAIM_AUC = {"B0_LR": 0.800, "M1_RF": 0.795, "M2_GBT": 0.788,
             "M3_GBT_full": 0.796, "M4_GBT_tuned": 0.813, "M6_Stacking": 0.800}
OUT["models"] = {}
for tag, feats in (("all_24", ALLV), ("drop6_18", DROP18)):
    X = df[feats].values.astype(float)
    OUT["models"][tag] = {}
    print(f"  [{tag}]")
    for name in CLAIM_AUC:
        aucs = []
        for s in SEEDS:
            m = build(s)[name]
            cv = StratifiedKFold(5, shuffle=True, random_state=s)
            p = cross_val_predict(m, X, y, cv=cv, method="predict_proba")[:, 1]
            aucs.append(roc_auc_score(y, p))
        mu, sd = float(np.mean(aucs)), float(np.std(aucs))
        OUT["models"][tag][name] = dict(auc=mu, sd=sd)
        d = mu - CLAIM_AUC[name]
        print(f"     {name:<14} {mu:.4f}±{sd:.4f}   claimed {CLAIM_AUC[name]:.3f}"
              f"   diff {d:+.4f}  {'OK' if abs(d) <= 0.005 else '<-- off'}")

# ── Q6 temporal hold-out across feature sets ────────────────────────────
print("\n" + "=" * 70)
print("Q6  temporal hold-out — manuscript claims M4=0.800, B0=0.831")
print("=" * 70)
tr, te = (df["year"] < 2000).values, (df["year"] >= 2000).values
print(f"  split: train n={tr.sum()} (rate {y[tr].mean():.3f}), "
      f"test n={te.sum()} (rate {y[te].mean():.3f})")
OUT["holdout"] = {}
for tag, feats in (("all_24", ALLV), ("drop6_18", DROP18)):
    X = df[feats].values.astype(float)
    row = {}
    for name in ("B0_LR", "M4_GBT_tuned"):
        m = build(42)[name].fit(X[tr], y[tr])
        row[name] = float(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    OUT["holdout"][tag] = row
    print(f"  [{tag}] B0={row['B0_LR']:.4f} (claim 0.831)   "
          f"M4={row['M4_GBT_tuned']:.4f} (claim 0.800)")

# rolling TSCV on both sets
print("\n  rolling time-series CV (claim 0.720 +- 0.075):")
OUT["rolling"] = {}
for tag, feats in (("all_24", ALLV), ("drop6_18", DROP18)):
    X = df[feats].values.astype(float)
    o = np.argsort(df["year"].values, kind="stable")
    Xo, yo = X[o], y[o]
    f = []
    for A, B in TimeSeriesSplit(n_splits=5, test_size=15).split(Xo):
        if len(np.unique(yo[B])) < 2:
            continue
        m = build(42)["M4_GBT_tuned"].fit(Xo[A], yo[A])
        f.append(float(roc_auc_score(yo[B], m.predict_proba(Xo[B])[:, 1])))
    OUT["rolling"][tag] = dict(folds=f, mean=float(np.mean(f)), sd=float(np.std(f)))
    print(f"    [{tag}] {np.mean(f):.4f} ± {np.std(f):.4f}   folds="
          f"{[round(x,3) for x in f]}")

# ── Q7 BN re-confirmation of the two Fig. 5 numbers ────────────────────
print("\n" + "=" * 70)
print("Q7  BN — sensitivity matched exactly in v1, so the CPTs are identical.")
print("     Re-deriving the two Fig. 5 numbers from that same network.")
print("=" * 70)
try:
    try:
        from pgmpy.models import DiscreteBayesianNetwork as BN
    except Exception:
        from pgmpy.models import BayesianNetwork as BN
    from pgmpy.estimators import BayesianEstimator
    from pgmpy.inference import VariableElimination
    bv = ["v01_agent_class", "v05_actor_type", "v13_conflict_zone",
          "v15_campaign_attacks", "v16_group_sophistication",
          "v19_state_sponsorship", "v22_mass_casualty_intent", "lethality"]
    d = df[bv].copy()
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
    bn = BN(E); bn.fit(d, estimator=BayesianEstimator, prior_type="BDeu",
                       equivalent_sample_size=5)
    inf = VariableElimination(bn)
    q = lambda ev: float(inf.query(["lethality"], evidence=ev).values[1])
    grid = {}
    for ac in (0, 1):
        for mc in (0, 1):
            grid[f"agent={ac},mc={mc}"] = q({"v01_agent_class": ac,
                                             "v22_mass_casualty_intent": mc})
    OUT["bn_heatmap"] = grid
    OUT["bn_scenarios"] = {
        "S0": q({}),
        "S1_mc0": q({"v22_mass_casualty_intent": 0}),
        "S2_agent0": q({"v01_agent_class": 0}),
        "S3_camp0": q({"v15_campaign_attacks": 0}),
        "S1+S2+S3_all0": q({"v22_mass_casualty_intent": 0, "v01_agent_class": 0,
                            "v15_campaign_attacks": 0}),
    }
    print("  heatmap cells P(lethal | agent, mc):")
    for k, v in grid.items():
        print(f"    {k:<18} {v:.4f}")
    print(f"  -> manuscript Fig. 5C states 0.800 for the top-right cell. "
          f"Computed: {grid['agent=1,mc=1']:.4f}")
    print("  scenarios:")
    for k, v in OUT["bn_scenarios"].items():
        print(f"    {k:<16} {v:.4f}")
    print(f"  -> manuscript Fig. 5B states 0.141 combined. "
          f"Computed: {OUT['bn_scenarios']['S1+S2+S3_all0']:.4f}")
except Exception as e:
    print("  BN failed:", e)

json.dump(OUT, open("verify_v2.json", "w"), indent=2, default=float)
print("\n-> verify_v2.json")
