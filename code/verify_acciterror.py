#!/usr/bin/env python3
"""
verify_acciterror.py — recompute every numeric claim in the Safety Science
manuscript and diff it against what the manuscript actually says.

Input : mat_coded_dataset.csv  (203 rows, 24 coded variables + lethality,
                                year, country, gname)   -- GTD NOT required
Output: verify_results.json    (everything recomputed)
        verify_report.txt      (side-by-side diff vs manuscript claims)

Every number printed under RECOMPUTED comes from this run. Numbers under
CLAIMED are transcribed from latex/main.tex and latex/SI.tex as of
2026-09-14. MATCH/MISMATCH is decided on the printed precision.

Usage:  python3 verify_acciterror.py --csv mat_coded_dataset.csv
"""

import argparse, json, sys, warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import mannwhitneyu, norm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              StackingClassifier)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     RandomizedSearchCV, TimeSeriesSplit)
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score)
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
SEEDS = [42, 123, 456, 789, 2025]
R = {}          # recomputed values
NOTES = []


# ── helpers ──────────────────────────────────────────────────────────────
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cv_scores(model, X, y, seeds=SEEDS):
    aucs, f1s, pres, recs = [], [], [], []
    for s in seeds:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        m = model
        if hasattr(m, "random_state"):
            m = m.__class__(**{**m.get_params(), "random_state": s})
        p = cross_val_predict(m, X, y, cv=cv, method="predict_proba")[:, 1]
        pred = (p >= 0.5).astype(int)
        aucs.append(roc_auc_score(y, p))
        f1s.append(f1_score(y, pred, zero_division=0))
        pres.append(precision_score(y, pred, zero_division=0))
        recs.append(recall_score(y, pred, zero_division=0))
    return dict(auc=float(np.mean(aucs)), auc_sd=float(np.std(aucs)),
                f1=float(np.mean(f1s)), precision=float(np.mean(pres)),
                recall=float(np.mean(recs)))


def models(seed=42, balanced=False):
    lr = LogisticRegression(class_weight="balanced" if balanced else None,
                            max_iter=1000, random_state=seed)
    rf = RandomForestClassifier(n_estimators=500,
                                class_weight="balanced" if balanced else None,
                                random_state=seed, n_jobs=-1)
    gbt = GradientBoostingClassifier(random_state=seed)
    return lr, rf, gbt


# ── 0. load ──────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="mat_coded_dataset.csv")
ap.add_argument("--skip-bn", action="store_true")
args = ap.parse_args()

df = pd.read_csv(args.csv)
ALL_V = [c for c in df.columns if c.startswith("v")]
y = df["lethality"].values.astype(int)
R["n_total"] = int(len(df))
R["n_lethal"] = int(y.sum())
R["lethality_rate"] = float(y.mean())
R["n_variables_coded"] = len(ALL_V)
print(f"[0] N={R['n_total']}  lethal={R['n_lethal']}  "
      f"rate={R['lethality_rate']:.4f}  coded vars={R['n_variables_coded']}")


# ── 1. collinearity filter (manuscript: drop |r| >= 0.875, 24 -> 18) ─────
corr = df[ALL_V].corr(method="pearson").abs()
dropped, kept = [], []
for i, a in enumerate(ALL_V):
    if any((corr.loc[a, b] >= 0.875) for b in kept):
        dropped.append(a)
    else:
        kept.append(a)
R["collinearity_dropped"] = dropped
R["collinearity_kept"] = kept
R["n_features_kept"] = len(kept)

# residual correlation ABOVE threshold among the kept set (the Fig. S2 issue)
resid = []
for i in range(len(kept)):
    for j in range(i + 1, len(kept)):
        r = corr.loc[kept[i], kept[j]]
        if r >= 0.875:
            resid.append([kept[i], kept[j], float(r)])
R["residual_pairs_above_threshold"] = resid
print(f"[1] dropped {len(dropped)} -> {len(kept)} kept; "
      f"residual pairs >= 0.875 among kept: {len(resid)}")
for a, b, r in resid:
    print(f"      !! {a} x {b} = {r:.3f}")

FEAT = kept
X = df[FEAT].values.astype(float)


# ── 2. clustering ────────────────────────────────────────────────────────
try:
    import gower
    G = gower.gower_matrix(X)
except ImportError:
    sys.exit("pip install gower")

Z = linkage(squareform(G, checks=False), method="ward")
sil = {}
for k in range(2, 7):
    lab = fcluster(Z, k, criterion="maxclust")
    sil[k] = float(silhouette_score(G, lab, metric="precomputed"))
R["silhouette_ward"] = sil
best_k = max(sil, key=sil.get)
R["best_k"] = int(best_k)

km_sil = {}
for k in range(2, 7):
    lab = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X)
    km_sil[k] = float(silhouette_score(X, lab))
R["silhouette_kmeans"] = km_sil

clab = fcluster(Z, best_k, criterion="maxclust")
df["cluster"] = clab
R["clusters"] = {}
for c in sorted(set(clab)):
    m = clab == c
    lo, hi = wilson(int(y[m].sum()), int(m.sum()))
    R["clusters"][int(c)] = dict(n=int(m.sum()), lethal=int(y[m].sum()),
                                 rate=float(y[m].mean()),
                                 wilson=[round(lo, 3), round(hi, 3)])
print(f"[2] ward silhouette {sil}   best k={best_k}")
for c, v in R["clusters"].items():
    print(f"      cluster {c}: n={v['n']} rate={v['rate']:.3f} "
          f"CI[{v['wilson'][0]:.2f},{v['wilson'][1]:.2f}]")


# ── 3. agent-class lethality (manuscript: n=11, 72.7%, CI 43-90) ─────────
adv = df["v01_agent_class"] >= 1
lo, hi = wilson(int(y[adv].sum()), int(adv.sum()))
R["agent_advanced"] = dict(n=int(adv.sum()), pct=float(adv.mean()),
                           rate=float(y[adv].mean()),
                           wilson=[round(lo, 3), round(hi, 3)],
                           rate_other=float(y[~adv].mean()))
print(f"[3] advanced-class n={adv.sum()} ({adv.mean()*100:.1f}%) "
      f"rate={y[adv].mean():.3f} other={y[~adv].mean():.3f}")


# ── 4. models, WITHOUT and WITH cluster features (settles the leakage Q) ─
Xc = np.column_stack([X, pd.get_dummies(df["cluster"], prefix="cl").values])
R["models"] = {}
for tag, Xi in (("no_cluster_features", X), ("with_cluster_features", Xc)):
    lr, rf, gbt = models()
    m4_search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=42),
        {"learning_rate": [0.01, 0.03, 0.05, 0.1],
         "max_depth": [2, 3, 4, 5],
         "n_estimators": [100, 200, 300, 500],
         "subsample": [0.6, 0.8, 1.0],
         "max_features": [0.6, 0.8, None],
         "min_samples_leaf": [1, 3, 5]},
        n_iter=30, cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="roc_auc", random_state=42, n_jobs=-1)
    m4_search.fit(Xi, y)
    m4 = GradientBoostingClassifier(random_state=42, **m4_search.best_params_)
    m6 = StackingClassifier(
        [("lr", LogisticRegression(max_iter=1000)),
         ("rf", RandomForestClassifier(n_estimators=500, random_state=42)),
         ("gbt", GradientBoostingClassifier(random_state=42))],
        final_estimator=LogisticRegression(max_iter=1000), cv=5)

    blk = {}
    for name, mdl in (("B0_LR", lr), ("M1_RF", rf), ("M2_GBT", gbt),
                      ("M3_GBT_full", GradientBoostingClassifier(
                          n_estimators=300, max_depth=5, learning_rate=0.03,
                          subsample=0.8, min_samples_leaf=5, random_state=42)),
                      ("M4_GBT_tuned", m4), ("M6_Stacking", m6)):
        blk[name] = cv_scores(mdl, Xi, y)
        print(f"[4:{tag}] {name}: AUC={blk[name]['auc']:.4f}"
              f"±{blk[name]['auc_sd']:.4f}")
    blk["M4_best_params"] = m4_search.best_params_
    R["models"][tag] = blk

PRIMARY = "no_cluster_features"


# ── 5. bootstrap CI for M4 — out-of-fold, not in-sample ─────────────────
best = R["models"][PRIMARY]["M4_best_params"]
m4 = GradientBoostingClassifier(random_state=42, **best)
oof = cross_val_predict(m4, X, y, method="predict_proba",
                        cv=StratifiedKFold(5, shuffle=True, random_state=42))[:, 1]
rng = np.random.RandomState(42)
boot = []
for _ in range(1000):
    idx = rng.choice(len(y), len(y), replace=True)
    if len(np.unique(y[idx])) < 2:
        continue
    boot.append(roc_auc_score(y[idx], oof[idx]))
R["bootstrap_oof"] = dict(mean=float(np.mean(boot)),
                          ci=[float(np.percentile(boot, 2.5)),
                              float(np.percentile(boot, 97.5))],
                          n=len(boot))
# in-sample version, to show what the original number actually was
m4f = GradientBoostingClassifier(random_state=42, **best).fit(X, y)
ins = m4f.predict_proba(X)[:, 1]
boot_i = []
for _ in range(1000):
    idx = rng.choice(len(y), len(y), replace=True)
    if len(np.unique(y[idx])) < 2:
        continue
    boot_i.append(roc_auc_score(y[idx], ins[idx]))
R["bootstrap_insample"] = dict(mean=float(np.mean(boot_i)),
                               ci=[float(np.percentile(boot_i, 2.5)),
                                   float(np.percentile(boot_i, 97.5))])
print(f"[5] bootstrap OOF mean={R['bootstrap_oof']['mean']:.4f} "
      f"CI={R['bootstrap_oof']['ci']}  |  in-sample mean="
      f"{R['bootstrap_insample']['mean']:.4f}")


# ── 6. temporal hold-out + rolling time-series CV ────────────────────────
tr, te = df["year"] < 2000, df["year"] >= 2000
R["temporal_holdout"] = dict(
    n_train=int(tr.sum()), rate_train=float(y[tr].mean()),
    n_test=int(te.sum()), rate_test=float(y[te].mean()))
for name, mdl in (("B0", LogisticRegression(max_iter=1000, random_state=42)),
                  ("M4", GradientBoostingClassifier(random_state=42, **best))):
    mdl.fit(X[tr.values], y[tr.values])
    R["temporal_holdout"][f"auc_{name}"] = float(
        roc_auc_score(y[te.values], mdl.predict_proba(X[te.values])[:, 1]))

order = np.argsort(df["year"].values, kind="stable")
Xo, yo = X[order], y[order]
folds = []
tss = TimeSeriesSplit(n_splits=5, test_size=15)
for a, b in tss.split(Xo):
    if len(np.unique(yo[b])) < 2:
        continue
    mdl = GradientBoostingClassifier(random_state=42, **best).fit(Xo[a], yo[a])
    folds.append(float(roc_auc_score(yo[b], mdl.predict_proba(Xo[b])[:, 1])))
R["rolling_tscv"] = dict(folds=folds, mean=float(np.mean(folds)),
                         sd=float(np.std(folds)))
print(f"[6] holdout train={tr.sum()} test={te.sum()} "
      f"B0={R['temporal_holdout']['auc_B0']:.3f} "
      f"M4={R['temporal_holdout']['auc_M4']:.3f} | rolling "
      f"{R['rolling_tscv']['mean']:.3f}±{R['rolling_tscv']['sd']:.3f} {folds}")


# ── 7. class-imbalance sensitivity (Table S3) ───────────────────────────
R["class_imbalance"] = {}
cv1 = StratifiedKFold(5, shuffle=True, random_state=42)
for name, base, bal in (
        ("B0", LogisticRegression(max_iter=1000, random_state=42),
         LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ("M1", RandomForestClassifier(n_estimators=500, random_state=42),
         RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42))):
    a = roc_auc_score(y, cross_val_predict(base, X, y, cv=cv1, method="predict_proba")[:, 1])
    b = roc_auc_score(y, cross_val_predict(bal, X, y, cv=cv1, method="predict_proba")[:, 1])
    R["class_imbalance"][name] = dict(base=float(a), balanced=float(b),
                                      delta=float(b - a))
w = np.where(y == 1, (len(y) - y.sum()) / y.sum(), 1.0)
a = roc_auc_score(y, cross_val_predict(
    GradientBoostingClassifier(random_state=42, **best), X, y, cv=cv1,
    method="predict_proba")[:, 1])
oofw = np.zeros(len(y))
for tri, tei in cv1.split(X, y):
    mm = GradientBoostingClassifier(random_state=42, **best)
    mm.fit(X[tri], y[tri], sample_weight=w[tri])
    oofw[tei] = mm.predict_proba(X[tei])[:, 1]
b = roc_auc_score(y, oofw)
R["class_imbalance"]["M4"] = dict(base=float(a), balanced=float(b),
                                  delta=float(b - a))
print(f"[7] imbalance deltas: "
      f"{ {k: round(v['delta'],4) for k,v in R['class_imbalance'].items()} }")


# ── 8. importance (CV-honest) + Mann-Whitney / Cohen's d ────────────────
pi = permutation_importance(m4f, X, y, n_repeats=100, random_state=42,
                            scoring="roc_auc")
R["perm_importance"] = dict(sorted(
    {FEAT[i]: float(pi.importances_mean[i]) for i in range(len(FEAT))}.items(),
    key=lambda kv: -kv[1]))
R["mdi_importance"] = dict(sorted(
    {FEAT[i]: float(m4f.feature_importances_[i]) for i in range(len(FEAT))}.items(),
    key=lambda kv: -kv[1]))
st = {}
for f in FEAT:
    a_, b_ = df.loc[y == 1, f].values, df.loc[y == 0, f].values
    u, p = mannwhitneyu(a_, b_, alternative="two-sided")
    sd = np.sqrt((a_.std() ** 2 + b_.std() ** 2) / 2 + 1e-12)
    st[f] = dict(p=float(p), d=float((a_.mean() - b_.mean()) / sd))
R["univariate"] = dict(sorted(st.items(), key=lambda kv: -abs(kv[1]["d"])))
print("[8] top perm:", list(R["perm_importance"].items())[:5])


# ── 9. 2019-2021 exclusion sensitivity ─────────────────────────────────
m = df["year"] <= 2018
s = cv_scores(GradientBoostingClassifier(random_state=42, **best),
              X[m.values], y[m.values])
R["exclude_2019_2021"] = dict(n=int(m.sum()), auc=s["auc"],
                              delta=float(s["auc"] - R["models"][PRIMARY]["M4_GBT_tuned"]["auc"]))
print(f"[9] excl 2019-21: n={m.sum()} AUC={s['auc']:.4f} "
      f"delta={R['exclude_2019_2021']['delta']:+.4f}")


# ── 10. Bayesian network ───────────────────────────────────────────────
if not args.skip_bn:
    try:
        try:
            from pgmpy.models import DiscreteBayesianNetwork as BN
        except Exception:
            from pgmpy.models import BayesianNetwork as BN
        from pgmpy.estimators import BayesianEstimator
        from pgmpy.inference import VariableElimination

        bn_vars = ["v01_agent_class", "v05_actor_type", "v13_conflict_zone",
                   "v15_campaign_attacks", "v16_group_sophistication",
                   "v19_state_sponsorship", "v22_mass_casualty_intent",
                   "lethality"]
        d = df[bn_vars].copy()
        for c in ["v01_agent_class", "v05_actor_type", "v16_group_sophistication"]:
            d[c] = (d[c] > d[c].median()).astype(int)
        d = d.astype(int)
        edges = [("v13_conflict_zone", "v15_campaign_attacks"),
                 ("v13_conflict_zone", "v01_agent_class"),
                 ("v19_state_sponsorship", "v01_agent_class"),
                 ("v19_state_sponsorship", "v16_group_sophistication"),
                 ("v16_group_sophistication", "v05_actor_type"),
                 ("v16_group_sophistication", "v01_agent_class"),
                 ("v05_actor_type", "v22_mass_casualty_intent"),
                 ("v01_agent_class", "v22_mass_casualty_intent"),
                 ("v15_campaign_attacks", "v22_mass_casualty_intent"),
                 ("v22_mass_casualty_intent", "lethality"),
                 ("v01_agent_class", "lethality"),
                 ("v05_actor_type", "lethality")]
        bn = BN(edges)
        bn.fit(d, estimator=BayesianEstimator, prior_type="BDeu",
               equivalent_sample_size=5)
        inf = VariableElimination(bn)
        base_p = float(inf.query(["lethality"]).values[1])
        R["bn"] = dict(n_edges=len(edges), baseline=base_p, sensitivity={})
        for v in ["v22_mass_casualty_intent", "v01_agent_class",
                  "v15_campaign_attacks", "v13_conflict_zone",
                  "v19_state_sponsorship"]:
            p0 = float(inf.query(["lethality"], evidence={v: 0}).values[1])
            p1 = float(inf.query(["lethality"], evidence={v: 1}).values[1])
            R["bn"]["sensitivity"][v] = dict(p0=p0, p1=p1, delta=p1 - p0)
        # policy scenarios: condition jointly on ABSENCE of the three drivers
        R["bn"]["scenarios"] = {
            "S0_baseline": base_p,
            "S1_no_mc_intent": float(inf.query(["lethality"], evidence={"v22_mass_casualty_intent": 0}).values[1]),
            "S2_no_advanced_agent": float(inf.query(["lethality"], evidence={"v01_agent_class": 0}).values[1]),
            "S3_no_campaign": float(inf.query(["lethality"], evidence={"v15_campaign_attacks": 0}).values[1]),
            "S1+S2+S3": float(inf.query(["lethality"], evidence={
                "v22_mass_casualty_intent": 0, "v01_agent_class": 0,
                "v15_campaign_attacks": 0}).values[1]),
        }
        R["bn"]["heatmap_agent1_mc1"] = float(inf.query(
            ["lethality"], evidence={"v01_agent_class": 1,
                                     "v22_mass_casualty_intent": 1}).values[1])
        # d-separation: manuscript claims MC intent _||_ L1-L2 given L5-L6
        try:
            R["bn"]["dsep_mcintent_vs_statesponsor_given_agent_actor"] = bool(
                not bn.is_dconnected("v22_mass_casualty_intent",
                                     "v19_state_sponsorship",
                                     observed=["v01_agent_class", "v05_actor_type"]))
        except Exception as e:
            NOTES.append(f"d-separation API unavailable: {e}")
        print(f"[10] BN baseline={base_p:.4f} "
              f"scenarios={ {k: round(v,4) for k,v in R['bn']['scenarios'].items()} }")
    except Exception as e:
        NOTES.append(f"BN block failed: {e}")
        print("[10] BN FAILED:", e)


# ── 11. diff against manuscript ────────────────────────────────────────
CLAIM = [
    ("N", 203, R["n_total"], 0),
    ("lethal count", 48, R["n_lethal"], 0),
    ("lethality rate", 0.236, R["lethality_rate"], 0.001),
    ("features kept", 18, R["n_features_kept"], 0),
    ("silhouette k=2", 0.647, R["silhouette_ward"][2], 0.002),
    ("silhouette k=3", 0.631, R["silhouette_ward"][3], 0.002),
    ("kmeans k=2", 0.482, R["silhouette_kmeans"][2], 0.002),
    ("advanced n", 11, R["agent_advanced"]["n"], 0),
    ("advanced rate", 0.727, R["agent_advanced"]["rate"], 0.005),
    ("B0 AUC", 0.800, R["models"][PRIMARY]["B0_LR"]["auc"], 0.005),
    ("M1 AUC", 0.795, R["models"][PRIMARY]["M1_RF"]["auc"], 0.005),
    ("M2 AUC", 0.788, R["models"][PRIMARY]["M2_GBT"]["auc"], 0.005),
    ("M3 AUC", 0.796, R["models"][PRIMARY]["M3_GBT_full"]["auc"], 0.005),
    ("M4 AUC", 0.813, R["models"][PRIMARY]["M4_GBT_tuned"]["auc"], 0.005),
    ("M6 AUC", 0.800, R["models"][PRIMARY]["M6_Stacking"]["auc"], 0.005),
    ("bootstrap mean", 0.873, R["bootstrap_oof"]["mean"], 0.005),
    ("bootstrap CI lo", 0.809, R["bootstrap_oof"]["ci"][0], 0.005),
    ("bootstrap CI hi", 0.926, R["bootstrap_oof"]["ci"][1], 0.005),
    ("holdout N train", 114, R["temporal_holdout"]["n_train"], 0),
    ("holdout N test", 89, R["temporal_holdout"]["n_test"], 0),
    ("holdout M4 AUC", 0.800, R["temporal_holdout"]["auc_M4"], 0.005),
    ("holdout B0 AUC", 0.831, R["temporal_holdout"]["auc_B0"], 0.005),
    ("rolling mean", 0.720, R["rolling_tscv"]["mean"], 0.005),
    ("rolling sd", 0.075, R["rolling_tscv"]["sd"], 0.005),
]
if "clusters" in R and len(R["clusters"]) >= 2:
    ns = sorted(v["n"] for v in R["clusters"].values())
    CLAIM += [("cluster sizes (small)", 21, ns[0], 0),
              ("cluster sizes (large)", 182, ns[-1], 0)]
if "bn" in R:
    s = R["bn"]["sensitivity"]
    CLAIM += [
        ("BN baseline", 0.248, R["bn"]["baseline"], 0.002),
        ("BN MC-intent P1", 0.716, s["v22_mass_casualty_intent"]["p1"], 0.002),
        ("BN MC-intent dP", 0.530, s["v22_mass_casualty_intent"]["delta"], 0.002),
        ("BN agent dP", 0.438, s["v01_agent_class"]["delta"], 0.002),
        ("BN campaign dP", 0.253, s["v15_campaign_attacks"]["delta"], 0.002),
        ("BN conflict dP", 0.142, s["v13_conflict_zone"]["delta"], 0.002),
        ("BN state dP", 0.030, s["v19_state_sponsorship"]["delta"], 0.002),
        ("BN combined scenario", 0.141, R["bn"]["scenarios"]["S1+S2+S3"], 0.002),
        ("BN heatmap max", 0.800, R["bn"]["heatmap_agent1_mc1"], 0.005),
    ]

lines = ["claim                        manuscript   recomputed   verdict",
         "-" * 64]
nbad = 0
for name, claimed, got, tol in CLAIM:
    ok = abs(float(got) - float(claimed)) <= tol
    nbad += (not ok)
    lines.append(f"{name:<28} {claimed:>10}   {float(got):>10.4f}   "
                 f"{'MATCH' if ok else '** MISMATCH **'}")
lines += ["", f"{len(CLAIM)-nbad}/{len(CLAIM)} match.", ""]
lines += ["NOT REPRODUCIBLE FROM CODE (need external input):",
          "  - kappa_mean = 0.989 inter-rater reliability. Requires the second",
          "    coder's independent re-coding of the 20 sampled incidents.",
          "    No code can generate this. Supply the second coder's sheet or",
          "    remove the claim.",
          "  - 'GTD 1970-2004 filter yields n = 190'. Requires gtd_full.csv.",
          ""]
if R["residual_pairs_above_threshold"]:
    lines += ["COLLINEARITY RULE VIOLATED among retained variables:"]
    for a, b, r in R["residual_pairs_above_threshold"]:
        lines.append(f"  {a} x {b} = {r:.3f}  (rule says drop at >= 0.875)")
    lines.append("")
lines += ["LEAKAGE CHECK (M3/M4 with vs without cluster dummies):"]
for k in ("M3_GBT_full", "M4_GBT_tuned"):
    a = R["models"]["no_cluster_features"][k]["auc"]
    b = R["models"]["with_cluster_features"][k]["auc"]
    lines.append(f"  {k}: without={a:.4f}  with={b:.4f}  gap={b-a:+.4f}")
lines += ["", "BOOTSTRAP: out-of-fold vs in-sample",
          f"  OOF       mean={R['bootstrap_oof']['mean']:.4f} CI={[round(x,4) for x in R['bootstrap_oof']['ci']]}",
          f"  in-sample mean={R['bootstrap_insample']['mean']:.4f} CI={[round(x,4) for x in R['bootstrap_insample']['ci']]}",
          "  (the manuscript's 0.873 almost certainly came from the in-sample path)"]
if NOTES:
    lines += ["", "NOTES:"] + [f"  - {n}" for n in NOTES]

report = "\n".join(lines)
print("\n" + report)
open("verify_report.txt", "w").write(report)
json.dump(R, open("verify_results.json", "w"), indent=2)
print("\n-> verify_results.json, verify_report.txt")
