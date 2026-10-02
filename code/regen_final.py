#!/usr/bin/env python3
"""
regen_final.py — regenerate every figure and the two remaining SI tables on
the corrected basis: 16-variable analytical matrix, 22-variable clustering,
and the Bayesian network with the outcome-derived node removed and regional
instability in its place.

Outputs
  fig2_cluster_radar.png        radar profile, 22-var clustering
  fig3_roc_importance.png       mean ROC over 5 seeds + out-of-fold importance
  fig4_bn_dag.png               new 8-node / 12-edge DAG
  fig5_bn_policy.png            tornado + scenarios (two panels, no heatmap)
  figS1_dendrogram.png          22-var dendrogram
  figS3_bootstrap_auc.png       out-of-fold vs in-sample, 16-var
  figS4_full_importance.png     all 16 variables, out-of-fold
  final_values.txt              every number plotted, for the manuscript

Usage: python3 regen_final.py --csv mat_coded_dataset.csv --out final
"""
import argparse, os, warnings, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              StackingClassifier)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, silhouette_score
from sklearn.inspection import permutation_importance
import gower

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="mat_coded_dataset.csv")
ap.add_argument("--out", default="final")
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
LOG = []
def say(s=""):
    print(s); LOG.append(s)

PAL = {"c1": "#C94F4A", "c2": "#E8943A", "c3": "#4AACB0", "c4": "#5B8DB8",
       "c5": "#8B7355", "c6": "#D4A853", "grey": "#9B9B9B", "dark": "#2C2C2C"}
FS_LABEL, FS_AXIS, FS_TICK, FS_BAR, FS_PANEL = 26, 16, 15, 10, 28
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.facecolor": "#F8F6F2",
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
def panel(ax, L):
    ax.text(-0.10, 1.18, L, transform=ax.transAxes, fontsize=FS_PANEL,
            fontweight="bold", ha="left", va="top")

LAYERS = [("M1", "Agent and materiel", ["v01_agent_class", "v02_delivery_complexity",
           "v03_acquisition_difficulty", "v04_weaponization_level"], "c1"),
          ("M2", "Perpetrator and execution", ["v05_actor_type", "v06_suicide_operation",
           "v07_team_coordination", "v08_claimed_responsibility"], "c2"),
          ("M3", "Operational conduct", ["v09_coordinated_attack", "v10_target_selectivity",
           "v11_operational_complexity", "v12_civilian_targeting",
           "v15_campaign_attacks"], "c6"),
          ("M4", "Environmental context", ["v13_conflict_zone",
           "v14_region_instability"], "c3"),
          ("M5", "Organisational capability", ["v16_group_sophistication",
           "v17_training_evidence", "v18_international_links"], "c5"),
          ("M6", "Institutional linkage", ["v19_state_sponsorship", "v20_cross_border",
           "v21_ideological_motivation"], "c4"),
          ("M7", "Intent and signalling", ["v22_mass_casualty_intent",
           "v23_propaganda_intent", "v24_wmd_signaling"], "grey")]
V2L = {v: c for c, _, vs, _ in LAYERS for v in vs}
L2C = {c: PAL[k] for c, _, _, k in LAYERS}

df = pd.read_csv(A.csv)
ALLV = [c for c in df.columns if c.startswith("v")]
COLLIN = ["v02_delivery_complexity", "v03_acquisition_difficulty",
          "v04_weaponization_level", "v07_team_coordination",
          "v17_training_evidence", "v23_propaganda_intent"]
LEAK = ["v22_mass_casualty_intent", "v24_wmd_signaling"]
FEAT = [c for c in ALLV if c not in COLLIN + LEAK]          # 16
CLU = [c for c in ALLV if c not in LEAK]                     # 22
X, Xc = df[FEAT].values.astype(float), df[CLU].values.astype(float)
y = df["lethality"].values.astype(int)
SEEDS = [42, 123, 456, 789, 2025]
M4P = dict(learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.6,
           max_features=0.6, min_samples_leaf=3)
say(f"N={len(df)}  analytical={len(FEAT)}  clustering={len(CLU)}")

# ── clustering (shared by fig2 and figS1) ──────────────────────────────
G = gower.gower_matrix(Xc)
Z = linkage(squareform(G, checks=False), method="ward")
lab = fcluster(Z, 2, criterion="maxclust")
df["cluster"] = lab
cut = float(np.sort(Z[:, 2])[-1])
sil = {k: float(silhouette_score(G, fcluster(Z, k, criterion="maxclust"),
                                 metric="precomputed")) for k in range(2, 7)}
say(f"silhouette {[round(sil[k],4) for k in range(2,7)]}  cut={cut:.4f}  "
    f"sizes={sorted(np.bincount(lab)[1:].tolist())}")
for c in sorted(set(lab)):
    say(f"  cluster {c}: n={(lab==c).sum()}  lethality={y[lab==c].mean():.4f}")

# ── fig2 radar ─────────────────────────────────────────────────────────
scores = {}
for code, name, vs, _ in LAYERS:
    av = [v for v in vs if v in CLU]
    if av:
        scores[code] = df.groupby("cluster")[av].mean().mean(axis=1)
codes = [c for c in scores]
ang = np.linspace(0, 2 * np.pi, len(codes), endpoint=False).tolist(); ang += ang[:1]
fig, ax = plt.subplots(figsize=(20, 10), subplot_kw=dict(polar=True))
for i, c in enumerate(sorted(set(lab))):
    vals = [scores[k][c] for k in codes]; vals += vals[:1]
    col = [PAL["c4"], PAL["c1"]][i]
    ax.plot(ang, vals, "o-", lw=2.8, color=col,
            label=f"Cluster {c} (n = {(lab==c).sum()}, lethality "
                  f"{y[lab==c].mean():.3f})")
    ax.fill(ang, vals, alpha=0.16, color=col)
ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
ax.set_xticks(ang[:-1])
ax.set_xticklabels([f"{c}\n{n}" for c, n, _, _ in LAYERS if c in codes],
                   size=FS_TICK - 2)
ax.set_ylim(0, 1); ax.tick_params(labelsize=FS_TICK - 3)
ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.12), fontsize=FS_TICK)
ax.set_title("MAT layer profile by cluster", fontsize=FS_LABEL, pad=28)
fig.savefig(f"{A.out}/fig2_cluster_radar.png", dpi=200, bbox_inches="tight")
plt.close(); say("-> fig2_cluster_radar.png")

# ── figS1 dendrogram ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 10))
dendrogram(Z, ax=ax, color_threshold=cut, no_labels=True,
           above_threshold_color=PAL["grey"])
ax.axhline(cut, ls="--", lw=2, color=PAL["c1"],
           label=f"$k=2$ cut (height = {cut:.2f})")
ax.set_xlabel(f"Incidents ($n$ = {len(df)}), leaves suppressed", fontsize=FS_AXIS)
ax.set_ylabel("Ward linkage distance", fontsize=FS_AXIS)
ax.set_title("Ward linkage on the 22-variable Gower distance matrix",
             fontsize=FS_LABEL - 2, pad=16)
ax.legend(fontsize=FS_TICK, loc="upper right"); ax.tick_params(labelsize=FS_TICK)
fig.savefig(f"{A.out}/figS1_dendrogram.png", dpi=200, bbox_inches="tight")
plt.close(); say("-> figS1_dendrogram.png")

# ── models ─────────────────────────────────────────────────────────────
def mk(s):
    return {"B0 Logistic Regression": LogisticRegression(class_weight="balanced",
                max_iter=500, random_state=s),
            "M1 Random Forest": RandomForestClassifier(n_estimators=200,
                class_weight="balanced", random_state=s, n_jobs=-1),
            "M2 GBT (default)": GradientBoostingClassifier(n_estimators=200,
                max_depth=4, learning_rate=0.05, subsample=0.8, random_state=s),
            "M3 GBT (full)": GradientBoostingClassifier(n_estimators=300,
                max_depth=5, learning_rate=0.03, subsample=0.8,
                min_samples_leaf=5, random_state=s),
            "M4 GBT (tuned)": GradientBoostingClassifier(random_state=s, **M4P),
            "M6 Stacking": StackingClassifier(
                [("lr", LogisticRegression(class_weight="balanced", max_iter=500)),
                 ("rf", RandomForestClassifier(n_estimators=200,
                     class_weight="balanced", random_state=s)),
                 ("gbt", GradientBoostingClassifier(random_state=s))],
                final_estimator=LogisticRegression(max_iter=500), cv=5)}

grid = np.linspace(0, 1, 201)
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
ax = axes[0]
for (nm, _), col in zip(mk(42).items(), [PAL["grey"], PAL["c3"], PAL["c2"],
                                         PAL["c5"], PAL["c1"], PAL["c4"]]):
    tp, au = [], []
    for s in SEEDS:
        cv = StratifiedKFold(5, shuffle=True, random_state=s)
        p = cross_val_predict(mk(s)[nm], X, y, cv=cv, method="predict_proba")[:, 1]
        f_, t_, _ = roc_curve(y, p)
        tp.append(np.interp(grid, f_, t_)); au.append(roc_auc_score(y, p))
    m = np.mean(tp, axis=0); m[0], m[-1] = 0.0, 1.0
    ax.plot(grid, m, lw=2.6, color=col,
            label=f"{nm} ({np.mean(au):.3f} $\\pm$ {np.std(au):.3f})")
    say(f"ROC {nm}: {np.mean(au):.4f}+-{np.std(au):.4f}")
ax.plot([0, 1], [0, 1], "--", color="grey", lw=1.5, alpha=0.6)
ax.set_xlabel("1 $-$ Specificity", fontsize=FS_AXIS)
ax.set_ylabel("Sensitivity", fontsize=FS_AXIS)
ax.set_title("Mean ROC over five seeds, 5-fold CV", fontsize=FS_LABEL, pad=16)
ax.legend(fontsize=FS_TICK - 2, loc="lower right", title="mean AUC $\\pm$ SD",
          title_fontsize=FS_TICK - 2)
ax.tick_params(labelsize=FS_TICK); panel(ax, "A")

imp = np.zeros(len(FEAT)); k = 0
for s in SEEDS:
    for tri, tei in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
        f_ = GradientBoostingClassifier(random_state=s, **M4P).fit(X[tri], y[tri])
        imp += permutation_importance(f_, X[tei], y[tei], n_repeats=30,
                                      random_state=s,
                                      scoring="roc_auc").importances_mean
        k += 1
imp = pd.Series(imp / k, index=FEAT).sort_values()
say("\nout-of-fold permutation importance (16 variables):")
for a_, b_ in imp[::-1].items():
    say(f"  {a_:<34}{V2L[a_]}  {b_:+.4f}")
top = imp[-10:]
ax = axes[1]
ax.barh(range(len(top)), top.values, color=[L2C[V2L[t]] for t in top.index],
        edgecolor="white", height=0.72)
ax.set_yticks(range(len(top)))
ax.set_yticklabels([f"{t[4:].replace('_',' ').title()}  ({V2L[t]})"
                    for t in top.index], fontsize=FS_BAR + 2)
ax.set_xlabel("Permutation importance, out-of-fold (mean AUC decrease)",
              fontsize=FS_AXIS)
ax.set_title("Feature importance by MAT layer", fontsize=FS_LABEL, pad=16)
ax.tick_params(labelsize=FS_TICK)
ax.legend(handles=[mpatches.Patch(color=L2C[c], label=c)
                   for c, _, _, _ in LAYERS if c in set(V2L[t] for t in FEAT)],
          fontsize=FS_TICK - 3, loc="lower right", ncol=2)
panel(ax, "B")
plt.subplots_adjust(wspace=0.38, top=0.82)
fig.savefig(f"{A.out}/fig3_roc_importance.png", dpi=200, bbox_inches="tight")
plt.close(); say("-> fig3_roc_importance.png")

# ── figS4 full importance ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 11))
ax.barh(range(len(imp)), imp.values, color=[L2C[V2L[t]] for t in imp.index],
        edgecolor="white", height=0.74)
ax.set_yticks(range(len(imp)))
ax.set_yticklabels([f"{t[4:].replace('_',' ').title()}  ({V2L[t]})"
                    for t in imp.index], fontsize=FS_BAR + 2)
ax.axvline(0, color=PAL["dark"], lw=1.2)
ax.set_xlabel("Permutation importance, out-of-fold (mean AUC decrease)",
              fontsize=FS_AXIS)
ax.set_title("Full importance profile, all 16 analytical variables",
             fontsize=FS_LABEL - 2, pad=16)
ax.tick_params(labelsize=FS_TICK)
ax.legend(handles=[mpatches.Patch(color=L2C[c], label=c)
                   for c, _, _, _ in LAYERS if c in set(V2L[t] for t in FEAT)],
          fontsize=FS_TICK - 2, loc="lower right", ncol=2)
fig.savefig(f"{A.out}/figS4_full_importance.png", dpi=200, bbox_inches="tight")
plt.close(); say("-> figS4_full_importance.png")

# ── figS3 bootstrap ────────────────────────────────────────────────────
oof = cross_val_predict(GradientBoostingClassifier(random_state=42, **M4P),
                        X, y, method="predict_proba",
                        cv=StratifiedKFold(5, shuffle=True, random_state=42))[:, 1]
ins = GradientBoostingClassifier(random_state=42, **M4P).fit(X, y).predict_proba(X)[:, 1]
rng = np.random.RandomState(42); bo, bi = [], []
for _ in range(1000):
    idx = rng.choice(len(y), len(y), replace=True)
    if len(np.unique(y[idx])) < 2:
        continue
    bo.append(roc_auc_score(y[idx], oof[idx])); bi.append(roc_auc_score(y[idx], ins[idx]))
fig, ax = plt.subplots(figsize=(20, 10))
for v, col, nm in ((bo, PAL["c4"], "out-of-fold predictions"),
                   (bi, PAL["c1"], "in-sample predictions")):
    ax.hist(v, bins=45, alpha=0.62, color=col, edgecolor="white",
            label=f"{nm}: mean {np.mean(v):.3f}, 95% PI "
                  f"[{np.percentile(v,2.5):.3f}, {np.percentile(v,97.5):.3f}]")
    ax.axvline(np.mean(v), ls="--", lw=2.2, color=col)
ax.set_xlabel("Bootstrap AUC ($n$ = 1,000 resamples)", fontsize=FS_AXIS)
ax.set_ylabel("Frequency", fontsize=FS_AXIS)
ax.set_title("Bootstrap AUC for the tuned GBT, 16-variable matrix",
             fontsize=FS_LABEL - 2, pad=16)
ax.legend(fontsize=FS_TICK - 1, loc="upper left"); ax.tick_params(labelsize=FS_TICK)
fig.savefig(f"{A.out}/figS3_bootstrap_auc.png", dpi=200, bbox_inches="tight")
plt.close()
say(f"-> figS3  OOF {np.mean(bo):.4f} [{np.percentile(bo,2.5):.4f},"
    f"{np.percentile(bo,97.5):.4f}]  in-sample {np.mean(bi):.4f} "
    f"[{np.percentile(bi,2.5):.4f},{np.percentile(bi,97.5):.4f}]")

# ── Bayesian network, new structure ────────────────────────────────────
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
     ("v01_agent_class", "lethality"), ("v05_actor_type", "lethality"),
     ("v15_campaign_attacks", "lethality")]
NICE = {"v14_region_instability": ("Regional\ninstability", "M4", "c3"),
        "v13_conflict_zone": ("Conflict\nzone", "M4", "c3"),
        "v19_state_sponsorship": ("State\nsponsorship", "M6", "c4"),
        "v16_group_sophistication": ("Group\nsophistication", "M5", "c5"),
        "v05_actor_type": ("Actor\ntype", "M2", "c2"),
        "v01_agent_class": ("Agent\nclass", "M1", "c1"),
        "v15_campaign_attacks": ("Campaign\npattern", "M3", "c6")}
POS = {"v14_region_instability": (0.09, 0.80), "v19_state_sponsorship": (0.09, 0.28),
       "v13_conflict_zone": (0.33, 0.80), "v16_group_sophistication": (0.33, 0.28),
       "v15_campaign_attacks": (0.58, 0.90), "v01_agent_class": (0.58, 0.56),
       "v05_actor_type": (0.58, 0.20), "lethality": (0.88, 0.56)}

def fit(ess):
    b = BN(E); b.fit(d, estimator=BayesianEstimator, prior_type="BDeu",
                     equivalent_sample_size=ess)
    return VariableElimination(b)
inf = fit(5)
q = lambda ev: float(inf.query(["lethality"], evidence=ev).values[1])
base = q({})
say(f"\nBN baseline P(lethal) = {base:.4f}")
sens = {v: round(q({v: 1}) - q({v: 0}), 4) for v in NICE}
for v, dd in sorted(sens.items(), key=lambda t: -t[1]):
    say(f"  {NICE[v][1]} {v:<30}P1={q({v:1}):.4f}  dP={dd:+.4f}")

fig, ax = plt.subplots(figsize=(20, 10)); ax.set_xlim(0, 1); ax.set_ylim(0.05, 1.0)
ax.axis("off"); ax.set_facecolor("white")
for s_, t_ in E:
    x1, y1 = POS[s_]; x2, y2 = POS[t_]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=PAL["dark"], lw=1.9,
                                alpha=0.55, shrinkA=30, shrinkB=32,
                                connectionstyle="arc3,rad=0.06"))
for n_, (x, y) in POS.items():
    out = n_ == "lethality"
    col = PAL["dark"] if out else PAL[NICE[n_][2]]
    ax.add_patch(plt.Circle((x, y), 0.075 if out else 0.062, fc=col, ec="white",
                            lw=2.6, zorder=3, alpha=0.94))
    txt = f"LETHALITY\n$P$ = {base:.3f}" if out else NICE[n_][0]
    ax.text(x, y, txt, ha="center", va="center", fontsize=FS_TICK - 4,
            color="white", fontweight="bold" if out else "normal", zorder=4)
    if not out:
        ax.text(x, y + 0.095, NICE[n_][1], ha="center", fontsize=FS_BAR + 3,
                color=col, fontweight="bold")
ax.set_title("MAT Bayesian network: 8 nodes, 12 edges, BDeu "
             "($\\alpha = 5$)", fontsize=FS_LABEL - 2, pad=14)
fig.savefig(f"{A.out}/fig4_bn_dag.png", dpi=200, bbox_inches="tight")
plt.close(); say("-> fig4_bn_dag.png")

# ── fig5 two panels ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
order = sorted(sens.items(), key=lambda t: t[1])
ax = axes[0]
ax.barh(range(len(order)), [v for _, v in order],
        color=[PAL[NICE[k][2]] for k, _ in order], edgecolor="white", height=0.66)
for i, (k_, v_) in enumerate(order):
    ax.text(v_ + 0.008, i, f"{v_:+.3f}", va="center", fontsize=FS_BAR + 1,
            color=PAL["dark"])
ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"{NICE[k][0].replace(chr(10),' ')} ({NICE[k][1]})"
                    for k, _ in order], fontsize=FS_BAR + 2)
ax.set_xlabel(r"$\Delta P(\mathrm{lethal})$, node high $-$ node low",
              fontsize=FS_AXIS)
ax.set_title("Node sensitivity", fontsize=FS_LABEL, pad=16)
ax.set_xlim(0, max(sens.values()) * 1.22); ax.tick_params(labelsize=FS_TICK)
panel(ax, "A")

SC = [("S0 baseline", {}),
      ("stable region", {"v14_region_instability": 0}),
      ("no advanced agent", {"v01_agent_class": 0}),
      ("no campaign", {"v15_campaign_attacks": 0}),
      ("all three", {"v14_region_instability": 0, "v01_agent_class": 0,
                     "v15_campaign_attacks": 0})]
vals = [q(e) for _, e in SC]
say("scenarios: " + ", ".join(f"{n}={v:.4f}" for (n, _), v in zip(SC, vals)))
ax = axes[1]
bars = ax.bar(range(len(vals)), vals,
              color=[PAL["grey"], PAL["c3"], PAL["c1"], PAL["c6"], PAL["c4"]],
              edgecolor="white")
for b_, v_ in zip(bars, vals):
    ax.text(b_.get_x() + b_.get_width() / 2, v_ + 0.005, f"{v_:.3f}",
            ha="center", fontsize=FS_BAR + 1, color=PAL["dark"])
ax.axhline(base, ls="--", lw=1.4, color=PAL["dark"], alpha=0.7)
ax.set_xticks(range(len(SC)))
ax.set_xticklabels([n for n, _ in SC], rotation=24, ha="right", fontsize=FS_BAR + 2)
ax.set_ylabel(r"$P(\mathrm{lethal}\mid\mathrm{evidence})$", fontsize=FS_AXIS)
ax.set_title("Conditional scenarios", fontsize=FS_LABEL, pad=16)
ax.set_ylim(0, max(vals) * 1.25); ax.tick_params(labelsize=FS_TICK)
panel(ax, "B")
plt.subplots_adjust(wspace=0.38, top=0.82)
fig.savefig(f"{A.out}/fig5_bn_policy.png", dpi=200, bbox_inches="tight")
plt.close(); say("-> fig5_bn_policy.png")

# ── TABLE S4 : BDeu sensitivity on the new network ─────────────────────
say("\n=== TABLE S4 (new network) ===")
NODES = ["v01_agent_class", "v14_region_instability", "v15_campaign_attacks",
         "v13_conflict_zone", "v05_actor_type", "v19_state_sponsorship"]
say("ESS   baseline " + "".join(f"{NICE[n][1]:>12}" for n in NODES))
for ess in (1, 5, 10, 20, 50):
    i2 = fit(ess); qq = lambda ev: float(i2.query(["lethality"], evidence=ev).values[1])
    say(f"{ess:<6}{qq({}):.4f}  " +
        "".join(f"{qq({n:1})-qq({n:0}):>+12.4f}" for n in NODES))

# ── TABLE S3 : class imbalance on 16 vars ──────────────────────────────
say("\n=== TABLE S3 (16 variables, seed 42) ===")
cv1 = StratifiedKFold(5, shuffle=True, random_state=42)
for nm, b_, bal in (("B0", LogisticRegression(max_iter=500, random_state=42),
                     LogisticRegression(max_iter=500, class_weight="balanced",
                                        random_state=42)),
                    ("M1", RandomForestClassifier(n_estimators=200, random_state=42),
                     RandomForestClassifier(n_estimators=200,
                                            class_weight="balanced", random_state=42))):
    a_ = roc_auc_score(y, cross_val_predict(b_, X, y, cv=cv1,
                                            method="predict_proba")[:, 1])
    c_ = roc_auc_score(y, cross_val_predict(bal, X, y, cv=cv1,
                                            method="predict_proba")[:, 1])
    say(f"  {nm}: base={a_:.4f}  balanced={c_:.4f}  delta={c_-a_:+.4f}")
w = np.where(y == 1, (len(y) - y.sum()) / y.sum(), 1.0)
a_ = roc_auc_score(y, cross_val_predict(
    GradientBoostingClassifier(random_state=42, **M4P), X, y, cv=cv1,
    method="predict_proba")[:, 1])
oo = np.zeros(len(y))
for tri, tei in cv1.split(X, y):
    m_ = GradientBoostingClassifier(random_state=42, **M4P)
    m_.fit(X[tri], y[tri], sample_weight=w[tri]); oo[tei] = m_.predict_proba(X[tei])[:, 1]
c_ = roc_auc_score(y, oo)
say(f"  M4: base={a_:.4f}  balanced={c_:.4f}  delta={c_-a_:+.4f}")

open(f"{A.out}/final_values.txt", "w").write("\n".join(LOG))
print(f"\n-> {A.out}/  7 PNG + final_values.txt")
