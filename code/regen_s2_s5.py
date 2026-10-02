#!/usr/bin/env python3
"""
Rebuild figS2 (correlation matrix) and figS5 (temporal validation) on the
16-variable analytical basis, with white panel backgrounds to match the rest
of the figure set.

figS2 shipped in the April draft was built on 18 variables and still contained
v22_mass_casualty_intent and v24_wmd_signaling, the two variables removed for
contamination by the outcome. Its caption quoted correlations involving those
variables. This script recomputes the matrix from the 16 variables actually
used and prints every pair with |r| >= 0.30 so the caption can be written from
the log rather than from the old figure.

Usage: python3 regen_s2_s5.py --csv mat_coded_public.csv --out figs_s
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import (StratifiedKFold, TimeSeriesSplit,
                                     cross_val_predict)

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="mat_coded_public.csv")
ap.add_argument("--out", default="figs_s")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)

LOG = []
def say(s):
    print(s); LOG.append(s)

PAL = {"c1": "#C94F4A", "c4": "#5B8DB8", "grey": "#9B9B9B", "dark": "#2C2C2C"}
FS_LABEL, FS_AXIS, FS_TICK, FS_BAR = 26, 16, 15, 10
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.facecolor": "white",
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
def panel(ax, L):
    ax.text(-0.10, 1.18, L, transform=ax.transAxes, fontsize=28,
            fontweight="bold", ha="left", va="top")

# Derived exactly as in regen_final_v2.py -- never retyped, so the two
# scripts cannot drift apart.
COLLIN = ["v02_delivery_complexity", "v03_acquisition_difficulty",
          "v04_weaponization_level", "v07_team_coordination",
          "v17_training_evidence", "v23_propaganda_intent"]
LEAK = ["v22_mass_casualty_intent", "v24_wmd_signaling"]
NICE = {"v01_agent_class": "Agent Class",
        "v05_actor_type": "Actor Type",
        "v06_suicide_operation": "Suicide Operation",
        "v08_claimed_responsibility": "Claimed Resp.",
        "v09_coordinated_attack": "Coordinated Attack",
        "v10_target_selectivity": "Target Selectivity",
        "v11_operational_complexity": "Op. Complexity",
        "v12_civilian_targeting": "Civilian Targeting",
        "v13_conflict_zone": "Conflict Zone",
        "v14_region_instability": "Region Instability",
        "v15_campaign_attacks": "Campaign Attacks",
        "v16_group_sophistication": "Group Sophistication",
        "v18_international_links": "Intl. Links",
        "v19_state_sponsorship": "State Sponsorship",
        "v20_cross_border": "Cross-Border",
        "v21_ideological_motivation": "Ideological Motivation"}

df = pd.read_csv(A.csv)
ALLV = [c for c in df.columns if c.startswith("v")]
FEAT = [c for c in ALLV if c not in COLLIN + LEAK]
assert len(FEAT) == 16, f"expected 16 analytical variables, got {len(FEAT)}"
assert set(FEAT) <= set(NICE), f"unlabelled: {set(FEAT) - set(NICE)}"
say(f"N={len(df)}  variables in matrix={len(FEAT)}")
X = df[FEAT].values.astype(float)
y = df["lethality"].values.astype(int)
SEEDS = [42, 123, 456, 789, 2025]
M4P = dict(learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.6,
           max_features=0.6, min_samples_leaf=3)

# =========================================================================
# FIGURE S2 — correlation matrix on the analytical variables
# =========================================================================
say("\n--- Figure S2 ---")
C = df[FEAT].corr(method="pearson")
labs = [NICE[c] for c in FEAT]

pairs = []
for i in range(len(FEAT)):
    for j in range(i + 1, len(FEAT)):
        pairs.append((abs(C.iloc[i, j]), C.iloc[i, j], NICE[FEAT[i]], NICE[FEAT[j]]))
pairs.sort(reverse=True)
say("  pairs with |r| >= 0.30, strongest first:")
for a_, r_, u, v in pairs:
    if a_ >= 0.30:
        say(f"    {u:<24} x {v:<24} r = {r_:+.3f}")
say(f"  max |r| in the matrix = {pairs[0][0]:.3f} "
    f"({pairs[0][2]} x {pairs[0][3]})")
say(f"  pairs at |r| >= 0.875: {sum(1 for p in pairs if p[0] >= 0.875)}")

n = len(FEAT)
fig, ax = plt.subplots(figsize=(13.5, 12))
ax.grid(False)
im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=12)
ax.set_yticklabels(labs, fontsize=12)
for i in range(n):
    for j in range(n):
        r = C.values[i, j]
        if abs(r) >= 0.28:
            ax.text(j, i, f"{r:.2f}", ha="center", va="center", fontsize=9.5,
                    fontweight="bold",
                    color="white" if abs(r) > 0.62 else "#141414")
ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
ax.grid(which="minor", color="white", lw=1.4)
ax.tick_params(which="minor", length=0)
cb = fig.colorbar(im, ax=ax, shrink=0.62, pad=0.02)
cb.set_label("Pearson $r$", fontsize=13)
ax.set_title(f"Pearson correlation, {n} analytical MAT variables",
             fontsize=19, pad=16, fontweight="bold")
fig.savefig(f"{A.out}/figS2_correlation.png", dpi=200, bbox_inches="tight",
            facecolor="white")
plt.close(); say("  -> figS2_correlation.png")

# =========================================================================
# FIGURE S5 — chronological hold-out + rolling TSCV
# =========================================================================
say("\n--- Figure S5 ---")
EN = "–"
tr, te = (df["year"] < 2000).values, (df["year"] >= 2000).values
say(f"  train n={tr.sum()} (rate {y[tr].mean():.3f})  "
    f"test n={te.sum()} (rate {y[te].mean():.3f})")
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
ax = axes[0]
for (name, mdl), col in zip(
        [("B0 Logistic Regression",
          LogisticRegression(class_weight="balanced", max_iter=500,
                             random_state=42)),
         ("M4 GBT (tuned)",
          GradientBoostingClassifier(random_state=42, **M4P))],
        [PAL["grey"], PAL["c1"]]):
    mdl.fit(X[tr], y[tr])
    p = mdl.predict_proba(X[te])[:, 1]
    fpr, tpr, _ = roc_curve(y[te], p)
    auc = roc_auc_score(y[te], p)
    ax.plot(fpr, tpr, lw=3, color=col, label=f"{name} (AUC = {auc:.3f})")
    say(f"  hold-out {name}: AUC={auc:.4f}")
ax.plot([0, 1], [0, 1], "--", color="grey", lw=1.5, alpha=0.6)
ax.set_xlabel("1 $-$ Specificity", fontsize=FS_AXIS)
ax.set_ylabel("Sensitivity", fontsize=FS_AXIS)
ax.set_title(f"Chronological hold-out\ntrain 1970{EN}1999 (n = {tr.sum()}), "
             f"test 2000{EN}2021 (n = {te.sum()})", fontsize=FS_LABEL - 4, pad=16)
ax.legend(fontsize=FS_TICK - 1, loc="lower right", framealpha=0.92)
ax.tick_params(labelsize=FS_TICK)
panel(ax, "A")

o = np.argsort(df["year"].values, kind="stable")
Xo, yo, yro = X[o], y[o], df["year"].values[o]
folds, flab = [], []
for a_, b_ in TimeSeriesSplit(n_splits=5, test_size=15).split(Xo):
    if len(np.unique(yo[b_])) < 2:
        continue
    f = GradientBoostingClassifier(random_state=42, **M4P).fit(Xo[a_], yo[a_])
    folds.append(float(roc_auc_score(yo[b_], f.predict_proba(Xo[b_])[:, 1])))
    flab.append(f"{yro[b_][0]}{EN}{yro[b_][-1]}")
mu, sd = float(np.mean(folds)), float(np.std(folds))
say(f"  rolling folds={[round(x, 4) for x in folds]}  mean={mu:.4f} sd={sd:.4f}")
# Model seed must track the CV seed, matching the convention used for
# Table 1; holding it at 42 gives 0.776 and would contradict the table.
cvauc = float(np.mean([roc_auc_score(
    y, cross_val_predict(GradientBoostingClassifier(random_state=s, **M4P), X, y,
                         cv=StratifiedKFold(5, shuffle=True, random_state=s),
                         method="predict_proba")[:, 1]) for s in SEEDS]))
say(f"  within-period CV AUC={cvauc:.4f}")

ax = axes[1]
ax.set_axisbelow(True)
bars = ax.bar(range(len(folds)), folds, color=PAL["c4"], edgecolor="white")
for b, v in zip(bars, folds):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
            ha="center", fontsize=FS_BAR + 1, color=PAL["dark"])
ax.axhline(mu, ls="--", color=PAL["c1"], lw=2,
           label=f"rolling mean = {mu:.3f} $\\pm$ {sd:.3f}")
ax.axhline(cvauc, ls=":", color=PAL["dark"], lw=2,
           label=f"within-period CV = {cvauc:.3f}")
ax.set_xticks(range(len(flab)))
ax.set_xticklabels(flab, fontsize=FS_BAR + 2, rotation=20, ha="right")
ax.set_ylabel("AUC", fontsize=FS_AXIS)
ax.set_xlabel("Rolling test window", fontsize=FS_AXIS)
ax.set_title("Rolling time-series cross-validation", fontsize=FS_LABEL, pad=16)
ax.set_ylim(0.4, 1.0)
ax.legend(fontsize=FS_TICK - 2, loc="lower right", framealpha=0.92)
ax.tick_params(labelsize=FS_TICK)
panel(ax, "B")
plt.subplots_adjust(wspace=0.38, top=0.80)
fig.savefig(f"{A.out}/figS5_temporal_validation.png", dpi=200,
            bbox_inches="tight", facecolor="white")
plt.close(); say("  -> figS5_temporal_validation.png")

open(f"{A.out}/s2_s5_values.txt", "w").write("\n".join(LOG))
say(f"\n-> {A.out}/  2 PNG + s2_s5_values.txt")
