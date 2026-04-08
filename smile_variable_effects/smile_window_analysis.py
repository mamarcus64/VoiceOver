"""
Smile-window gaze dynamics analysis.

Unit of analysis: each smile window (precise [start_ts, end_ts]).
Features are z-scored against the subject's non-smile sentence distribution
within the same narrative valence — so z=+1 means "1 SD above this person's
typical same-valence sentence."

Figures:
  sw_contrast.png          — overall + by valence + by content domain
  sw_neg_clusters.png      — negative-valence smiles broken out by topic cluster
  sw_subjects.png          — subject-level distributions (heterogeneity)
  sw_score_interaction.png — does effect vary with smile confidence score?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

BASE      = Path("/Users/marcus/Desktop/usc/VoiceOver")
INPUT_PKL = BASE / "smile_variable_effects/smile_window_features.pkl"
FIG_DIR   = BASE / "smile_variable_effects"

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

FEAT_LABELS = {
    "gaze_path_rate_z": "Gaze path rate (z vs. subject baseline)",
    "blink_rate_z":     "Blink rate (z vs. subject baseline)",
}
FEATS = list(FEAT_LABELS.keys())

VAL_COLORS = {"negative": "#C0392B", "neutral": "#7F8C8D", "positive": "#27AE60"}
DOM_COLORS = {
    "pre-war": "#3498DB", "wartime": "#C0392B",
    "liberation": "#27AE60", "post-war": "#E67E22", "present-day": "#9B59B6",
}
CLUSTER_COLORS = {
    "imprisonment/captivity": "#8B0000",
    "physical suffering":     "#C0392B",
    "emotional reflection":   "#E67E22",
    "family/loss":            "#9B59B6",
    "persecution/politics":   "#2C3E50",
    "displacement/flight":    "#2980B9",
    "post-war aftermath":     "#16A085",
    "pre-war life":           "#27AE60",
    "liberation":             "#F39C12",
    "other":                  "#95A5A6",
}

print("Loading smile window features...", flush=True)
sw = pd.read_pickle(INPUT_PKL)
print(f"  {len(sw):,} smiles, {sw['subject'].nunique()} subjects")

# ── Helper: one-sample t-test of z-scores (are they different from 0?) ────────
def one_sample_test(vals):
    """t-test against 0 on the per-subject means (each subject contributes one value)."""
    subj_means = vals.groupby(sw.loc[vals.index, "subject"]).mean()
    t, p = stats.ttest_1samp(subj_means.dropna(), 0)
    d    = subj_means.mean() / subj_means.std()
    ci95 = 1.96 * subj_means.std() / np.sqrt(len(subj_means))
    return {"mean": subj_means.mean(), "ci95": ci95, "d": d, "t": t, "p": p,
            "n_subj": len(subj_means), "n_smiles": len(vals)}

def sig_str(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def annotate_bar(ax, x, y_top, p, d, fontsize=8):
    ann = sig_str(p)
    color = "black" if ann != "ns" else "#888"
    ax.text(x, y_top, ann, ha="center", va="bottom", fontsize=fontsize,
            fontweight="bold" if ann != "ns" else "normal", color=color)
    ax.text(x, y_top * 0.92 if y_top > 0 else y_top * 1.08,
            f"d={d:+.2f}", ha="center", va="top", fontsize=6.5, color="#555")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Contrast by valence and content domain
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 1: contrast plots...", flush=True)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    "Gaze dynamics during smile windows\n"
    "(z-scored vs. subject's non-smile baseline in same narrative context)",
    fontsize=12, fontweight="bold", y=0.99,
)

panels = [
    (axes[0, 0], "narrative_valence",
     ["negative", "neutral", "positive"],
     VAL_COLORS, "gaze_path_rate_z", "By narrative valence"),
    (axes[0, 1], "narrative_valence",
     ["negative", "neutral", "positive"],
     VAL_COLORS, "blink_rate_z", "By narrative valence"),
    (axes[1, 0], "content_domain",
     ["pre-war", "wartime", "liberation", "post-war", "present-day"],
     DOM_COLORS, "gaze_path_rate_z", "By content domain"),
    (axes[1, 1], "content_domain",
     ["pre-war", "wartime", "liberation", "post-war", "present-day"],
     DOM_COLORS, "blink_rate_z", "By content domain"),
]

for ax, col, cats, cmap, feat, subtitle in panels:
    results = []
    for cat in cats:
        mask = sw[col] == cat
        if mask.sum() < 50:
            continue
        vals = sw.loc[mask, feat].dropna()
        res  = one_sample_test(vals)
        results.append((cat, res))

    if not results:
        ax.set_visible(False)
        continue

    xs     = np.arange(len(results))
    labels = [r[0] for r in results]
    means  = [r[1]["mean"] for r in results]
    cis    = [r[1]["ci95"] for r in results]
    ps     = [r[1]["p"] for r in results]
    ds     = [r[1]["d"] for r in results]
    colors = [cmap.get(lbl, "#999") for lbl in labels]

    bars = ax.bar(xs, means, color=colors, alpha=0.75, width=0.55, edgecolor="white")
    ax.errorbar(xs, means, yerr=cis, fmt="none", color="black", capsize=4, lw=1.5)
    ax.axhline(0, color="black", lw=0.9, linestyle="--")

    y_range = max(abs(m) + c for m, c in zip(means, cis)) if means else 0.1
    ax.set_ylim(-y_range * 1.5, y_range * 1.6)

    for i, (m, c, p, d) in enumerate(zip(means, cis, ps, ds)):
        sign = 1 if m >= 0 else -1
        annotate_bar(ax, i, (abs(m) + c) * sign * 1.15, p, d)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(FEAT_LABELS[feat], fontsize=9)
    ax.set_title(f"{FEAT_LABELS[feat].split('(')[0].strip()}\n{subtitle}", fontsize=9, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8, ls="--")

    for i, (r_lbl, res) in enumerate(results):
        ax.text(i, ax.get_ylim()[0] * 0.9,
                f"n={res['n_subj']}", ha="center", va="top", fontsize=6.5, color="gray")

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(FIG_DIR / "sw_contrast.png", bbox_inches="tight")
plt.close()
print("  Saved sw_contrast.png", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Negative-valence smiles broken out by topic cluster
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 2: negative subclusters...", flush=True)

neg_sw = sw[sw["narrative_valence"] == "negative"].copy()
clusters = (
    neg_sw["neg_cluster"]
    .value_counts()
    .loc[lambda s: s >= 50]
    .index.tolist()
)
# Sort by gaze_path_rate_z mean for readability
cluster_means = {c: neg_sw.loc[neg_sw["neg_cluster"]==c, "gaze_path_rate_z"].mean()
                 for c in clusters}
clusters = sorted(clusters, key=lambda c: cluster_means[c])

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle(
    "Gaze dynamics during smiles — breakdown of NEGATIVE valence by topic\n"
    "(z=0: subject's typical negative sentence; positive = more active than baseline)",
    fontsize=11, fontweight="bold",
)

for ax, feat in zip(axes, FEATS):
    results = []
    for cat in clusters:
        vals = neg_sw.loc[neg_sw["neg_cluster"] == cat, feat].dropna()
        if len(vals) < 30:
            continue
        res = one_sample_test(vals)
        results.append((cat, res))

    xs     = np.arange(len(results))
    labels = [r[0] for r in results]
    means  = [r[1]["mean"] for r in results]
    cis    = [r[1]["ci95"] for r in results]
    ps     = [r[1]["p"] for r in results]
    ds     = [r[1]["d"] for r in results]
    n_sm   = [r[1]["n_smiles"] for r in results]
    colors = [CLUSTER_COLORS.get(lbl, "#999") for lbl in labels]

    ax.barh(xs, means, color=colors, alpha=0.8, height=0.6, edgecolor="white")
    ax.errorbar(means, xs, xerr=cis, fmt="none", color="black", capsize=4, lw=1.5)
    ax.axvline(0, color="black", lw=1, ls="--")

    y_range = max(abs(m) + c for m, c in zip(means, cis)) if means else 0.1
    for i, (m, c, p, d, n) in enumerate(zip(means, cis, ps, ds, n_sm)):
        xpos = (abs(m) + c) * (1 if m >= 0 else -1)
        xpos += 0.005 * (1 if m >= 0 else -1)
        ax.text(xpos, i, f"{sig_str(p)}  d={d:+.2f}", va="center", fontsize=7.5,
                color="black" if sig_str(p) != "ns" else "#888")
        ax.text(-y_range * 1.3, i, f"n={n:,}", va="center", ha="right",
                fontsize=7, color="gray")

    ax.set_yticks(xs)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(FEAT_LABELS[feat], fontsize=9)
    ax.set_title(FEAT_LABELS[feat], fontsize=9, fontweight="bold")
    ax.set_xlim(-y_range * 1.6, y_range * 1.8)

plt.tight_layout()
fig.savefig(FIG_DIR / "sw_neg_clusters.png", bbox_inches="tight")
plt.close()
print("  Saved sw_neg_clusters.png", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Subject-level distributions (heterogeneity)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 3: subject-level distributions...", flush=True)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle(
    "Subject-level smile gaze dynamics (each dot = one subject's mean z-score)\n"
    "Positive = more eye activity during smiles than that subject's same-context baseline",
    fontsize=11, fontweight="bold", y=0.99,
)

valences = ["negative", "neutral", "positive"]
for row, feat in enumerate(FEATS):
    for col, val in enumerate(valences):
        ax = axes[row, col]
        mask = sw["narrative_valence"] == val
        subj_means = (
            sw.loc[mask & sw[feat].notna()]
            .groupby("subject")[feat].mean()
            .dropna()
        )
        if len(subj_means) < 10:
            ax.set_visible(False)
            continue

        t, p    = stats.ttest_1samp(subj_means, 0)
        d       = subj_means.mean() / subj_means.std()
        n_pos   = (subj_means > 0).sum()
        n_neg   = (subj_means < 0).sum()
        sorted_s = subj_means.sort_values().values
        colors   = ["#E05C2A" if v > 0 else "#4A7FB5" for v in sorted_s]

        ax.barh(np.arange(len(sorted_s)), sorted_s, color=colors, height=0.8, alpha=0.6)
        ax.axvline(0, color="black", lw=1)
        ax.axvline(subj_means.mean(), color="darkred", lw=1.5, ls="--",
                   label=f"Mean={subj_means.mean():+.3f}")

        p_str = sig_str(p)
        ax.set_title(
            f"{FEAT_LABELS[feat].split('(')[0].strip()}\n"
            f"{val}  |  d={d:+.3f} {p_str}  (n={len(subj_means)} subj)",
            fontsize=8, fontweight="bold",
        )
        ax.set_xlabel("Mean z-score during smiles", fontsize=8)
        ax.set_yticks([])
        ax.text(0.02, 0.97, f"↑ more active: {n_pos} ({n_pos/len(subj_means)*100:.0f}%)",
                transform=ax.transAxes, va="top", fontsize=7.5, color="#E05C2A")
        ax.text(0.02, 0.88, f"↓ less active: {n_neg} ({n_neg/len(subj_means)*100:.0f}%)",
                transform=ax.transAxes, va="top", fontsize=7.5, color="#4A7FB5")
        ax.legend(fontsize=7, loc="lower right")

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG_DIR / "sw_subjects.png", bbox_inches="tight")
plt.close()
print("  Saved sw_subjects.png", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Smile score interaction — high vs. low confidence smiles
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 4: smile score interaction...", flush=True)

q25, q75 = sw["smile_score"].quantile([0.25, 0.75])
sw["score_group"] = pd.cut(sw["smile_score"],
                           bins=[0, q25, q75, 1],
                           labels=["low (Q1)", "mid (Q2–3)", "high (Q4)"])

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle(
    f"Effect by smile confidence score\n"
    f"(Q1 ≤ {q25:.2f} < mid < {q75:.2f} ≤ Q4)",
    fontsize=11, fontweight="bold",
)

score_colors = {"low (Q1)": "#BDC3C7", "mid (Q2–3)": "#85929E", "high (Q4)": "#2C3E50"}

for ax, feat in zip(axes, FEATS):
    groups  = ["low (Q1)", "mid (Q2–3)", "high (Q4)"]
    results = []
    for g in groups:
        vals = sw.loc[sw["score_group"] == g, feat].dropna()
        if len(vals) < 30:
            continue
        res = one_sample_test(vals)
        results.append((g, res))

    xs     = np.arange(len(results))
    labels = [r[0] for r in results]
    means  = [r[1]["mean"] for r in results]
    cis    = [r[1]["ci95"] for r in results]
    ps     = [r[1]["p"] for r in results]
    ds     = [r[1]["d"] for r in results]
    colors = [score_colors[lbl] for lbl in labels]

    ax.bar(xs, means, color=colors, alpha=0.85, width=0.5, edgecolor="white")
    ax.errorbar(xs, means, yerr=cis, fmt="none", color="black", capsize=5, lw=1.5)
    ax.axhline(0, color="black", lw=0.9, ls="--")

    y_range = max(abs(m) + c for m, c in zip(means, cis)) if means else 0.1
    ax.set_ylim(-y_range * 1.4, y_range * 1.8)

    for i, (m, c, p, d) in enumerate(zip(means, cis, ps, ds)):
        sign = 1 if m >= 0 else -1
        annotate_bar(ax, i, (abs(m) + c) * sign * 1.1, p, d)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(FEAT_LABELS[feat], fontsize=9)
    ax.set_title(FEAT_LABELS[feat], fontsize=9, fontweight="bold")

plt.tight_layout()
fig.savefig(FIG_DIR / "sw_score_interaction.png", bbox_inches="tight")
plt.close()
print("  Saved sw_score_interaction.png", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Console summary
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("STATISTICAL SUMMARY — smile-window analysis")
print("="*72)
print(f"  q25 smile score={q25:.3f}, q75={q75:.3f}")

for feat in FEATS:
    print(f"\n{'─'*65}")
    print(f"Feature: {feat}")
    print(f"{'─'*65}")
    print(f"  {'Condition':<35s}  {'mean z':>7s}  {'d':>6s}  {'p':>10s}  sig  n_subj")
    print(f"  {'─'*35}  {'─'*7}  {'─'*6}  {'─'*10}  {'─'*3}  {'─'*6}")

    # Overall
    res = one_sample_test(sw[feat].dropna())
    print(f"  {'Overall':<35s}  {res['mean']:>+7.4f}  {res['d']:>+6.3f}  {res['p']:>10.3e}  {sig_str(res['p'])}  {res['n_subj']}")

    # By valence
    for val in ["negative", "neutral", "positive"]:
        vals = sw.loc[sw["narrative_valence"]==val, feat].dropna()
        if len(vals) < 50: continue
        res = one_sample_test(vals)
        print(f"  {'  '+val:<35s}  {res['mean']:>+7.4f}  {res['d']:>+6.3f}  {res['p']:>10.3e}  {sig_str(res['p'])}  {res['n_subj']}")

    # Negative subclusters
    for clust in clusters:
        vals = sw.loc[(sw["narrative_valence"]=="negative") & (sw["neg_cluster"]==clust), feat].dropna()
        if len(vals) < 30: continue
        res = one_sample_test(vals)
        print(f"  {'    neg:'+clust:<35s}  {res['mean']:>+7.4f}  {res['d']:>+6.3f}  {res['p']:>10.3e}  {sig_str(res['p'])}  {res['n_subj']}")

    # By domain
    for dom in ["pre-war", "wartime", "liberation", "post-war", "present-day"]:
        vals = sw.loc[sw["content_domain"]==dom, feat].dropna()
        if len(vals) < 50: continue
        res = one_sample_test(vals)
        print(f"  {'  dom:'+dom:<35s}  {res['mean']:>+7.4f}  {res['d']:>+6.3f}  {res['p']:>10.3e}  {sig_str(res['p'])}  {res['n_subj']}")

    # By smile score
    for g in ["low (Q1)", "mid (Q2–3)", "high (Q4)"]:
        vals = sw.loc[sw["score_group"]==g, feat].dropna()
        if len(vals) < 50: continue
        res = one_sample_test(vals)
        print(f"  {'  score:'+g:<35s}  {res['mean']:>+7.4f}  {res['d']:>+6.3f}  {res['p']:>10.3e}  {sig_str(res['p'])}  {res['n_subj']}")

print("\nDone.")
