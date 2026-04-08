"""
Gaze dynamics analysis: gaze path rate and blink rate during smiles.

Uses per-subject z-scored features from sentence_table_openface.pkl.

Figures:
  gaze_dynamics_contrast.png   — within-subject smile vs no-smile by valence & domain
  gaze_dynamics_subjects.png   — subject-level delta distributions (heterogeneity)
  gaze_dynamics_perievent.png  — peri-smile trajectory (±4 sentences)
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pathlib import Path

BASE       = Path("/Users/marcus/Desktop/usc/VoiceOver")
INPUT_PKL  = BASE / "smile_variable_effects/sentence_table_openface.pkl"
FIG_DIR    = BASE / "smile_variable_effects"

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})
COLORS = {"smile": "#E05C2A", "no_smile": "#4A7FB5"}
VAL_COLORS  = {"negative": "#C0392B", "neutral": "#7F8C8D", "positive": "#27AE60"}
FEAT_LABELS = {"gaze_path_rate_z": "Gaze path rate (z-score)", "blink_rate_z": "Blink rate (z-score)"}
FEAT_KEYS   = list(FEAT_LABELS.keys())

print("Loading data...", flush=True)
df = pd.read_pickle(INPUT_PKL)

# Only sentences with both z-scored features
has_data = df["gaze_path_rate_z"].notna() & df["blink_rate_z"].notna()
df = df[has_data].copy()
print(f"  {len(df):,} sentences with both features, {df['subject'].nunique()} subjects")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: within-subject paired contrast + effect size
# Returns: group means, CI, paired t-test result for (smile, no-smile) groups
# ─────────────────────────────────────────────────────────────────────────────
def paired_contrast(sub_df, feat):
    """For each subject, compute mean(smile) - mean(no_smile). Return deltas."""
    s_smile = sub_df[sub_df["has_smile"]].groupby("subject")[feat].mean()
    s_none  = sub_df[~sub_df["has_smile"]].groupby("subject")[feat].mean()
    both    = pd.DataFrame({"sm": s_smile, "ns": s_none}).dropna()
    if len(both) < 10:
        return None
    delta = both["sm"] - both["ns"]
    t, p  = stats.ttest_rel(both["sm"], both["ns"])
    d     = delta.mean() / delta.std()
    ci95  = 1.96 * delta.std() / np.sqrt(len(delta))
    return {"n": len(both), "delta_mean": delta.mean(), "ci95": ci95,
            "t": t, "p": p, "d": d, "deltas": delta.values}

def sig_str(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Contrasts by valence and content domain
# ─────────────────────────────────────────────────────────────────────────────
print("Building Figure 1: contrasts...", flush=True)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Gaze dynamics during smiles vs. non-smile sentences\n(within-subject z-scored, paired by subject)",
             fontsize=12, fontweight="bold", y=0.98)

panels = [
    (axes[0, 0], "narrative_valence",
     ["negative", "neutral", "positive"],
     "By narrative valence", "gaze_path_rate_z"),
    (axes[0, 1], "narrative_valence",
     ["negative", "neutral", "positive"],
     "By narrative valence", "blink_rate_z"),
    (axes[1, 0], "content_domain",
     ["pre-war", "wartime", "liberation", "post-war", "present-day"],
     "By content domain", "gaze_path_rate_z"),
    (axes[1, 1], "content_domain",
     ["pre-war", "wartime", "liberation", "post-war", "present-day"],
     "By content domain", "blink_rate_z"),
]

for ax, groupby_col, categories, title, feat in panels:
    deltas, labels, cis, ps, ns = [], [], [], [], []
    for cat in categories:
        sub = df[df[groupby_col] == cat]
        res = paired_contrast(sub, feat)
        if res is None or res["n"] < 30:
            continue
        deltas.append(res["delta_mean"])
        cis.append(res["ci95"])
        ps.append(res["p"])
        ns.append(res["n"])
        labels.append(cat)

    xs = np.arange(len(labels))
    bar_colors = [("#C0392B" if "negative" in l or "wartime" in l else
                   "#27AE60" if "positive" in l or "liberation" in l or "pre-war" in l else
                   "#4A7FB5") for l in labels]
    bars = ax.bar(xs, deltas, color=bar_colors, alpha=0.75, width=0.55, edgecolor="white")
    ax.errorbar(xs, deltas, yerr=cis, fmt="none", color="black", capsize=4, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    # Significance annotations
    y_top = max(abs(d) + c for d, c in zip(deltas, cis)) * 1.15
    for i, (d, c, p, n) in enumerate(zip(deltas, cis, ps, ns)):
        ann = sig_str(p)
        y_pos = d + c + y_top * 0.04
        ax.text(i, y_pos, ann, ha="center", va="bottom", fontsize=9, fontweight="bold",
                color="black" if ann != "ns" else "gray")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(FEAT_LABELS[feat], fontsize=9)
    ax.set_title(f"{FEAT_LABELS[feat].split('(')[0].strip()}\n{title}", fontsize=9, fontweight="bold")

    # Add n labels under bars
    for i, (n, d) in enumerate(zip(ns, deltas)):
        ax.text(i, ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.02,
                f"n={n}", ha="center", va="bottom", fontsize=7, color="gray")

plt.tight_layout(rect=[0, 0, 1, 0.97])
out1 = FIG_DIR / "gaze_dynamics_contrast.png"
fig.savefig(out1, bbox_inches="tight")
plt.close()
print(f"  Saved {out1}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Subject-level delta distributions (heterogeneity)
# ─────────────────────────────────────────────────────────────────────────────
print("Building Figure 2: subject deltas...", flush=True)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Subject-level smile effect on gaze dynamics\n(each dot = one subject's Δ[smile − no-smile])",
             fontsize=12, fontweight="bold", y=0.99)

valences = ["negative", "neutral", "positive"]
feats    = ["gaze_path_rate_z", "blink_rate_z"]

for row, feat in enumerate(feats):
    for col, val in enumerate(valences):
        ax = axes[row, col]
        sub = df[df["narrative_valence"] == val]
        res = paired_contrast(sub, feat)
        if res is None:
            ax.set_visible(False)
            continue

        deltas = res["deltas"]
        n_pos  = (deltas > 0).sum()
        n_neg  = (deltas < 0).sum()

        # Sort for better readability
        sorted_d = np.sort(deltas)
        colors   = [COLORS["smile"] if d > 0 else COLORS["no_smile"] for d in sorted_d]
        ax.barh(np.arange(len(sorted_d)), sorted_d, color=colors, height=0.8, alpha=0.6)
        ax.axvline(0, color="black", linewidth=1)
        ax.axvline(res["delta_mean"], color="darkred", linewidth=1.5, linestyle="--",
                   label=f"Mean Δ={res['delta_mean']:+.3f}")

        p_str = sig_str(res["p"])
        ax.set_title(f"{FEAT_LABELS[feat].split('(')[0].strip()}\n{val} · "
                     f"d={res['d']:+.3f} {p_str}  (n={res['n']})",
                     fontsize=8, fontweight="bold")
        ax.set_xlabel("Δ z-score (smile − no-smile)", fontsize=8)
        ax.set_yticks([])
        ax.text(0.02, 0.96, f"↑ more during smile: {n_pos} ({n_pos/len(deltas)*100:.0f}%)",
                transform=ax.transAxes, va="top", fontsize=7.5, color=COLORS["smile"])
        ax.text(0.02, 0.88, f"↓ less during smile: {n_neg} ({n_neg/len(deltas)*100:.0f}%)",
                transform=ax.transAxes, va="top", fontsize=7.5, color=COLORS["no_smile"])
        ax.legend(fontsize=7, loc="lower right")

plt.tight_layout(rect=[0, 0, 1, 0.96])
out2 = FIG_DIR / "gaze_dynamics_subjects.png"
fig.savefig(out2, bbox_inches="tight")
plt.close()
print(f"  Saved {out2}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: Peri-smile sentence trajectory (±4 sentences)
# ─────────────────────────────────────────────────────────────────────────────
print("Building Figure 3: peri-event trajectory...", flush=True)

LAGS    = list(range(-4, 5))   # -4 to +4 sentences relative to smile sentence
MIN_OBS = 30

def peri_event_trajectory(pool_df, feat, lags=LAGS):
    """
    For each smile sentence in pool_df, look up feature values at neighbouring
    sentences (same subject, same tape, sequential index). Return mean ± SE
    at each lag across all qualifying smile events.
    """
    # Build lookup: (subject, video_id, sent_idx_in_video) → feat value
    lookup = pool_df.set_index(["subject", "video_id", "sent_idx_in_video"])[feat]

    # Find smile sentences
    smile_rows = pool_df[pool_df["has_smile"]].copy()

    lag_vals = {l: [] for l in lags}
    for _, row in smile_rows.iterrows():
        base_idx = row["sent_idx_in_video"]
        for lag in lags:
            key = (row["subject"], row["video_id"], base_idx + lag)
            if key in lookup.index:
                v = lookup[key]
                if not np.isnan(v):
                    lag_vals[lag].append(v)

    means = [np.mean(lag_vals[l]) if len(lag_vals[l]) >= MIN_OBS else np.nan for l in lags]
    ses   = [np.std(lag_vals[l]) / np.sqrt(len(lag_vals[l]))
             if len(lag_vals[l]) >= MIN_OBS else np.nan for l in lags]
    counts = [len(lag_vals[l]) for l in lags]
    return np.array(means), np.array(ses), counts

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Peri-smile gaze dynamics trajectory\n(sentences relative to smile sentence, z-scored within subject)",
             fontsize=12, fontweight="bold", y=0.99)

lags_arr = np.array(LAGS)

for row, feat in enumerate(feats):
    for col, val in enumerate(valences):
        ax = axes[row, col]
        pool = df[df["narrative_valence"] == val]
        means, ses, counts = peri_event_trajectory(pool, feat)

        ax.fill_between(lags_arr, means - ses, means + ses, alpha=0.25,
                        color=VAL_COLORS[val])
        ax.plot(lags_arr, means, color=VAL_COLORS[val], linewidth=2.5, marker="o",
                markersize=5, label=val)
        ax.axvline(0, color="black", linewidth=1.5, linestyle="--", alpha=0.7,
                   label="Smile sentence")

        # Mark onset with annotation
        smile_idx = LAGS.index(0)
        if not np.isnan(means[smile_idx]):
            ax.annotate("smile", (0, means[smile_idx]),
                        xytext=(0.4, means[smile_idx] + ses[smile_idx] * 1.5),
                        fontsize=7, color="black", ha="left",
                        arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

        ax.set_xlabel("Lag (sentences)", fontsize=9)
        ax.set_ylabel(FEAT_LABELS[feat], fontsize=9)
        ax.set_title(f"{FEAT_LABELS[feat].split('(')[0].strip()}\n{val} narrative context",
                     fontsize=9, fontweight="bold")
        ax.set_xticks(LAGS)
        ax.grid(axis="y", alpha=0.3)
        n_smile = counts[smile_idx]
        ax.text(0.97, 0.03, f"n={n_smile:,} smiles", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="gray")

plt.tight_layout(rect=[0, 0, 1, 0.96])
out3 = FIG_DIR / "gaze_dynamics_perievent.png"
fig.savefig(out3, bbox_inches="tight")
plt.close()
print(f"  Saved {out3}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4: Gaze path rate vs blink rate scatter — do they co-vary at smiles?
# ─────────────────────────────────────────────────────────────────────────────
print("Building Figure 4: path vs blink scatter...", flush=True)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.suptitle("Gaze path rate vs blink rate per subject × valence\n(within-subject Δ: smile sentences − non-smile sentences)",
             fontsize=11, fontweight="bold")

for ax, val in zip(axes, valences):
    sub = df[df["narrative_valence"] == val]
    sm_path  = sub[sub["has_smile"]].groupby("subject")["gaze_path_rate_z"].mean()
    ns_path  = sub[~sub["has_smile"]].groupby("subject")["gaze_path_rate_z"].mean()
    sm_blink = sub[sub["has_smile"]].groupby("subject")["blink_rate_z"].mean()
    ns_blink = sub[~sub["has_smile"]].groupby("subject")["blink_rate_z"].mean()
    both = pd.DataFrame({"dp": sm_path - ns_path, "db": sm_blink - ns_blink}).dropna()
    if len(both) < 10:
        ax.set_visible(False)
        continue

    r, p = stats.pearsonr(both["dp"], both["db"])
    ax.scatter(both["dp"], both["db"], alpha=0.35, s=20, color=VAL_COLORS[val])
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")

    # Regression line
    m, b = np.polyfit(both["dp"], both["db"], 1)
    xs = np.linspace(both["dp"].min(), both["dp"].max(), 100)
    ax.plot(xs, m*xs + b, color=VAL_COLORS[val], linewidth=2)

    ax.set_xlabel("Δ gaze path rate (z)", fontsize=9)
    ax.set_ylabel("Δ blink rate (z)", fontsize=9)
    p_str = sig_str(p)
    ax.set_title(f"{val.capitalize()} narrative\nr={r:.3f} {p_str}  (n={len(both)})",
                 fontsize=9, fontweight="bold")

plt.tight_layout()
out4 = FIG_DIR / "gaze_dynamics_correlation.png"
fig.savefig(out4, bbox_inches="tight")
plt.close()
print(f"  Saved {out4}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Console summary: statistical results
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("STATISTICAL SUMMARY")
print("="*72)

for feat in feats:
    print(f"\n{'─'*60}")
    print(f"Feature: {feat}")
    print(f"{'─'*60}")
    print(f"  {'Context':<25s}  {'Δ mean':>8s}  {'d':>6s}  {'p':>10s}  sig")
    print(f"  {'─'*25}  {'─'*8}  {'─'*6}  {'─'*10}  {'─'*3}")
    # Overall
    res = paired_contrast(df, feat)
    if res:
        print(f"  {'Overall':<25s}  {res['delta_mean']:>+8.4f}  {res['d']:>+6.3f}  {res['p']:>10.3e}  {sig_str(res['p'])}")
    # By valence
    for val in ["negative", "neutral", "positive"]:
        res = paired_contrast(df[df["narrative_valence"]==val], feat)
        if res:
            print(f"  {'  '+val:<25s}  {res['delta_mean']:>+8.4f}  {res['d']:>+6.3f}  {res['p']:>10.3e}  {sig_str(res['p'])}")
    # By domain
    for dom in ["pre-war", "wartime", "liberation", "post-war", "present-day"]:
        res = paired_contrast(df[df["content_domain"]==dom], feat)
        if res:
            print(f"  {'  '+dom:<25s}  {res['delta_mean']:>+8.4f}  {res['d']:>+6.3f}  {res['p']:>10.3e}  {sig_str(res['p'])}")

print("\nDone.")
