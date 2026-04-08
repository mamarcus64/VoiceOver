"""
Peri-event smile analysis: what happens around smiles in the narrative?

Figures produced:
  1. perievent_valence.png      — peri-event valence trajectories (group)
  2. perievent_categorical.png  — peri-event shifts in LLM features (group)
  3. subject_deltas.png         — raincloud plots of per-subject δ
  4. delta_correlation.png      — correlation matrix of δ across features
  5. subject_portraits.png      — individual timelines for extreme subjects
"""

import pickle, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d
from scipy import stats
from pathlib import Path

BASE    = Path("/Users/marcus/Desktop/usc/VoiceOver")
PKL     = BASE / "smile_variable_effects/sentence_table.pkl"
FIG_DIR = BASE / "smile_variable_effects/figures"
FIG_DIR.mkdir(exist_ok=True)

K = 3       # window ±k sentences
MIN_EVENTS_SUBJECT = 5  # minimum negative-smile events per subject for δ

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load & prepare
# ══════════════════════════════════════════════════════════════════════════════
print("Loading sentence table...", flush=True)
df = pd.read_pickle(PKL)

# Numeric valence encoding
VALENCE_MAP = {"negative": 0.0, "neutral": 0.5, "positive": 1.0, "mixed": np.nan}
df["narr_val_num"]    = df["narrative_valence"].map(VALENCE_MAP)
df["presday_val_num"] = df["present_day_valence"].map(VALENCE_MAP)

# Z-score continuous valences within subject (for cross-signal comparison)
for col in ["audio_valence", "eyegaze_valence"]:
    mu  = df.groupby("subject")[col].transform("mean")
    std = df.groupby("subject")[col].transform("std").clip(lower=1e-9)
    df[f"{col}_z"] = (df[col] - mu) / std
# Also z-score narrative numeric
mu  = df.groupby("subject")["narr_val_num"].transform("mean")
std = df.groupby("subject")["narr_val_num"].transform("std").clip(lower=1e-9)
df["narr_val_z"] = (df["narr_val_num"] - mu) / std

# Boolean masks
df["is_negative"] = df["narrative_valence"] == "negative"

print(f"  {len(df):,} sentences, {df['subject'].nunique()} subjects", flush=True)
print(f"  {df['has_smile'].sum():,} smile sentences, "
      f"{(df['has_smile'] & df['is_negative']).sum():,} negative+smile", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Build lag columns (vectorised via groupby.shift)
# ══════════════════════════════════════════════════════════════════════════════
print("Building lag columns...", flush=True)

CONTINUOUS_FEATURES = ["narr_val_num", "audio_valence", "eyegaze_valence"]
CATEGORICAL_FEATURES = {
    "memory_type":  "external",       # P(external) = stepping outside narrative
    "temporal_syntax": "present_reflection",  # P(present_reflection) = reflective shift
    "narrative_structure": "evaluation",      # P(evaluation) = commentary/irony
}

# For continuous features: lag columns
for feat in CONTINUOUS_FEATURES:
    for lag in range(-K, K + 1):
        col = f"{feat}_L{lag}"
        df[col] = df.groupby("subject")[feat].shift(-lag)

# For categorical features: binary indicator, then lag
for feat, target_val in CATEGORICAL_FEATURES.items():
    bincol = f"{feat}_is_{target_val}"
    df[bincol] = (df[feat] == target_val).astype(float)
    for lag in range(-K, K + 1):
        df[f"{bincol}_L{lag}"] = df.groupby("subject")[bincol].shift(-lag)

# Smile-free window flag: True if ANY smile exists within ±K lags (within subject)
# Used to exclude smile-adjacent sentences from the baseline.
smile_window = pd.Series(False, index=df.index)
for lag in range(-K, K + 1):
    smile_window |= df.groupby("subject")["has_smile"].shift(-lag).fillna(False).astype(bool)
df["smile_in_window"] = smile_window


# ══════════════════════════════════════════════════════════════════════════════
# 3. Peri-event trajectory computation
# ══════════════════════════════════════════════════════════════════════════════
def perievent_trajectory(df, event_mask, baseline_mask, feat_prefix, lags):
    """
    Compute mean ± SE of feature at each lag for events vs baselines.
    First averages within-subject, then across subjects.
    """
    lag_cols = [f"{feat_prefix}_L{l}" for l in lags]
    results = {}
    for label, mask in [("smile", event_mask), ("baseline", baseline_mask)]:
        sub = df.loc[mask, ["subject"] + lag_cols]
        subj_means = sub.groupby("subject")[lag_cols].mean()
        mean = subj_means.mean()
        se   = subj_means.sem()
        results[label] = {
            "lags": np.array(lags),
            "mean": mean.values,
            "se":   se.values,
        }
    return results


def compute_subject_deltas(df, event_mask, baseline_mask, feat_prefix,
                           post_lags=(1, 2, 3), min_events=MIN_EVENTS_SUBJECT):
    """
    Per-subject δ = mean feature at post_lags for events − baselines.
    Returns Series indexed by subject.
    """
    post_cols = [f"{feat_prefix}_L{l}" for l in post_lags]

    event_sub = df.loc[event_mask, ["subject"] + post_cols]
    base_sub  = df.loc[baseline_mask, ["subject"] + post_cols]

    # Require minimum events per subject
    event_counts = event_sub.groupby("subject").size()
    eligible = event_counts[event_counts >= min_events].index

    e_means = event_sub[event_sub["subject"].isin(eligible)].groupby("subject")[post_cols].mean().mean(axis=1)
    b_means = base_sub[base_sub["subject"].isin(eligible)].groupby("subject")[post_cols].mean().mean(axis=1)

    delta = (e_means - b_means).dropna()
    return delta


def permutation_variance_test(df, pool_mask, smile_col, feat_prefix,
                              post_lags=(1, 2, 3), n_perms=200,
                              min_events=MIN_EVENTS_SUBJECT):
    """
    Permutation test: is variance of subject δ greater than expected by chance?
    Shuffles smile/no-smile labels within each subject, recomputes δ variance.
    Vectorised: pre-extracts arrays per subject to avoid repeated DataFrame ops.
    """
    post_cols = [f"{feat_prefix}_L{l}" for l in post_lags]
    sub = df.loc[pool_mask, ["subject", smile_col] + post_cols].dropna()

    # Pre-extract arrays per eligible subject
    prepped = []
    for subj, g in sub.groupby("subject"):
        labels = g[smile_col].values
        vals   = g[post_cols].values.mean(axis=1)  # average across post-lags
        n_ev   = labels.sum()
        n_bl   = (~labels).sum()
        if n_ev >= min_events and n_bl >= min_events:
            prepped.append((labels.copy(), vals))

    if len(prepped) < 10:
        return np.nan, 0

    def _var_of_deltas(prepped_list, shuffle=False):
        ds = np.empty(len(prepped_list))
        for i, (labels, vals) in enumerate(prepped_list):
            if shuffle:
                np.random.shuffle(labels)
            ds[i] = vals[labels].mean() - vals[~labels].mean()
        return np.nanvar(ds)

    real_var  = _var_of_deltas(prepped)
    null_vars = np.array([_var_of_deltas(prepped, shuffle=True) for _ in range(n_perms)])
    p = float(np.mean(null_vars >= real_var))
    return p, len(prepped)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Define event/baseline masks
# ══════════════════════════════════════════════════════════════════════════════
lags = list(range(-K, K + 1))

# Condition A: ALL smiles vs no-smile
mask_smile_all     = df["has_smile"]
mask_baseline_all  = ~df["has_smile"]

# Condition B: Smiles during NEGATIVE narratives vs smile-free-window negative
# Baseline excludes any sentence that has a smile anywhere within ±K lags,
# so the baseline represents "purely negative stretches" with no nearby smiling.
mask_neg           = df["is_negative"]
mask_smile_neg     = df["has_smile"] & mask_neg
mask_baseline_neg  = (~df["smile_in_window"]) & mask_neg

n_baseline = mask_baseline_neg.sum()
print(f"  Strict baseline (no smile in ±{K} window): {n_baseline:,} sentences", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Peri-event valence trajectories (group level)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 1: peri-event valence trajectories...", flush=True)

fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
fig.suptitle("Peri-Event Valence Trajectories Around Smiles\n"
             "(within-subject means ± SE across subjects)",
             fontsize=13, fontweight="bold")

VALENCE_FEATS = [
    ("narr_val_num",    "LLM Narrative Valence"),
    ("audio_valence",   "Audio Valence (continuous)"),
    ("eyegaze_valence", "Eyegaze Valence (continuous)"),
]
CONDITIONS = [
    ("All Smiles",                mask_smile_all, mask_baseline_all),
    ("Smiles During Negative",    mask_smile_neg, mask_baseline_neg),
]

for row, (feat, label) in enumerate(VALENCE_FEATS):
    for col, (cond_label, ev_mask, bl_mask) in enumerate(CONDITIONS):
        ax = axes[row, col]
        res = perievent_trajectory(df, ev_mask, bl_mask, feat, lags)
        for key, color, lbl in [("smile", "#e11d48", "Smile"),
                                 ("baseline", "#6b7280", "Baseline")]:
            r = res[key]
            ax.plot(r["lags"], r["mean"], color=color, lw=2, label=lbl)
            ax.fill_between(r["lags"],
                            r["mean"] - r["se"],
                            r["mean"] + r["se"],
                            color=color, alpha=0.15)
        ax.axvline(0, color="#ddd", lw=1, ls="--")
        ax.set_xlabel("Lag (sentences)")
        if col == 0:
            ax.set_ylabel(label, fontsize=9)
        if row == 0:
            ax.set_title(cond_label, fontweight="bold")
        ax.legend(fontsize=8)

fig.savefig(FIG_DIR / "perievent_valence.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved perievent_valence.png", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Peri-event categorical feature shifts (group level)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 2: peri-event categorical shifts...", flush=True)

fig, axes = plt.subplots(len(CATEGORICAL_FEATURES), 2,
                         figsize=(13, 3.5 * len(CATEGORICAL_FEATURES)),
                         constrained_layout=True)
fig.suptitle("Peri-Event Probability Shifts in Categorical Features\n"
             "(P(value) at each lag, smile events vs baseline)",
             fontsize=13, fontweight="bold")

for row, (feat, target_val) in enumerate(CATEGORICAL_FEATURES.items()):
    bincol = f"{feat}_is_{target_val}"
    nice_label = f"P({target_val.replace('_', ' ')})"
    for col_i, (cond_label, ev_mask, bl_mask) in enumerate(CONDITIONS):
        ax = axes[row, col_i]
        res = perievent_trajectory(df, ev_mask, bl_mask, bincol, lags)
        for key, color, lbl in [("smile", "#e11d48", "Smile"),
                                 ("baseline", "#6b7280", "Baseline")]:
            r = res[key]
            ax.plot(r["lags"], r["mean"], color=color, lw=2, label=lbl, marker="o", ms=4)
            ax.fill_between(r["lags"], r["mean"] - r["se"],
                            r["mean"] + r["se"], color=color, alpha=0.15)
        ax.axvline(0, color="#ddd", lw=1, ls="--")
        ax.set_xlabel("Lag (sentences)")
        if col_i == 0:
            ax.set_ylabel(nice_label, fontsize=9)
        if row == 0:
            ax.set_title(cond_label, fontweight="bold")
        ax.legend(fontsize=8)

fig.savefig(FIG_DIR / "perievent_categorical.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved perievent_categorical.png", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Compute subject-level deltas
# ══════════════════════════════════════════════════════════════════════════════
print("Computing subject-level deltas...", flush=True)

DELTA_FEATURES = {
    "narr_val_num":                        "Narrative Valence",
    "audio_valence":                       "Audio Valence",
    "eyegaze_valence":                     "Eyegaze Valence",
    "memory_type_is_external":             "P(External Memory)",
    "temporal_syntax_is_present_reflection":"P(Present Reflection)",
    "narrative_structure_is_evaluation":    "P(Evaluation)",
}

# Compute δ for the negative-smile condition
deltas = {}
for feat, label in DELTA_FEATURES.items():
    d = compute_subject_deltas(df, mask_smile_neg, mask_baseline_neg, feat)
    deltas[feat] = d
    n = len(d)
    if n > 0:
        print(f"  {label}: n={n}, mean δ={d.mean():.4f}, std={d.std():.4f}", flush=True)

# Heterogeneity tests
print("\nHeterogeneity tests (permutation)...", flush=True)
het_results = {}
for feat, label in DELTA_FEATURES.items():
    p, n = permutation_variance_test(df, mask_neg, "has_smile", feat)
    het_results[feat] = (p, n)
    print(f"  {label}: p={p:.3f} (n={n} subjects)", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Raincloud plots of per-subject δ
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 3: subject-level delta distributions...", flush=True)

feats_to_plot = [f for f in DELTA_FEATURES if len(deltas.get(f, [])) > 0]
n_feats = len(feats_to_plot)

fig, axes = plt.subplots(n_feats, 1, figsize=(10, 2.5 * n_feats),
                         constrained_layout=True)
fig.suptitle("Per-Subject δ: Smile-During-Negative vs Baseline\n"
             "(δ > 0 = post-smile shift toward positive / higher value)",
             fontsize=12, fontweight="bold")
if n_feats == 1:
    axes = [axes]

for i, feat in enumerate(feats_to_plot):
    ax = axes[i]
    d = deltas[feat].dropna()
    label = DELTA_FEATURES[feat]
    p_het = het_results.get(feat, (np.nan, 0))[0]

    # Half-violin (kernel density)
    from scipy.stats import gaussian_kde
    if len(d) > 5:
        kde = gaussian_kde(d.values, bw_method=0.3)
        x_grid = np.linspace(d.min() - d.std(), d.max() + d.std(), 200)
        density = kde(x_grid)
        density = density / density.max() * 0.35  # normalise height
        ax.fill_betweenx(x_grid, 0.55, 0.55 + density, color="#e11d48", alpha=0.25)
        ax.plot(0.55 + density, x_grid, color="#e11d48", lw=1)

    # Jitter strip
    jitter = np.random.uniform(-0.15, 0.15, len(d))
    ax.scatter(jitter, d.values, s=8, alpha=0.4, color="#e11d48", zorder=3)

    # Box plot
    bp = ax.boxplot(d.values, positions=[-0.3], vert=True, widths=0.2,
                    patch_artist=True, showfliers=False)
    bp["boxes"][0].set_facecolor("#fecdd3")
    bp["boxes"][0].set_edgecolor("#e11d48")
    bp["medians"][0].set_color("#e11d48")

    ax.axhline(0, color="#999", lw=1, ls="--")
    ax.set_xlim(-0.6, 1.2)
    ax.set_xticks([])

    title = f"{label}  (n={len(d)}"
    if not np.isnan(p_het):
        sig = " ***" if p_het < 0.001 else " **" if p_het < 0.01 else " *" if p_het < 0.05 else ""
        title += f", het. p={p_het:.3f}{sig}"
    title += ")"
    ax.set_title(title, fontsize=9, fontweight="bold", loc="left")
    ax.set_ylabel("δ")

fig.savefig(FIG_DIR / "subject_deltas.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved subject_deltas.png", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Correlation matrix of subject-level δ
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 4: delta correlation matrix...", flush=True)

delta_df = pd.DataFrame({DELTA_FEATURES[f]: deltas[f] for f in feats_to_plot})
delta_df = delta_df.dropna()

if len(delta_df) > 20:
    corr = delta_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.index, fontsize=8)
    for yi in range(len(corr)):
        for xi in range(len(corr)):
            ax.text(xi, yi, f"{corr.iloc[yi, xi]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(corr.iloc[yi, xi]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Correlation of Per-Subject δ Across Features\n"
                 "(smiles during negative narratives)",
                 fontweight="bold", fontsize=11)
    fig.savefig(FIG_DIR / "delta_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved delta_correlation.png", flush=True)
else:
    print("  Skipped (too few subjects with all features)", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Individual subject portraits (extremes of δ_narrative_valence)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 5: individual subject portraits...", flush=True)

narr_delta = deltas.get("narr_val_num", pd.Series(dtype=float))
if len(narr_delta) > 10:
    sorted_d = narr_delta.sort_values()
    bracers = sorted_d.head(3).index.tolist()
    hopers  = sorted_d.tail(3).index.tolist()
    example_subjects = bracers + hopers
    n_examples = len(example_subjects)

    fig, axes = plt.subplots(n_examples, 1, figsize=(14, 3 * n_examples),
                             constrained_layout=True)
    fig.suptitle("Individual Subject Portraits: Narrative Valence Around Smiles\n"
                 "Top 3 = most negative δ ('bracers'), Bottom 3 = most positive δ ('hopers')",
                 fontsize=12, fontweight="bold")

    for i, subj in enumerate(example_subjects):
        ax = axes[i]
        sub = df[df["subject"] == subj].sort_values("subj_sent_idx")
        x = sub["subj_sent_idx"].values
        v = sub["narr_val_num"].values

        # Plot valence as a colored scatter
        colors = np.where(v < 0, "#ef4444", np.where(v > 0, "#22c55e", "#d1d5db"))
        ax.scatter(x, v, c=colors, s=6, alpha=0.5, zorder=2)

        # Smooth trend line
        valid = ~np.isnan(v)
        if valid.sum() > 10:
            from scipy.interpolate import UnivariateSpline
            try:
                spl = UnivariateSpline(x[valid], v[valid], s=len(x[valid]) * 0.5)
                ax.plot(x[valid], spl(x[valid]), color="#1d4ed8", lw=1.5, alpha=0.7)
            except Exception:
                pass

        # Mark smile events during negative
        smile_neg = sub[(sub["has_smile"]) & (sub["is_negative"])]
        ax.scatter(smile_neg["subj_sent_idx"].values,
                   smile_neg["narr_val_num"].values,
                   marker="v", s=40, color="#e11d48", edgecolors="white",
                   linewidths=0.5, zorder=5, label="Smile during negative")

        # Tape boundaries
        n_tapes = int(sub["tape"].nunique())
        if n_tapes > 1:
            tape_boundaries = sub.groupby("tape_rank")["subj_sent_idx"].min().values[1:]
            for tb in tape_boundaries:
                ax.axvline(tb, color="#ccc", lw=0.6, ls=":")

        delta_val = narr_delta.get(subj, np.nan)
        tag = "BRACER" if subj in bracers else "HOPER"
        ax.set_title(f"Subject {subj} ({tag}, δ={delta_val:.3f}, "
                     f"{len(smile_neg)} neg-smile events, "
                     f"{len(sub)} sentences, {n_tapes} tapes)",
                     fontsize=9, fontweight="bold", loc="left")
        ax.set_ylabel("Valence")
        ax.set_ylim(-1.5, 1.5)
        ax.axhline(0, color="#eee", lw=0.8)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
        if i == n_examples - 1:
            ax.set_xlabel("Sentence index (concatenated across tapes)")

    fig.savefig(FIG_DIR / "subject_portraits.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved subject_portraits.png", flush=True)
else:
    print("  Skipped (too few subjects)", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Summary stats
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total sentences:          {len(df):,}")
print(f"Smile sentences:          {df['has_smile'].sum():,} ({df['has_smile'].mean()*100:.1f}%)")
print(f"Negative sentences:       {mask_neg.sum():,}")
print(f"Negative + smile:         {mask_smile_neg.sum():,}")
print(f"Subjects with ≥{MIN_EVENTS_SUBJECT} neg-smile events: {len(narr_delta)}")
print()
print("Group-level narrative valence trajectory (negative+smile condition):")
res = perievent_trajectory(df, mask_smile_neg, mask_baseline_neg, "narr_val_num", lags)
for lag in lags:
    idx = lag + K
    sm = res["smile"]["mean"][idx]
    bl = res["baseline"]["mean"][idx]
    print(f"  lag {lag:+d}: smile={sm:.4f}  baseline={bl:.4f}  Δ={sm-bl:.4f}")
print("\nDone.")
