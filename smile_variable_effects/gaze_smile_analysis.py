"""
Gaze × Smile analyses:
  A. Conditional gaze contrasts: smile vs non-smile across narrative contexts
  B. Within-subject correlations: gaze aversion ↔ valence among smile sentences
  C. Frame-level dynamics: gaze trajectory ±1.5s around smile onset
  E. Onset-to-offset gaze shift during smiles by narrative context

Figures: gaze_contrasts.png, gaze_within_subject_corr.png,
         gaze_smile_onset.png, gaze_smile_shift.png
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

BASE    = Path("/Users/marcus/Desktop/usc/VoiceOver/smile_variable_effects")
FIG_DIR = BASE / "figures"

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...", flush=True)
df       = pd.read_pickle(BASE / "sentence_table_gaze.pkl")
traj     = pickle.loads((BASE / "smile_trajectories.pkl").read_bytes())
shift_df = pd.read_pickle(BASE / "smile_gaze_shifts.pkl")

VALENCE_MAP = {"negative": -1, "neutral": 0, "positive": 1, "mixed": np.nan}
df["narr_val_num"] = df["narrative_valence"].map(VALENCE_MAP)
has_gaze = df["gaze_mean_sd"].notna()

print(f"  {has_gaze.sum():,} sentences with gaze data", flush=True)
print(f"  {(has_gaze & df['has_smile']).sum():,} smile sentences with gaze", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# A. Conditional Gaze Contrasts
# ══════════════════════════════════════════════════════════════════════════════
print("Analysis A: conditional contrasts...", flush=True)

def grouped_contrast(df, feature_col, feature_vals, gaze_col="gaze_mean_sd"):
    """
    For each (feature_value, smile/no-smile): compute within-subject mean gaze,
    then return group mean ± SE.
    """
    results = []
    for val in feature_vals:
        for smile, label in [(True, "Smile"), (False, "No smile")]:
            mask = has_gaze & (df[feature_col] == val) & (df["has_smile"] == smile)
            sub = df.loc[mask, ["subject", gaze_col]]
            subj_means = sub.groupby("subject")[gaze_col].mean()
            results.append({
                "feature_val": val,
                "smile":       label,
                "mean":        subj_means.mean(),
                "se":          subj_means.sem(),
                "n_subjects":  len(subj_means),
            })
    return pd.DataFrame(results)


PANELS = [
    ("narrative_valence",    ["negative", "neutral", "positive"],
     "Narrative Valence"),
    ("memory_type",          ["internal", "external"],
     "Memory Type"),
    ("content_domain",       ["pre-war", "wartime", "liberation", "post-war"],
     "Content Domain"),
    ("narrative_structure",  ["complicating_action", "evaluation", "orientation", "resolution"],
     "Narrative Structure"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
fig.suptitle("Gaze Aversion by Narrative Context: Smile vs Non-Smile Sentences\n"
             "(within-subject means ± SE; higher = more averted from interviewer)",
             fontsize=12, fontweight="bold")

COLORS = {"Smile": "#e11d48", "No smile": "#6b7280"}

for ax, (col, vals, title) in zip(axes.flat, PANELS):
    res = grouped_contrast(df, col, vals)
    x = np.arange(len(vals))
    w = 0.35
    for i, (label, color) in enumerate(COLORS.items()):
        sub = res[res["smile"] == label]
        offset = -w/2 + i * w
        bars = ax.bar(x + offset, sub["mean"].values, w,
                       yerr=sub["se"].values, capsize=3,
                       color=color, alpha=0.85, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("_", " ") for v in vals], fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Mean gaze SD level", fontsize=9)
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)

fig.savefig(FIG_DIR / "gaze_contrasts.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved gaze_contrasts.png", flush=True)

# Print key contrasts
for col, vals, title in PANELS:
    res = grouped_contrast(df, col, vals)
    print(f"\n  {title}:")
    for _, r in res.iterrows():
        print(f"    {r['feature_val']:25s} {r['smile']:9s}  "
              f"mean={r['mean']:.4f} ± {r['se']:.4f}  (n={r['n_subjects']})")


# ══════════════════════════════════════════════════════════════════════════════
# B. Within-Subject Correlations
# ══════════════════════════════════════════════════════════════════════════════
print("\nAnalysis B: within-subject correlations...", flush=True)

def within_subject_corr(df, mask, gaze_col="gaze_mean_sd", val_col="narr_val_num", min_n=20):
    """
    For each subject, compute Spearman ρ between gaze and valence among
    the masked sentences. Returns Series of ρ values.
    """
    sub = df.loc[mask & df[gaze_col].notna() & df[val_col].notna(),
                 ["subject", gaze_col, val_col]]
    rhos = []
    for subj, g in sub.groupby("subject"):
        if len(g) < min_n:
            continue
        rho, _ = stats.spearmanr(g[gaze_col], g[val_col])
        if not np.isnan(rho):
            rhos.append({"subject": subj, "rho": rho})
    return pd.DataFrame(rhos)


rho_smile    = within_subject_corr(df, df["has_smile"] & has_gaze)
rho_nosmile  = within_subject_corr(df, (~df["has_smile"]) & has_gaze)

fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
fig.suptitle("Within-Subject ρ(Gaze Aversion, Narrative Valence)\n"
             "ρ < 0 = more averted during negative content",
             fontsize=12, fontweight="bold")

bins = np.linspace(-0.6, 0.6, 50)
ax.hist(rho_nosmile["rho"], bins=bins, alpha=0.45, color="#6b7280",
        label=f"Non-smile sentences (n={len(rho_nosmile)})", density=True)
ax.hist(rho_smile["rho"], bins=bins, alpha=0.55, color="#e11d48",
        label=f"Smile sentences (n={len(rho_smile)})", density=True)
ax.axvline(0, color="#ccc", lw=1, ls="--")

for rdf, color, label in [(rho_smile, "#e11d48", "Smile"),
                           (rho_nosmile, "#6b7280", "Non-smile")]:
    m = rdf["rho"].mean()
    ax.axvline(m, color=color, lw=2, ls="-",
               label=f"  {label} mean ρ = {m:.3f}")

ax.set_xlabel("Spearman ρ (gaze aversion vs narrative valence)", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.legend(fontsize=9)

fig.savefig(FIG_DIR / "gaze_within_subject_corr.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved gaze_within_subject_corr.png", flush=True)

# Stats
for label, rdf in [("Smile", rho_smile), ("Non-smile", rho_nosmile)]:
    m = rdf["rho"].mean()
    t, p = stats.ttest_1samp(rdf["rho"], 0)
    print(f"  {label}: mean ρ = {m:.4f}, t={t:.2f}, p={p:.1e}, n={len(rdf)}")


# ══════════════════════════════════════════════════════════════════════════════
# C. Frame-Level Gaze Dynamics Around Smile Onset
# ══════════════════════════════════════════════════════════════════════════════
print("\nAnalysis C: frame-level dynamics around smile onset...", flush=True)

subjects  = np.array(traj["subjects"])
valences  = np.array(traj["valences"])
matrix    = traj["matrix"]               # (n_smiles, 91), float32
window    = traj["onset_window"]          # 45
offsets   = np.arange(-window, window + 1)
time_s    = offsets / 30.0                # seconds

# Split by negative vs non-negative
neg_mask = valences == "negative"

def subject_mean_trajectory(mask):
    """
    Within each subject, average trajectories for the masked smiles.
    Return (mean_across_subjects, se_across_subjects) at each offset.
    """
    sub_mat = matrix[mask]
    sub_subj = subjects[mask]
    unique_subj = np.unique(sub_subj)

    subj_curves = []
    for s in unique_subj:
        s_mask = sub_subj == s
        if s_mask.sum() < 3:
            continue
        curve = np.nanmean(sub_mat[s_mask], axis=0)
        subj_curves.append(curve)

    if not subj_curves:
        return np.full(len(offsets), np.nan), np.full(len(offsets), np.nan)
    arr = np.array(subj_curves)
    mean = np.nanmean(arr, axis=0)
    se   = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0).clip(1))
    return mean, se


mean_neg, se_neg = subject_mean_trajectory(neg_mask)
mean_other, se_other = subject_mean_trajectory(~neg_mask)

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
fig.suptitle("Gaze Aversion Around Smile Onset (±1.5s, 30fps)\n"
             "(within-subject means ± SE, higher = more averted)",
             fontsize=12, fontweight="bold")

ax.plot(time_s, mean_neg, color="#dc2626", lw=2, label=f"Negative context (n={neg_mask.sum():,})")
ax.fill_between(time_s, mean_neg - se_neg, mean_neg + se_neg, color="#dc2626", alpha=0.15)
n_other = (~neg_mask).sum()
ax.plot(time_s, mean_other, color="#2563eb", lw=2, label=f"Non-negative context (n={n_other:,})")
ax.fill_between(time_s, mean_other - se_other, mean_other + se_other, color="#2563eb", alpha=0.15)

ax.axvline(0, color="#999", lw=1.5, ls="--", label="Smile onset")
ax.set_xlabel("Time relative to smile onset (seconds)", fontsize=10)
ax.set_ylabel("Mean gaze SD level", fontsize=10)
ax.legend(fontsize=9)
ax.set_xlim(time_s[0], time_s[-1])

fig.savefig(FIG_DIR / "gaze_smile_onset.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved gaze_smile_onset.png", flush=True)
print(f"  Negative smiles: {neg_mask.sum():,}, Non-negative: {(~neg_mask).sum():,}")


# ══════════════════════════════════════════════════════════════════════════════
# E. Onset-to-Offset Gaze Shift
# ══════════════════════════════════════════════════════════════════════════════
print("\nAnalysis E: gaze shift during smiles...", flush=True)

shift_df = shift_df[shift_df["narrative_valence"].isin(["negative", "neutral", "positive"])].copy()

# Within-subject mean shift by valence
subj_shifts = (shift_df.groupby(["subject", "narrative_valence"])["gaze_shift"]
               .mean().reset_index())

fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
fig.suptitle("Gaze Shift During Smiles (last third − first third of gaze_x)\n"
             "(positive = moved toward interviewer; within-subject means)",
             fontsize=12, fontweight="bold")

VALENCE_COLORS = {"negative": "#dc2626", "neutral": "#6b7280", "positive": "#16a34a"}
VALENCE_ORDER  = ["negative", "neutral", "positive"]

# ── Panel 1: Violin/box by valence ───────────────────────────────────────────
ax = axes[0]
data_by_val = [subj_shifts[subj_shifts["narrative_valence"] == v]["gaze_shift"].dropna()
               for v in VALENCE_ORDER]

parts = ax.violinplot(data_by_val, positions=[0, 1, 2], showmedians=False,
                      showextrema=False)
for pc, v in zip(parts["bodies"], VALENCE_ORDER):
    pc.set_facecolor(VALENCE_COLORS[v])
    pc.set_alpha(0.3)

bp = ax.boxplot(data_by_val, positions=[0, 1, 2], widths=0.25,
                patch_artist=True, showfliers=False)
for patch, v in zip(bp["boxes"], VALENCE_ORDER):
    patch.set_facecolor(VALENCE_COLORS[v])
    patch.set_alpha(0.5)
for med in bp["medians"]:
    med.set_color("black")

ax.axhline(0, color="#ccc", lw=1, ls="--")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels([v.capitalize() for v in VALENCE_ORDER])
ax.set_xlabel("Narrative Valence")
ax.set_ylabel("Gaze shift (last third − first third)")
ax.set_title("Distribution of Per-Subject Mean Shift", fontweight="bold", fontsize=10)

for i, v in enumerate(VALENCE_ORDER):
    d = data_by_val[i]
    m = d.mean()
    t, p = stats.ttest_1samp(d, 0)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    ax.text(i, ax.get_ylim()[1] * 0.9, f"μ={m:.4f}\n{sig}",
            ha="center", fontsize=8, color=VALENCE_COLORS[v])

# ── Panel 2: Mean shift with SE bars ─────────────────────────────────────────
ax = axes[1]
for i, v in enumerate(VALENCE_ORDER):
    d = data_by_val[i]
    m = d.mean()
    se = d.sem()
    ax.bar(i, m, yerr=se, capsize=5, color=VALENCE_COLORS[v], alpha=0.8, width=0.5)
ax.axhline(0, color="#ccc", lw=1, ls="--")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels([v.capitalize() for v in VALENCE_ORDER])
ax.set_xlabel("Narrative Valence")
ax.set_ylabel("Mean gaze shift ± SE")
ax.set_title("Group Mean (within-subject first)", fontweight="bold", fontsize=10)

fig.savefig(FIG_DIR / "gaze_smile_shift.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved gaze_smile_shift.png", flush=True)

for v in VALENCE_ORDER:
    d = subj_shifts[subj_shifts["narrative_valence"] == v]["gaze_shift"]
    t, p = stats.ttest_1samp(d.dropna(), 0)
    print(f"  {v:10s}: mean shift = {d.mean():.5f}, t={t:.2f}, p={p:.3e}, n={len(d.dropna())}")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Sentences with gaze:     {has_gaze.sum():,}")
print(f"Smile + gaze:            {(has_gaze & df['has_smile']).sum():,}")
print(f"Smile trajectories:      {len(matrix):,}")
print(f"Smile gaze shifts:       {len(shift_df):,}")
print("\nDone.")
