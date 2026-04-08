"""
Duchenne smile analyses × narrative context — continuous AU6 approach.

AU6_r (orbicularis oculi / cheek raiser) is treated as a continuous variable
throughout. No binary threshold is imposed.

Figures:
  D1  duchenne_by_narrative.png   — Mean AU6 + distribution by valence/domain/structure/syntax
  D2  duchenne_temporal.png       — AU6 intensity over normalised interview position
  D3  duchenne_perievent.png      — Peri-event trajectories by AU6 quartile (dose-response)
  D4  duchenne_subject_trait.png  — Subject mean AU6 correlated with narrative measures
  D5  au6_au12_context.png        — AU6 vs AU12 scatter coloured by narrative context
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d
from scipy import stats
from pathlib import Path

BASE      = Path("/Users/marcus/Desktop/usc/VoiceOver")
SMILE_PKL = BASE / "smile_variable_effects/smile_au_table.pkl"
SENT_PKL  = BASE / "smile_variable_effects/sentence_table.pkl"
FIG_DIR   = BASE / "smile_variable_effects/figures"
FIG_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# Load & clean
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data...", flush=True)
sm   = pd.read_pickle(SMILE_PKL)
sent = pd.read_pickle(SENT_PKL)

sm["video_id"] = sm["video_id"].astype(str)

sm_valid = sm[
    sm["n_frames"].gt(0) &
    sm["narrative_valence"].notna() &
    sm["au_AU06_r"].notna() &
    sm["au_AU12_r"].notna()
].copy()

print(f"Smiles with AU + sentence data: {len(sm_valid):,}")
print(f"AU6 range: {sm_valid['au_AU06_r'].min():.3f} – {sm_valid['au_AU06_r'].max():.3f}  "
      f"mean={sm_valid['au_AU06_r'].mean():.3f}  std={sm_valid['au_AU06_r'].std():.3f}")

# AU6 quartile label (used in peri-event dose-response)
sm_valid["au6_q"] = pd.qcut(sm_valid["au_AU06_r"], 4,
                             labels=["Q1 (low)","Q2","Q3","Q4 (high)"])

VALENCE_ORDER   = ["negative","neutral","positive"]
DOMAIN_ORDER    = ["pre-war","wartime","liberation","post-war","present-day","other"]
STRUCTURE_ORDER = ["orientation","complicating_action","evaluation","resolution","other"]
SYNTAX_ORDER    = ["strict_past","habitual_past","present_reflection",
                   "present_narration","present_reliving"]

Q_COLORS = ["#9ca3af","#60a5fa","#f59e0b","#dc2626"]   # Q1→Q4
DOMAIN_COLORS = {
    "pre-war":"#6d28d9","wartime":"#dc2626","liberation":"#16a34a",
    "post-war":"#0891b2","present-day":"#d97706","other":"#9ca3af",
}


# ══════════════════════════════════════════════════════════════════════════════
# Helper: within-subject mean AU6 per group level → group-level CI
# ══════════════════════════════════════════════════════════════════════════════
def au6_by_group(df, col, order, min_n=20):
    rows = []
    for val in order:
        sub = df[df[col] == val]
        if len(sub) < min_n:
            rows.append(dict(val=val, mean=np.nan, se=np.nan, n=0))
            continue
        subj = sub.groupby("subject")["au_AU06_r"].mean().dropna()
        rows.append(dict(val=val, mean=subj.mean(), se=subj.sem(), n=len(subj)))
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE D1: Mean AU6 by narrative context (4-panel, violin + mean)
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure D1: AU6 by narrative context...", flush=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
fig.suptitle("AU6 (Cheek Raiser) Intensity During Smiles × Narrative Context\n"
             "violin = full distribution; dot = within-subject mean ± SE",
             fontsize=13, fontweight="bold")

CONTEXT_PANELS = [
    ("narrative_valence",   VALENCE_ORDER,   "Narrative Valence",   axes[0,0]),
    ("content_domain",      DOMAIN_ORDER,    "Content Domain",      axes[0,1]),
    ("narrative_structure", STRUCTURE_ORDER, "Narrative Structure", axes[1,0]),
    ("temporal_syntax",     SYNTAX_ORDER,    "Temporal Syntax",     axes[1,1]),
]

for col, order, label, ax in CONTEXT_PANELS:
    present = [v for v in order if v in sm_valid[col].values]
    data_by_val = [sm_valid.loc[sm_valid[col]==v, "au_AU06_r"].values for v in present]
    x = np.arange(len(present))

    # Violin
    parts = ax.violinplot(data_by_val, positions=x, showmedians=False,
                          showextrema=False, widths=0.6)
    for pc in parts["bodies"]:
        pc.set_facecolor("#bfdbfe")
        pc.set_edgecolor("#1d4ed8")
        pc.set_alpha(0.5)

    # Within-subject means ± SE
    stats_df = au6_by_group(sm_valid, col, present)
    ax.errorbar(x, stats_df["mean"].values, yerr=stats_df["se"].values,
                fmt="o", color="#1d4ed8", ms=7, capsize=4, lw=1.8, zorder=5)

    # Grand mean reference line
    grand = sm_valid.groupby("subject")["au_AU06_r"].mean().mean()
    ax.axhline(grand, color="#dc2626", lw=1.2, ls="--", alpha=0.7,
               label=f"Grand mean ({grand:.3f})")

    # Annotate pairwise contrast: first vs each
    baseline_m = stats_df["mean"].iloc[0]
    for xi, row in stats_df.iterrows():
        if xi == 0 or np.isnan(row["mean"]) or row["n"] < 20:
            continue
        delta = row["mean"] - baseline_m
        ax.annotate(f"Δ{delta:+.3f}", xy=(xi, row["mean"]),
                    xytext=(xi + 0.15, row["mean"] + 0.015),
                    fontsize=6.5, color="#1e40af")

    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("_"," ") for v in present],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("AU6_r intensity")
    ax.set_title(label, fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)

out = FIG_DIR / "D1_duchenne_by_narrative.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {out.name}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE D2: AU6 intensity over interview position
# ══════════════════════════════════════════════════════════════════════════════
print("Figure D2: AU6 over interview position...", flush=True)

N_BINS = 40
bin_edges   = np.linspace(0, 1, N_BINS + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
SIGMA = 2.0

sm_valid["bin"] = pd.cut(sm_valid["concat_pos"], bins=bin_edges,
                         labels=False, include_lowest=True).astype("Int64")

# Per-subject mean AU6 per bin
pivot_au6 = (sm_valid.dropna(subset=["bin"])
             .groupby(["subject","bin"])["au_AU06_r"]
             .mean()
             .unstack(fill_value=np.nan)
             .reindex(columns=range(N_BINS)))
mean_au6  = pivot_au6.mean()
se_au6    = pivot_au6.sem()
mean_au6_s = gaussian_filter1d(mean_au6.values,  sigma=SIGMA)
se_au6_s   = gaussian_filter1d(se_au6.values,    sigma=SIGMA)

# Also: AU12 for comparison
pivot_au12 = (sm_valid.dropna(subset=["bin"])
              .groupby(["subject","bin"])["au_AU12_r"]
              .mean()
              .unstack(fill_value=np.nan)
              .reindex(columns=range(N_BINS)))
mean_au12_s = gaussian_filter1d(pivot_au12.mean().values, sigma=SIGMA)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True,
                         gridspec_kw={"height_ratios": [1.5, 1]})
fig.suptitle("AU6 (Duchenne Marker) and AU12 (Lip Corner) Intensity Over Interview\n"
             "(per-subject mean AU intensity during smile windows, normalised position)",
             fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(bin_centers, mean_au6_s,  color="#1d4ed8", lw=2.5, label="AU6 (cheek raiser)")
ax.fill_between(bin_centers,
                (mean_au6_s - se_au6_s).clip(0),
                mean_au6_s + se_au6_s,
                color="#1d4ed8", alpha=0.18)
ax.plot(bin_centers, mean_au12_s, color="#dc2626", lw=2, ls="--",
        alpha=0.8, label="AU12 (lip corner) — reference")
ax.set_ylabel("Mean AU intensity during smiles")
ax.set_xlim(0, 1)
ax.legend(fontsize=9)
ax.tick_params(labelbottom=False)
ax.set_title("Higher AU6 = More Duchenne (felt) smiling", fontsize=9, fontstyle="italic", loc="right")

# Ratio: AU6/AU12
ratio = mean_au6_s / np.where(mean_au12_s > 0, mean_au12_s, np.nan)
ax2 = axes[1]
ax2.plot(bin_centers, gaussian_filter1d(ratio, sigma=1.5),
         color="#7c3aed", lw=2, label="AU6/AU12 ratio (Duchenne index)")
ax2.axhline(np.nanmean(ratio), color="#ccc", lw=1, ls="--", alpha=0.7,
            label=f"Mean ratio ({np.nanmean(ratio):.3f})")
ax2.set_xlabel("Normalised Position in Interview")
ax2.set_ylabel("AU6 / AU12 ratio")
ax2.set_xlim(0, 1)
ax2.legend(fontsize=9)

out = FIG_DIR / "D2_duchenne_temporal.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {out.name}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE D3: Peri-event narrative valence — AU6 quartile dose-response
# ══════════════════════════════════════════════════════════════════════════════
print("Figure D3: peri-event dose-response by AU6 quartile...", flush=True)

K = 3
VALENCE_MAP = {"negative": -1, "neutral": 0, "positive": 1, "mixed": np.nan}
sent["narr_val_num"] = sent["narrative_valence"].map(VALENCE_MAP)
sent["video_id"] = sent["video_id"].astype(str)
for lag in range(-K, K+1):
    sent[f"nvn_L{lag}"] = sent.groupby("subject")["narr_val_num"].shift(-lag)
lag_cols = [f"nvn_L{l}" for l in range(-K, K+1)]

# Build lookup: video_id → sorted sentence rows
sent_valid = sent[sent["first_word_ms"].notna() & sent["last_word_ms"].notna()].copy()
by_vid = {vid: g.reset_index(drop=True)
          for vid, g in sent_valid.groupby("video_id")}

def get_trajectories_for_smiles(smile_subset):
    """For each smile in subset, find the best-overlap sentence, return lag columns."""
    rows = []
    for _, row in smile_subset.iterrows():
        vid  = row["video_id"]
        t0ms = row["start_ts"] * 1000
        t1ms = row["end_ts"]   * 1000
        if vid not in by_vid:
            continue
        sg  = by_vid[vid]
        fw  = sg["first_word_ms"].values
        lw  = sg["last_word_ms"].values
        ovlp_mask = (fw < t1ms) & (lw > t0ms)
        if not ovlp_mask.any():
            continue
        ovlp_len = (np.minimum(t1ms, lw[ovlp_mask])
                    - np.maximum(t0ms, fw[ovlp_mask]))
        best = sg.iloc[np.where(ovlp_mask)[0][np.argmax(ovlp_len)]]
        rows.append({"subject": best["subject"],
                     **{c: best[c] for c in lag_cols}})
    return pd.DataFrame(rows)

def summarise(df):
    subj = df.groupby("subject")[lag_cols].mean()
    return subj.mean().values, subj.sem().values

# Restrict to smiles during negative-valence sentences
sm_neg = sm_valid[sm_valid["narrative_valence"] == "negative"].copy()

# Baseline (no smile, negative sentence)
sent_neg_no_smile = sent[
    (sent["narrative_valence"] == "negative") &
    ~sent["has_smile"] &
    sent["narr_val_num"].notna()
][["subject"] + lag_cols]

lags = list(range(-K, K+1))
fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
fig.suptitle("Peri-Event Narrative Valence × AU6 Intensity (Dose-Response)\n"
             "Smiles during negative-valence content; Q1=low AU6 → Q4=high AU6",
             fontsize=13, fontweight="bold")

for ax_i, (ax, title, subset) in enumerate(zip(
        axes,
        ["All Smiles During Negative", "Negative-Content Smiles Only (zoomed)"],
        [sm_neg, sm_neg])):

    # Baseline
    bl_mean, bl_se = summarise(sent_neg_no_smile)
    ax.plot(lags, bl_mean, color="#374151", lw=2.2, ls="--",
            label=f"Baseline (no smile, n={len(sent_neg_no_smile):,})", zorder=5)
    ax.fill_between(lags, bl_mean-bl_se, bl_mean+bl_se, color="#374151", alpha=0.12)

    # AU6 quartiles
    for q, qcolor in zip(["Q1 (low)","Q2","Q3","Q4 (high)"], Q_COLORS):
        qsub = subset[subset["au6_q"] == q]
        if len(qsub) < 50:
            continue
        print(f"  {q}: n={len(qsub)} smiles", flush=True)
        traj = get_trajectories_for_smiles(qsub)
        if len(traj) < 10:
            continue
        mean, se = summarise(traj)
        ax.plot(lags, mean, color=qcolor, lw=2,
                label=f"{q} (n={len(qsub):,})")
        ax.fill_between(lags, mean-se, mean+se, color=qcolor, alpha=0.15)

    ax.axvline(0, color="#ddd", lw=1, ls=":")
    ax.set_xlabel("Lag (sentences from smile sentence)")
    ax.set_ylabel("Narrative Valence (−1=neg, 0=neutral, +1=pos)")
    ax.set_xticks(lags)
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")

    if ax_i == 1:
        # Zoom in to the post-smile window
        ax.set_xlim(-1, K)

out = FIG_DIR / "D3_duchenne_perievent.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {out.name}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE D4: Subject-level mean AU6 correlated with narrative measures
# ══════════════════════════════════════════════════════════════════════════════
print("Figure D4: subject-level AU6 correlations...", flush=True)

subj_au6 = (sm_valid.groupby("subject")
            .agg(mean_au6=("au_AU06_r","mean"),
                 mean_au12=("au_AU12_r","mean"),
                 n_smiles=("au_AU06_r","count"))
            .reset_index())
subj_au6 = subj_au6[subj_au6["n_smiles"] >= 10]
# Only compute ratio where AU12 is meaningfully active (avoid divide-by-near-zero)
subj_au6["au6_au12_ratio"] = np.where(
    subj_au6["mean_au12"] > 0.2,
    subj_au6["mean_au6"] / subj_au6["mean_au12"],
    np.nan
)

subj_narr = sent.groupby("subject").agg(
    mean_val        =("narr_val_num",        "mean"),
    pct_negative    =("narrative_valence",   lambda x: (x=="negative").mean()),
    pct_wartime     =("content_domain",      lambda x: (x=="wartime").mean()),
    pct_liberation  =("content_domain",      lambda x: (x=="liberation").mean()),
    pct_evaluation  =("narrative_structure", lambda x: (x=="evaluation").mean()),
    pct_pres_reflect=("temporal_syntax",     lambda x: (x=="present_reflection").mean()),
).reset_index()

trait_df = subj_au6.merge(subj_narr, on="subject")

SCATTER_PAIRS = [
    ("pct_wartime",      "% Wartime Content"),
    ("pct_liberation",   "% Liberation Content"),
    ("pct_negative",     "% Negative Narrative"),
    ("pct_evaluation",   "% Evaluation Structure"),
    ("pct_pres_reflect", "% Present Reflection"),
    ("mean_val",         "Mean Narrative Valence"),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
fig.suptitle("Subject Mean AU6 Intensity vs Narrative Profile\n"
             "(each dot = one subject; regression line + r, p)",
             fontsize=13, fontweight="bold")

for ax, (xcol, xlabel) in zip(axes.flat, SCATTER_PAIRS):
    x = trait_df[xcol].values
    y = trait_df["au6_au12_ratio"].values   # use ratio: normalises for baseline AU12
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 20:
        continue

    ax.scatter(x[mask], y[mask], s=12, alpha=0.35, color="#1d4ed8")
    slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])
    xline = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(xline, slope*xline + intercept, color="#dc2626", lw=2)

    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."
    ax.set_title(f"r={r:.3f}  p={p:.4f}  {sig}", fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("AU6/AU12 ratio (Duchenne index)", fontsize=8)
    ax.text(0.05, 0.92, f"n={mask.sum()}", transform=ax.transAxes,
            fontsize=8, color="#6b7280")

out = FIG_DIR / "D4_duchenne_subject_trait.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {out.name}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE D5: AU6 vs AU12 scatter coloured by content domain
# ══════════════════════════════════════════════════════════════════════════════
print("Figure D5: AU6 vs AU12 by context...", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
fig.suptitle("AU6 vs AU12 During Smiles: Intensity Profile by Narrative Context\n"
             "Dots above the diagonal have proportionally more AU6 (more Duchenne)",
             fontsize=13, fontweight="bold")

# Left: coloured by content domain
ax = axes[0]
for dom in DOMAIN_ORDER:
    sub = sm_valid[sm_valid["content_domain"] == dom]
    if len(sub) < 20:
        continue
    ax.scatter(sub["au_AU12_r"], sub["au_AU06_r"],
               s=4, alpha=0.2, color=DOMAIN_COLORS[dom], label=dom)

diag = np.linspace(0, sm_valid["au_AU12_r"].quantile(0.99), 100)
ax.plot(diag, diag, color="#9ca3af", lw=1, ls="--", alpha=0.5, label="AU6=AU12")
ax.set_xlabel("AU12_r (lip corner — smile amplitude)")
ax.set_ylabel("AU6_r (cheek raiser — Duchenne marker)")
ax.set_title("By Content Domain", fontweight="bold", fontsize=10)
legend = ax.legend(fontsize=7, markerscale=3)
for lh in legend.legend_handles:
    try: lh.set_alpha(1)
    except Exception: pass

# Right: mean AU6/AU12 per domain with error bars (cleaner summary)
ax2 = axes[1]
dom_stats = []
for dom in DOMAIN_ORDER:
    sub = sm_valid[(sm_valid["content_domain"] == dom) & (sm_valid["au_AU12_r"] > 0.2)]
    if len(sub) < 20:
        continue
    subj = sub.groupby("subject").apply(
        lambda g: g["au_AU06_r"].mean() / g["au_AU12_r"].mean()
    ).dropna()
    dom_stats.append(dict(domain=dom, mean=subj.mean(), se=subj.sem(),
                          n=len(subj), color=DOMAIN_COLORS[dom]))
dom_stats_df = pd.DataFrame(dom_stats)

x = np.arange(len(dom_stats_df))
bars = ax2.bar(x, dom_stats_df["mean"], color=dom_stats_df["color"],
               alpha=0.8, edgecolor="white", width=0.6)
ax2.errorbar(x, dom_stats_df["mean"], yerr=dom_stats_df["se"],
             fmt="none", color="#1f2937", capsize=4, lw=1.5)
grand_ratio = (sm_valid[sm_valid["au_AU12_r"] > 0.2]
               .groupby("subject")
               .apply(lambda g: g["au_AU06_r"].mean() / g["au_AU12_r"].mean())
               .mean())
ax2.axhline(grand_ratio, color="#6b7280", lw=1.2, ls="--",
            label=f"Grand mean ({grand_ratio:.3f})")
ax2.set_xticks(x)
ax2.set_xticklabels([r["domain"].replace("-"," ") for _, r in dom_stats_df.iterrows()],
                    rotation=25, ha="right", fontsize=9)
ax2.set_ylabel("Mean AU6/AU12 ratio (per-subject)")
ax2.set_title("Duchenne Index by Content Domain\n(higher = more genuine smiling)",
              fontweight="bold", fontsize=10)
ax2.legend(fontsize=9)

for xi, row in dom_stats_df.iterrows():
    ax2.text(xi, row["mean"] + row["se"] + 0.003,
             f"n={row['n']}", ha="center", fontsize=7, color="#374151")

out = FIG_DIR / "D5_au6_au12_context.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {out.name}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Summary stats
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print("SUMMARY — AU6 Continuous Analysis", flush=True)
print("="*60, flush=True)
print(f"Smiles analysed: {len(sm_valid):,}")

print(f"\nMean AU6 by narrative valence (within-subject):")
for v in VALENCE_ORDER:
    sub = sm_valid[sm_valid["narrative_valence"]==v]
    r = sub.groupby("subject")["au_AU06_r"].mean().dropna()
    print(f"  {v:10s}: {r.mean():.4f} ± {r.sem():.4f}  (n={len(r)})")

print(f"\nMean AU6/AU12 ratio by content domain (AU12>0.2 only):")
for dom in DOMAIN_ORDER:
    sub = sm_valid[(sm_valid["content_domain"]==dom) & (sm_valid["au_AU12_r"] > 0.2)]
    r = sub.groupby("subject").apply(
        lambda g: g["au_AU06_r"].mean() / g["au_AU12_r"].mean()
    ).dropna()
    if len(r):
        print(f"  {dom:15s}: {r.mean():.4f} ± {r.sem():.4f}  (n={len(r)})")

print(f"\nSubject-level correlations (AU6/AU12 ratio vs narrative):")
for xcol, xlabel in SCATTER_PAIRS:
    x = trait_df[xcol].values
    y = trait_df["au6_au12_ratio"].values
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 20: continue
    r, p = stats.pearsonr(x[mask], y[mask])
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."
    print(f"  {xlabel:30s}: r={r:+.3f}  p={p:.4f}  {sig}")

print("\nDone.", flush=True)
