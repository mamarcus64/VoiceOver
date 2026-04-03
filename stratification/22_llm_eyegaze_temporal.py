"""
Exploratory visualizations of LLM-annotated eyegaze features over time.
Shows within-subject temporal trends (normalized position 0→1 through interview).
"""

import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

DATA_DIR = "/Users/marcus/Desktop/usc/VoiceOver/data/llm_annotated_eyegaze"
FIG_DIR  = "/Users/marcus/Desktop/usc/VoiceOver/stratification/figures"
N_BINS   = 40

# ── 1. Load all data ──────────────────────────────────────────────────────────
print("Loading data...", flush=True)
records = []
for fname in sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".json")):
    stem = fname.replace(".json", "")
    parts = stem.rsplit(".", 1)
    subject_id = parts[0]
    with open(os.path.join(DATA_DIR, fname)) as fh:
        d = json.load(fh)
    sents = d["sentences"]
    if not sents:
        continue
    max_seg = max(s["segment_idx"] for s in sents)
    for s in sents:
        rel_pos = s["segment_idx"] / max_seg if max_seg > 0 else 0.0
        records.append((
            subject_id,
            rel_pos,
            s["memory_type"],
            s["content_domain"],
            s["temporal_syntax"],
            s["narrative_structure"],
            s["narrative_valence"],
            s["present_day_valence"],
        ))

df = pd.DataFrame(records, columns=[
    "subject", "rel_pos", "memory_type", "content_domain",
    "temporal_syntax", "narrative_structure", "narrative_valence", "present_day_valence"
])
print(f"  {len(df):,} sentences, {df['subject'].nunique():,} subjects", flush=True)

# ── 2. Assign time bins ───────────────────────────────────────────────────────
bin_edges   = np.linspace(0, 1, N_BINS + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
df["bin"] = pd.cut(df["rel_pos"], bins=bin_edges, labels=False, include_lowest=True).astype("Int64")

# ── 3. Vectorized temporal_proportions ───────────────────────────────────────
def temporal_proportions(df, col, values, sigma=1.8):
    """
    For each subject×bin, compute the fraction of each value.
    Average across subjects, return smoothed mean ± SE.
    """
    # one-hot encode
    dummies = pd.get_dummies(df[col]).reindex(columns=values, fill_value=0)
    dummies["subject"] = df["subject"].values
    dummies["bin"]     = df["bin"].values
    dummies = dummies.dropna(subset=["bin"])

    # sum per subject×bin, then normalise to proportion
    grp = dummies.groupby(["subject", "bin"])[values].sum()
    row_totals = grp.sum(axis=1)
    props = grp.div(row_totals.clip(lower=1e-9), axis=0)  # proportions per subject×bin

    # average across subjects for each bin
    mean_by_bin = props.groupby("bin").mean().reindex(range(N_BINS))
    se_by_bin   = props.groupby("bin").sem().reindex(range(N_BINS))

    results = {}
    for v in values:
        mean_v = mean_by_bin[v].ffill().bfill().values
        se_v   = se_by_bin[v].fillna(0).values
        results[v] = (bin_centers, gaussian_filter1d(mean_v, sigma=sigma), se_v)
    return results

# ── 4. Colours ────────────────────────────────────────────────────────────────
PALETTE = {
    "internal":           "#2563eb",
    "external":           "#f59e0b",
    "pre-war":            "#6d28d9",
    "wartime":            "#dc2626",
    "liberation":         "#16a34a",
    "post-war":           "#0891b2",
    "present-day":        "#d97706",
    "other":              "#9ca3af",
    "strict_past":        "#1e40af",
    "habitual_past":      "#7c3aed",
    "present_reflection": "#0369a1",
    "present_narration":  "#059669",
    "present_reliving":   "#dc2626",
    "orientation":        "#1d4ed8",
    "complicating_action":"#dc2626",
    "evaluation":         "#d97706",
    "resolution":         "#16a34a",
    "positive":           "#16a34a",
    "negative":           "#dc2626",
    "neutral":            "#9ca3af",
}

def plot_lines(ax, res, title, ylabel="Proportion"):
    for v, (x, mean, se) in res.items():
        c = PALETTE.get(v, "#888")
        ax.plot(x, mean, color=c, lw=2, label=v.replace("_", " "))
        ax.fill_between(x, (mean - se).clip(0), mean + se, color=c, alpha=0.18)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Normalised Position in Interview")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0)
    ax.legend(fontsize=8)

# ── 5. Figure 1: 4-panel overview ────────────────────────────────────────────
print("Computing fig 1...", flush=True)
fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
fig.suptitle(
    "Within-Subject Temporal Trends in LLM-Annotated Features\n"
    "(mean ± SE across subjects, normalised interview position)",
    fontsize=13, fontweight="bold"
)

plot_lines(axes[0, 0],
    temporal_proportions(df, "memory_type", ["internal", "external"]),
    "Memory Type")
axes[0, 0].axhline(0.5, color="#ccc", lw=0.8, ls="--")

plot_lines(axes[0, 1],
    temporal_proportions(df, "content_domain",
        ["pre-war","wartime","liberation","post-war","present-day","other"]),
    "Content Domain")

plot_lines(axes[1, 0],
    temporal_proportions(df, "temporal_syntax",
        ["strict_past","habitual_past","present_reflection",
         "present_narration","present_reliving"]),
    "Temporal Syntax")

plot_lines(axes[1, 1],
    temporal_proportions(df, "narrative_valence", ["positive","negative","neutral"]),
    "Narrative Valence")

out1 = os.path.join(FIG_DIR, "22_llm_temporal_overview.png")
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out1}", flush=True)

# ── 6. Figure 2: Narrative structure + present-day valence ───────────────────
print("Computing fig 2...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
fig.suptitle(
    "Within-Subject Temporal Trends — Narrative Structure & Present-Day Valence",
    fontsize=13, fontweight="bold"
)

plot_lines(axes[0],
    temporal_proportions(df, "narrative_structure",
        ["orientation","complicating_action","evaluation","resolution","other"]),
    "Narrative Structure")

plot_lines(axes[1],
    temporal_proportions(df, "present_day_valence", ["positive","negative","neutral"]),
    "Present-Day Valence")

out2 = os.path.join(FIG_DIR, "23_llm_temporal_structure_valence.png")
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out2}", flush=True)

# ── 7. Figure 3: Stacked area — content domain arc ───────────────────────────
print("Computing fig 3...", flush=True)
domains_ordered = ["pre-war","wartime","liberation","post-war","present-day","other"]
res = temporal_proportions(df, "content_domain", domains_ordered, sigma=2.5)

fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
fig.suptitle(
    "Content Domain Arc Through the Interview\n(mean proportion, stacked, smoothed)",
    fontsize=13, fontweight="bold"
)
xs      = res[domains_ordered[0]][0]
stacks  = np.array([res[v][1] for v in domains_ordered])
stacks  = stacks / stacks.sum(axis=0, keepdims=True).clip(1e-9)
bottoms = np.zeros(len(xs))
for v, row in zip(domains_ordered, stacks):
    ax.fill_between(xs, bottoms, bottoms + row, color=PALETTE[v], alpha=0.85, label=v)
    bottoms += row
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Normalised Position in Interview", fontsize=11)
ax.set_ylabel("Proportion", fontsize=11)
ax.legend(loc="upper right", fontsize=9, ncol=2)

out3 = os.path.join(FIG_DIR, "24_llm_domain_arc_stacked.png")
fig.savefig(out3, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out3}", flush=True)

# ── 8. Figure 4: Heatmap grid ─────────────────────────────────────────────────
print("Computing fig 4...", flush=True)
fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
fig.suptitle("Feature Proportion Heatmaps Over Normalised Interview Time",
             fontsize=13, fontweight="bold")

def heatmap(ax, df, col, values, title):
    res = temporal_proportions(df, col, values, sigma=2)
    mat = np.array([res[v][1] for v in values])
    im  = ax.imshow(mat, aspect="auto", interpolation="bilinear",
                    extent=[0, 1, -0.5, len(values) - 0.5],
                    origin="lower", cmap="YlOrRd", vmin=0)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels([v.replace("_", " ") for v in values], fontsize=8)
    ax.set_xlabel("Normalised Position")
    ax.set_title(title, fontweight="bold")
    return im

heatmap(axes[0, 0], df, "content_domain",
        ["pre-war","wartime","liberation","post-war","present-day","other"],
        "Content Domain")
heatmap(axes[0, 1], df, "temporal_syntax",
        ["strict_past","habitual_past","present_reflection",
         "present_narration","present_reliving"],
        "Temporal Syntax")
heatmap(axes[1, 0], df, "narrative_structure",
        ["orientation","complicating_action","evaluation","resolution","other"],
        "Narrative Structure")
im = heatmap(axes[1, 1], df, "narrative_valence",
             ["positive","negative","neutral"], "Narrative Valence")
fig.colorbar(im, ax=axes[1, 1], label="Proportion")

out4 = os.path.join(FIG_DIR, "25_llm_feature_heatmaps.png")
fig.savefig(out4, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out4}", flush=True)

# ── 9. Summary stats ──────────────────────────────────────────────────────────
print("\n── Overall feature distributions ──")
for col in ["memory_type","content_domain","temporal_syntax",
            "narrative_structure","narrative_valence","present_day_valence"]:
    print(f"\n{col}:")
    print(df[col].value_counts(normalize=True).mul(100).round(1).to_string())
print("\nDone.")
