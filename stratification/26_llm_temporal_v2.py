"""
Extended temporal visualisations of LLM-annotated eyegaze features.

Fig 26 – Side-by-side: per-tape aggregate vs per-subject (concatenated tapes).
Fig 27 – Individual examples: 5 selected subjects, full arc across all their tapes.
"""

import json, os, collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d

DATA_DIR = "/Users/marcus/Desktop/usc/VoiceOver/data/llm_annotated_eyegaze"
FIG_DIR  = "/Users/marcus/Desktop/usc/VoiceOver/stratification/figures"

# ── 1. Load & compute two rel_pos columns ─────────────────────────────────────
print("Loading data...", flush=True)
records = []
for fname in sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".json")):
    stem    = fname.replace(".json", "")
    subject = stem.rsplit(".", 1)[0]
    tape    = float(stem.rsplit(".", 1)[1])
    with open(os.path.join(DATA_DIR, fname)) as fh:
        d = json.load(fh)
    sents = d["sentences"]
    if not sents:
        continue
    max_seg = max(s["segment_idx"] for s in sents)
    for s in sents:
        within_tape = s["segment_idx"] / max_seg if max_seg > 0 else 0.0
        records.append({
            "subject":            subject,
            "tape":               tape,
            "segment_idx":        s["segment_idx"],
            "within_tape_pos":    within_tape,          # 0→1 within this single tape
            "memory_type":        s["memory_type"],
            "content_domain":     s["content_domain"],
            "temporal_syntax":    s["temporal_syntax"],
            "narrative_structure":s["narrative_structure"],
            "narrative_valence":  s["narrative_valence"],
            "present_day_valence":s["present_day_valence"],
        })

df = pd.DataFrame(records)
print(f"  {len(df):,} sentences, {df['subject'].nunique():,} subjects", flush=True)

# ── 2. Add per-subject concatenated position ──────────────────────────────────
# For each subject, rank tapes (0,1,2...) and map each sentence to
# concat_pos = (tape_rank + within_tape_pos) / n_tapes
subject_tape_ranks = (
    df[["subject", "tape"]]
    .drop_duplicates()
    .sort_values(["subject", "tape"])
    .assign(tape_rank=lambda x: x.groupby("subject").cumcount())
)
subject_n_tapes = subject_tape_ranks.groupby("subject")["tape"].count().rename("n_tapes")
subject_tape_ranks = subject_tape_ranks.join(subject_n_tapes, on="subject")

df = df.merge(subject_tape_ranks[["subject","tape","tape_rank","n_tapes"]],
              on=["subject","tape"], how="left")
df["concat_pos"] = (df["tape_rank"] + df["within_tape_pos"]) / df["n_tapes"]

# ── 3. Shared helpers ─────────────────────────────────────────────────────────
N_BINS = 40
bin_edges   = np.linspace(0, 1, N_BINS + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

PALETTE = {
    "internal":"#2563eb","external":"#f59e0b",
    "pre-war":"#6d28d9","wartime":"#dc2626","liberation":"#16a34a",
    "post-war":"#0891b2","present-day":"#d97706","other":"#9ca3af",
    "strict_past":"#1e40af","habitual_past":"#7c3aed",
    "present_reflection":"#0369a1","present_narration":"#059669",
    "present_reliving":"#dc2626",
    "orientation":"#1d4ed8","complicating_action":"#dc2626",
    "evaluation":"#d97706","resolution":"#16a34a",
    "positive":"#16a34a","negative":"#dc2626","neutral":"#9ca3af",
}

def temporal_proportions(sub_df, pos_col, col, values, sigma=1.8):
    sub_df = sub_df.copy()
    sub_df["bin"] = pd.cut(sub_df[pos_col], bins=bin_edges,
                           labels=False, include_lowest=True).astype("Int64")
    dummies = pd.get_dummies(sub_df[col]).reindex(columns=values, fill_value=0)
    dummies["subject"] = sub_df["subject"].values
    dummies["bin"]     = sub_df["bin"].values
    dummies = dummies.dropna(subset=["bin"])

    grp   = dummies.groupby(["subject","bin"])[values].sum()
    props = grp.div(grp.sum(axis=1).clip(lower=1e-9), axis=0)

    mean_by_bin = props.groupby("bin").mean().reindex(range(N_BINS))
    se_by_bin   = props.groupby("bin").sem().reindex(range(N_BINS))

    out = {}
    for v in values:
        m = mean_by_bin[v].ffill().bfill().values
        s = se_by_bin[v].fillna(0).values
        out[v] = (bin_centers, gaussian_filter1d(m, sigma=sigma), s)
    return out

def plot_lines(ax, res, title=None, legend=True, ylabel="Proportion", xlabel=True):
    for v, (x, mean, se) in res.items():
        c = PALETTE.get(v, "#888")
        ax.plot(x, mean, color=c, lw=1.8, label=v.replace("_", " "))
        ax.fill_between(x, (mean-se).clip(0), mean+se, color=c, alpha=0.15)
    if title: ax.set_title(title, fontweight="bold", fontsize=10)
    if xlabel: ax.set_xlabel("Normalised Position in Interview", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0)
    ax.tick_params(labelsize=8)
    if legend: ax.legend(fontsize=7, ncol=1)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 26 – Per-tape vs Per-subject side-by-side
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 26: per-tape vs per-subject aggregate...", flush=True)

FEATURES = [
    ("content_domain",
     ["pre-war","wartime","liberation","post-war","present-day","other"],
     "Content Domain"),
    ("temporal_syntax",
     ["strict_past","habitual_past","present_reflection",
      "present_narration","present_reliving"],
     "Temporal Syntax"),
    ("narrative_valence",
     ["positive","negative","neutral"],
     "Narrative Valence"),
    ("narrative_structure",
     ["orientation","complicating_action","evaluation","resolution","other"],
     "Narrative Structure"),
]

# For per-tape we use 'within_tape_pos'; each file is a separate "subject" unit
df_pertape = df.copy()
df_pertape["subject"] = df["subject"] + "_" + df["tape"].astype(str)

fig, axes = plt.subplots(len(FEATURES), 2,
                         figsize=(14, 4 * len(FEATURES)),
                         constrained_layout=True)
fig.suptitle("Per-Tape vs Per-Subject (Concatenated Tapes) Temporal Trends\n"
             "mean ± SE across all subjects, normalised interview position",
             fontsize=13, fontweight="bold")

for row, (col, vals, label) in enumerate(FEATURES):
    # Left: per-tape (each file = independent story)
    res_tape = temporal_proportions(df_pertape, "within_tape_pos", col, vals)
    ax = axes[row, 0]
    plot_lines(ax, res_tape,
               title=f"{label} — Per-Tape" if row == 0 else f"{label}",
               legend=(row == 0))
    if row == 0:
        ax.set_title("Per-Tape (each video independent)", fontweight="bold")

    # Right: per-subject (concatenated across tapes)
    res_subj = temporal_proportions(df, "concat_pos", col, vals)
    ax = axes[row, 1]
    plot_lines(ax, res_subj,
               title=f"{label} — Per-Subject",
               legend=(row == 0))
    if row == 0:
        ax.set_title("Per-Subject (tapes concatenated)", fontweight="bold")

# Add column header boxes
for ax, title in zip(axes[0], ["Per-Tape (each video independent)",
                                "Per-Subject (tapes concatenated)"]):
    ax.set_title(title, fontweight="bold", fontsize=11)

# Add row labels on left spine
for row, (_, _, label) in enumerate(FEATURES):
    axes[row, 0].set_ylabel(f"{label}\nProportion", fontsize=8)

out26 = os.path.join(FIG_DIR, "26_llm_pertape_vs_persubject.png")
fig.savefig(out26, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out26}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 27 – Individual subject examples (5 subjects)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 27: individual subject examples...", flush=True)

# Pick 5 subjects: varied, multi-tape, different domain distributions
EXAMPLE_SUBJECTS = ["27335", "14060", "2684", "12462", "24195"]

# Retrieve name/label from file (use subject ID; could enrich later)
def subject_label(subj):
    n_tapes = int(df[df["subject"]==subj]["tape"].nunique())
    n_sents = int((df["subject"]==subj).sum())
    return f"Subject {subj}\n({n_tapes} tapes, {n_sents:,} sentences)"

DOMAINS_ORDERED = ["pre-war","wartime","liberation","post-war","present-day","other"]

fig = plt.figure(figsize=(15, 4.5 * len(EXAMPLE_SUBJECTS)), constrained_layout=True)
fig.suptitle("Individual Subject Temporal Arcs\n(smoothed proportions across concatenated tapes)",
             fontsize=13, fontweight="bold")

outer = gridspec.GridSpec(len(EXAMPLE_SUBJECTS), 3, figure=fig,
                          hspace=0.5, wspace=0.35)

N_BINS_IND = 30   # fewer bins for individual (less data)

for row_i, subj in enumerate(EXAMPLE_SUBJECTS):
    sub_df = df[df["subject"] == subj].copy()
    bin_e  = np.linspace(0, 1, N_BINS_IND + 1)
    bc     = (bin_e[:-1] + bin_e[1:]) / 2

    def ind_proportions(col, vals, sigma=2.0):
        sub_df["bin"] = pd.cut(sub_df["concat_pos"], bins=bin_e,
                               labels=False, include_lowest=True).astype("Int64")
        dummies = pd.get_dummies(sub_df[col]).reindex(columns=vals, fill_value=0)
        dummies["bin"] = sub_df["bin"].values
        grp = dummies.groupby("bin")[vals].sum().reindex(range(N_BINS_IND))
        totals = grp.sum(axis=1).clip(lower=1e-9)
        props  = grp.div(totals, axis=0).ffill().bfill()
        return {v: gaussian_filter1d(props[v].values, sigma=sigma) for v in vals}

    # ── Stacked domain arc ────────────────────────────────────────────────
    ax0 = fig.add_subplot(outer[row_i, 0])
    dom_curves = ind_proportions("content_domain", DOMAINS_ORDERED)
    stacks = np.array([dom_curves[v] for v in DOMAINS_ORDERED])
    stacks = stacks / stacks.sum(axis=0, keepdims=True).clip(1e-9)
    bottoms = np.zeros(N_BINS_IND)
    for v, row in zip(DOMAINS_ORDERED, stacks):
        ax0.fill_between(bc, bottoms, bottoms + row,
                         color=PALETTE[v], alpha=0.88, label=v)
        bottoms += row
    ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)
    ax0.set_ylabel(subject_label(subj), fontsize=7.5, labelpad=4)
    ax0.tick_params(labelsize=7)
    if row_i == 0:
        ax0.set_title("Content Domain Arc", fontweight="bold", fontsize=9)
        ax0.legend(fontsize=6, loc="lower right", ncol=2)
    if row_i == len(EXAMPLE_SUBJECTS) - 1:
        ax0.set_xlabel("Normalised Position", fontsize=8)

    # ── Temporal syntax lines ─────────────────────────────────────────────
    ax1 = fig.add_subplot(outer[row_i, 1])
    SYNTAXES = ["strict_past","habitual_past","present_reflection",
                "present_narration","present_reliving"]
    syn_curves = ind_proportions("temporal_syntax", SYNTAXES)
    for v, curve in syn_curves.items():
        ax1.plot(bc, curve, color=PALETTE[v], lw=1.8,
                 label=v.replace("_"," "))
    ax1.set_xlim(0, 1); ax1.set_ylim(0)
    ax1.tick_params(labelsize=7)
    if row_i == 0:
        ax1.set_title("Temporal Syntax", fontweight="bold", fontsize=9)
        ax1.legend(fontsize=6)
    if row_i == len(EXAMPLE_SUBJECTS) - 1:
        ax1.set_xlabel("Normalised Position", fontsize=8)

    # ── Valence (narrative + present-day) ─────────────────────────────────
    ax2 = fig.add_subplot(outer[row_i, 2])
    VALS = ["positive", "negative", "neutral"]
    narr_curves = ind_proportions("narrative_valence", VALS)
    pd_curves   = ind_proportions("present_day_valence", VALS)
    for v in VALS:
        c = PALETTE[v]
        ax2.plot(bc, narr_curves[v], color=c, lw=1.8,
                 label=f"{v} (narrative)")
        ax2.plot(bc, pd_curves[v],   color=c, lw=1.2, ls="--",
                 alpha=0.6)
    ax2.set_xlim(0, 1); ax2.set_ylim(0)
    ax2.tick_params(labelsize=7)
    if row_i == 0:
        ax2.set_title("Valence (solid=narrative, dashed=present-day)",
                      fontweight="bold", fontsize=9)
        ax2.legend(fontsize=6)
    if row_i == len(EXAMPLE_SUBJECTS) - 1:
        ax2.set_xlabel("Normalised Position", fontsize=8)

    # Tape boundary markers (faint vertical lines)
    n_tapes = int(sub_df["tape"].nunique())
    if n_tapes > 1:
        for ti in range(1, n_tapes):
            boundary = ti / n_tapes
            for ax in [ax0, ax1, ax2]:
                ax.axvline(boundary, color="#aaa", lw=0.7, ls=":", alpha=0.7)

out27 = os.path.join(FIG_DIR, "27_llm_individual_examples.png")
fig.savefig(out27, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out27}", flush=True)

print("Done.")
