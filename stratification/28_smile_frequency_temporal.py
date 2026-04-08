"""
Fig 28 – Smile frequency over normalised interview position (per-subject).

Panel layout:
  Row 0: Smile rate (smiles/min) – mean ± SE across subjects
  Row 1: Content domain stacked area (for interpretive context)
  Row 2: Narrative valence lines (for interpretive context)
"""

import json, os, collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

DATA_DIR      = Path("/Users/marcus/Desktop/usc/VoiceOver/data")
EYEGAZE_DIR   = DATA_DIR / "llm_annotated_eyegaze"
TRANSCRIPT_DIR = DATA_DIR / "transcripts_llm"
SMILES_FILE   = DATA_DIR / "detected_smiles.json"
FIG_DIR       = Path("/Users/marcus/Desktop/usc/VoiceOver/stratification/figures")

N_BINS   = 40
SIGMA    = 2.0
bin_edges   = np.linspace(0, 1, N_BINS + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_width   = 1.0 / N_BINS  # fraction of total duration per bin

# ── 1. Build per-video metadata from eyegaze files ───────────────────────────
print("Scanning eyegaze files for video metadata...", flush=True)
video_meta = {}   # video_id → {subject, tape, speech_start_ms, speech_end_ms}

for fname in sorted(f for f in os.listdir(EYEGAZE_DIR) if f.endswith(".json")):
    vid  = fname.replace(".json", "")
    stem = vid.rsplit(".", 1)
    subject, tape = stem[0], float(stem[1])

    with open(EYEGAZE_DIR / fname) as fh:
        d = json.load(fh)

    ms_vals = [s["first_word_ms"] for s in d["sentences"] if s.get("first_word_ms") is not None] + \
              [s["last_word_ms"]  for s in d["sentences"] if s.get("last_word_ms")  is not None]
    if not ms_vals:
        continue

    video_meta[vid] = {
        "subject":        subject,
        "tape":           tape,
        "speech_start_ms": min(ms_vals),
        "speech_end_ms":   max(ms_vals),
    }

print(f"  {len(video_meta)} videos with metadata", flush=True)

# ── 2. Build per-subject tape rank / n_tapes ─────────────────────────────────
by_subject = collections.defaultdict(list)
for vid, meta in video_meta.items():
    by_subject[meta["subject"]].append((meta["tape"], vid))

tape_rank  = {}   # vid → rank (0-indexed)
n_tapes_map = {}  # vid → n_tapes
for subj, tape_vids in by_subject.items():
    tape_vids_sorted = sorted(tape_vids)
    n = len(tape_vids_sorted)
    for rank, (_, vid) in enumerate(tape_vids_sorted):
        tape_rank[vid]   = rank
        n_tapes_map[vid] = n

# ── 3. Load smiles and compute concat_pos ────────────────────────────────────
print("Processing smiles...", flush=True)
raw = json.load(open(SMILES_FILE))
smile_list = raw["smiles"]

smile_records = []
for sm in smile_list:
    vid = sm["video_id"]
    if vid not in video_meta:
        continue
    meta  = video_meta[vid]
    start = sm["start_ts"] * 1000   # → ms
    end   = sm["end_ts"]   * 1000
    dur   = meta["speech_end_ms"] - meta["speech_start_ms"]
    if dur <= 0:
        continue

    within_tape = (start - meta["speech_start_ms"]) / dur
    within_tape = np.clip(within_tape, 0.0, 1.0)

    rank    = tape_rank.get(vid)
    n_tapes = n_tapes_map.get(vid)
    if rank is None:
        continue

    concat_pos = (rank + within_tape) / n_tapes

    smile_records.append({
        "subject":    meta["subject"],
        "video_id":   vid,
        "concat_pos": concat_pos,
        "duration_ms": end - start,
        "score":       sm["score"],
    })

smile_df = pd.DataFrame(smile_records)
print(f"  {len(smile_df):,} smiles across {smile_df['subject'].nunique():,} subjects",
      flush=True)

# ── 4. Compute per-subject smile rate per bin ─────────────────────────────────
# Rate = smiles / actual_minutes in that bin.
# actual_minutes for subject s, bin b = bin_width * total_duration_minutes(s)
# So rate_sb = count_sb / (bin_width * total_dur_min_s)
# Then average across subjects.

# First get per-subject total duration in minutes
subj_duration = {}
for subj, tape_vids in by_subject.items():
    total_ms = sum(video_meta[vid]["speech_end_ms"] - video_meta[vid]["speech_start_ms"]
                   for _, vid in tape_vids if vid in video_meta)
    subj_duration[subj] = total_ms / 60_000   # minutes

smile_df["bin"] = pd.cut(smile_df["concat_pos"], bins=bin_edges,
                         labels=False, include_lowest=True).astype("Int64")
subjects = sorted(smile_df["subject"].unique())
rate_matrix = np.full((len(subjects), N_BINS), np.nan)

for si, subj in enumerate(subjects):
    sub    = smile_df[smile_df["subject"] == subj]
    dur_min = subj_duration.get(subj, 0)
    if dur_min <= 0:
        continue
    bin_duration_min = bin_width * dur_min   # actual minutes this bin spans
    for b in range(N_BINS):
        count = (sub["bin"] == b).sum()
        rate_matrix[si, b] = count / bin_duration_min

mean_rate = np.nanmean(rate_matrix, axis=0)
se_rate   = np.nanstd(rate_matrix, axis=0) / np.sqrt(np.sum(~np.isnan(rate_matrix), axis=0).clip(1))
mean_rate_s = gaussian_filter1d(mean_rate, sigma=SIGMA)
se_rate_s   = gaussian_filter1d(se_rate,   sigma=SIGMA)

# ── 5. Load LLM features for context panels ──────────────────────────────────
print("Loading LLM sentence features...", flush=True)
sent_records = []
for fname in sorted(f for f in os.listdir(EYEGAZE_DIR) if f.endswith(".json")):
    vid = fname.replace(".json", "")
    if vid not in video_meta:
        continue
    with open(EYEGAZE_DIR / fname) as fh:
        d = json.load(fh)
    rank    = tape_rank.get(vid)
    n_tapes = n_tapes_map.get(vid)
    subject = video_meta[vid]["subject"]
    sents = d["sentences"]
    if not sents:
        continue
    max_seg = max(s["segment_idx"] for s in sents)
    for s in sents:
        within_tape = s["segment_idx"] / max_seg if max_seg > 0 else 0.0
        concat_pos  = (rank + within_tape) / n_tapes if n_tapes else within_tape
        sent_records.append({
            "subject":         subject,
            "concat_pos":      concat_pos,
            "content_domain":  s["content_domain"],
            "narrative_valence": s["narrative_valence"],
        })

sent_df = pd.DataFrame(sent_records)
print(f"  {len(sent_df):,} sentences loaded", flush=True)

def temporal_proportions(sub_df, col, values, sigma=SIGMA):
    sub_df = sub_df.copy()
    sub_df["bin"] = pd.cut(sub_df["concat_pos"], bins=bin_edges,
                           labels=False, include_lowest=True).astype("Int64")
    dummies = pd.get_dummies(sub_df[col]).reindex(columns=values, fill_value=0)
    dummies["subject"] = sub_df["subject"].values
    dummies["bin"]     = sub_df["bin"].values
    dummies = dummies.dropna(subset=["bin"])
    grp   = dummies.groupby(["subject", "bin"])[values].sum()
    props = grp.div(grp.sum(axis=1).clip(lower=1e-9), axis=0)
    mean_b = props.groupby("bin").mean().reindex(range(N_BINS))
    se_b   = props.groupby("bin").sem().reindex(range(N_BINS))
    out = {}
    for v in values:
        m = mean_b[v].ffill().bfill().values
        s = se_b[v].fillna(0).values
        out[v] = (bin_centers, gaussian_filter1d(m, sigma=sigma), s)
    return out

DOMAINS  = ["pre-war", "wartime", "liberation", "post-war", "present-day", "other"]
VALENCES = ["positive", "negative", "neutral"]
PALETTE  = {
    "pre-war":"#6d28d9","wartime":"#dc2626","liberation":"#16a34a",
    "post-war":"#0891b2","present-day":"#d97706","other":"#9ca3af",
    "positive":"#16a34a","negative":"#dc2626","neutral":"#9ca3af",
}

dom_res = temporal_proportions(sent_df, "content_domain", DOMAINS)
val_res = temporal_proportions(sent_df, "narrative_valence", VALENCES)

# ── 6. Figure ─────────────────────────────────────────────────────────────────
print("Plotting...", flush=True)
fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True,
                         gridspec_kw={"height_ratios": [1.6, 1.2, 1.0]})
fig.suptitle("Smile Frequency Over Normalised Interview Position\n"
             "(per-subject concatenated tapes, mean ± SE across subjects)",
             fontsize=13, fontweight="bold")

# ── Panel 0: Smile rate ───────────────────────────────────────────────────────
ax = axes[0]
color_smile = "#e11d48"
ax.plot(bin_centers, mean_rate_s, color=color_smile, lw=2.5, label="Smile rate")
ax.fill_between(bin_centers,
                (mean_rate_s - se_rate_s).clip(0),
                mean_rate_s + se_rate_s,
                color=color_smile, alpha=0.2)
ax.set_ylabel("Smiles / minute", fontsize=10)
ax.set_xlim(0, 1)
ax.set_ylim(0)
ax.set_title("Smile Rate", fontweight="bold", fontsize=10)
ax.tick_params(labelbottom=False)

# Annotate overall mean
overall_mean = np.nanmean(rate_matrix[rate_matrix > 0])
ax.axhline(overall_mean, color="#999", lw=1, ls="--", alpha=0.7,
           label=f"Grand mean ({overall_mean:.2f}/min)")
ax.legend(fontsize=9)

# ── Panel 1: Content domain stacked area ─────────────────────────────────────
ax = axes[1]
xs      = dom_res[DOMAINS[0]][0]
stacks  = np.array([dom_res[v][1] for v in DOMAINS])
stacks  = stacks / stacks.sum(axis=0, keepdims=True).clip(1e-9)
bottoms = np.zeros(N_BINS)
for v, row in zip(DOMAINS, stacks):
    ax.fill_between(xs, bottoms, bottoms + row, color=PALETTE[v], alpha=0.85, label=v)
    bottoms += row
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_ylabel("Proportion", fontsize=10)
ax.set_title("Content Domain (context)", fontweight="bold", fontsize=10)
ax.legend(fontsize=8, ncol=3, loc="upper right")
ax.tick_params(labelbottom=False)

# ── Panel 2: Narrative valence ────────────────────────────────────────────────
ax = axes[2]
for v, (x, mean, se) in val_res.items():
    c = PALETTE[v]
    ax.plot(x, mean, color=c, lw=2, label=v.capitalize())
    ax.fill_between(x, (mean - se).clip(0), mean + se, color=c, alpha=0.15)
ax.set_xlim(0, 1); ax.set_ylim(0)
ax.set_xlabel("Normalised Position in Interview", fontsize=10)
ax.set_ylabel("Proportion", fontsize=10)
ax.set_title("Narrative Valence (context)", fontweight="bold", fontsize=10)
ax.legend(fontsize=9)

out = FIG_DIR / "28_smile_frequency_temporal.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ── Quick stats ────────────────────────────────────────────────────────────────
print(f"\nSmile stats:")
print(f"  Grand mean rate:  {np.nanmean(rate_matrix[rate_matrix > 0]):.3f} smiles/min")
print(f"  Peak bin position: {bin_centers[np.argmax(mean_rate_s)]:.2f} (of 0-1 scale)")
print(f"  Subjects with smiles: {len(subjects)}")
