"""
Extract per-smile AU features from OpenFace CSVs and build a smile-level
feature table joining:
  - Smile detections (video_id, start_ts, end_ts, score)
  - AU means during the smile window (AU4, AU6, AU12, AU15, + all others)
  - AU subject baseline (z-score normalisation)
  - LLM sentence features for the overlapping sentence (from sentence_table.pkl)
  - Gaze features (from smile_gaze_features.csv)
  - Interview position (concat_pos)

Output: smile_variable_effects/smile_au_table.pkl
"""

import json, os, pickle, time, collections
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count, set_start_method

BASE       = Path("/Users/marcus/Desktop/usc/VoiceOver")
OF_DIR     = Path("/Users/marcus/Desktop/usc/openface_results")
SMILES_FILE = BASE / "data/detected_smiles.json"
SENT_PKL   = BASE / "smile_variable_effects/sentence_table.pkl"
GAZE_FILE  = BASE / "data/smile_gaze_features.csv"
OUT        = BASE / "smile_variable_effects/smile_au_table.pkl"

AU_COLS = ["AU01_r","AU02_r","AU04_r","AU05_r","AU06_r","AU07_r",
           "AU09_r","AU10_r","AU12_r","AU14_r","AU15_r","AU17_r",
           "AU20_r","AU23_r","AU25_r","AU26_r","AU45_r"]
KEY_AUS = ["AU04_r","AU06_r","AU12_r","AU15_r"]   # the ones we care most about

if __name__ == "__main__":
    try:
        set_start_method("fork")
    except RuntimeError:
        pass

# ── 1. Load smiles grouped by video ──────────────────────────────────────────
print("Loading smiles...", flush=True)
raw = json.loads(SMILES_FILE.read_text())["smiles"]
by_video = collections.defaultdict(list)
for sm in raw:
    by_video[sm["video_id"]].append(sm)

# ── 2. Per-video AU extraction ────────────────────────────────────────────────
def process_video(vid):
    smiles = by_video[vid]
    of_path = OF_DIR / vid / "result.csv"
    if not of_path.exists():
        return [(sm["video_id"], sm["start_ts"], sm["end_ts"], sm["score"])
                + (np.nan,) * (len(AU_COLS) * 2 + 2)
                for sm in smiles]

    try:
        of = pd.read_csv(of_path, usecols=["timestamp"] + AU_COLS)
    except Exception:
        return []

    # Mark valid frames (not all-zero gaze — proxy for face detected)
    # Use AU12 > 0 or any AU > 0 as a rough validity check
    valid_mask = of[AU_COLS].max(axis=1) > 0
    ts   = of["timestamp"].values
    of_v = of.copy()

    # Per-video baseline: mean/std of each AU across ALL valid frames
    valid_rows = of[valid_mask]
    baseline_mean = valid_rows[AU_COLS].mean()
    baseline_std  = valid_rows[AU_COLS].std().clip(lower=1e-6)

    records = []
    for sm in smiles:
        t0 = sm["start_ts"]
        t1 = sm["end_ts"]
        mask = (ts >= t0) & (ts <= t1) & valid_mask
        n = mask.sum()
        if n == 0:
            au_means = {f"au_{a}": np.nan for a in AU_COLS}
            au_z     = {f"auz_{a}": np.nan for a in KEY_AUS}
        else:
            win = of.loc[mask, AU_COLS]
            au_means = {f"au_{a}": win[a].mean() for a in AU_COLS}
            au_z     = {f"auz_{a}": (win[a].mean() - baseline_mean[a]) / baseline_std[a]
                        for a in KEY_AUS}

        rec = {
            "video_id": sm["video_id"],
            "start_ts": sm["start_ts"],
            "end_ts":   sm["end_ts"],
            "score":    sm["score"],
            "n_frames": int(n),
            **au_means,
            **au_z,
        }
        records.append(rec)
    return records


# ── 3. Parallel execution ─────────────────────────────────────────────────────
print(f"Extracting AU features from {len(by_video)} videos "
      f"({cpu_count()} workers)...", flush=True)
t0 = time.time()

all_records = []
vids = list(by_video.keys())
with Pool(processes=max(1, cpu_count() - 1)) as pool:
    for i, recs in enumerate(pool.imap_unordered(process_video, vids, chunksize=10)):
        all_records.extend(recs)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(vids)} videos ({time.time()-t0:.1f}s)", flush=True)

print(f"  Done in {time.time()-t0:.1f}s", flush=True)

# ── 4. Build smile AU DataFrame ───────────────────────────────────────────────
print("Building smile AU table...", flush=True)
smile_au = pd.DataFrame(all_records)
smile_au["video_id"] = smile_au["video_id"].astype(str)

# Duchenne classification: AU6 ≥ 0.5 (a standard threshold used in FACS literature)
smile_au["au_AU06_r"] = smile_au.get("au_AU06_r", np.nan)
smile_au["au_AU12_r"] = smile_au.get("au_AU12_r", np.nan)
smile_au["au_AU15_r"] = smile_au.get("au_AU15_r", np.nan)
smile_au["au_AU04_r"] = smile_au.get("au_AU04_r", np.nan)

smile_au["is_duchenne"] = (smile_au["au_AU06_r"] >= 0.5).astype("boolean")

# AU profile classification (4-way)
def classify_profile(row):
    au6  = row["au_AU06_r"]
    au12 = row["au_AU12_r"]
    au15 = row["au_AU15_r"]
    au4  = row["au_AU04_r"]
    if pd.isna(au6) or pd.isna(au12):
        return "unknown"
    duchenne = au6 >= 0.5
    bitter   = (au15 >= 0.3) or (au4 >= 0.5)
    if duchenne and not bitter:
        return "genuine"         # AU6+AU12, no distress markers
    elif duchenne and bitter:
        return "complex"         # AU6+AU12+AU4/15 — ambivalent
    elif not duchenne and bitter:
        return "bitter"          # AU12+AU4/15 — sad/bitter smile
    else:
        return "polite"          # AU12 alone — social/performed

smile_au["au_profile"] = smile_au.apply(classify_profile, axis=1)

print(f"Profile distribution:")
print(smile_au["au_profile"].value_counts())
print(f"Duchenne rate: {smile_au['is_duchenne'].mean()*100:.1f}%")

# ── 5. Join gaze features ─────────────────────────────────────────────────────
print("Joining gaze features...", flush=True)
gaze = pd.read_csv(GAZE_FILE)
gaze["video_id"] = gaze["video_id"].astype(str)
smile_au = smile_au.merge(
    gaze[["video_id","start_ts","end_ts","frac_looking_at_interviewer",
          "gaze_deviation_euc","gaze_stability_x"]],
    on=["video_id","start_ts","end_ts"], how="left"
)

# ── 6. Join sentence-level LLM features ──────────────────────────────────────
print("Joining LLM sentence features...", flush=True)
sent = pd.read_pickle(SENT_PKL)
sent["video_id"] = sent["video_id"].astype(str)

# For each smile, find the best-overlapping sentence
# We'll do this as an interval join: for each smile, find sentences where
# first_word_ms <= smile_end_ms*1000 AND last_word_ms >= smile_start_ms*1000
# with the most overlap
sent_valid = sent[sent["first_word_ms"].notna() & sent["last_word_ms"].notna()].copy()
sent_valid["fw"] = sent_valid["first_word_ms"].values
sent_valid["lw"] = sent_valid["last_word_ms"].values

SENT_COLS = ["subject","tape","tape_rank","n_tapes","concat_pos",
             "narrative_valence","content_domain","temporal_syntax",
             "narrative_structure","memory_type",
             "audio_valence","eyegaze_valence","has_smile"]

# Group sentences by video for fast lookup
sent_by_vid = {vid: g for vid, g in sent_valid.groupby("video_id")}

print("  Matching smiles to sentences...", flush=True)
matched_rows = []
for _, row in smile_au.iterrows():
    vid  = row["video_id"]
    t0ms = row["start_ts"] * 1000
    t1ms = row["end_ts"]   * 1000
    if vid not in sent_by_vid:
        matched_rows.append({c: np.nan for c in SENT_COLS})
        continue
    sg = sent_by_vid[vid]
    # Find overlapping sentences
    ovlp = sg[(sg["fw"] < t1ms) & (sg["lw"] > t0ms)]
    if len(ovlp) == 0:
        matched_rows.append({c: np.nan for c in SENT_COLS})
        continue
    # Pick the one with maximum overlap
    overlap_len = (np.minimum(t1ms, ovlp["lw"].values) -
                   np.maximum(t0ms, ovlp["fw"].values))
    best = ovlp.iloc[np.argmax(overlap_len)]
    matched_rows.append({c: best[c] for c in SENT_COLS})

matched_df = pd.DataFrame(matched_rows)
smile_au = pd.concat([smile_au.reset_index(drop=True),
                      matched_df.reset_index(drop=True)], axis=1)

# ── 7. Save ───────────────────────────────────────────────────────────────────
smile_au.to_pickle(OUT)
print(f"\nSaved {len(smile_au):,} smiles to {OUT}")
print(f"  With AU data:       {smile_au['n_frames'].gt(0).sum():,}")
print(f"  With sentence join: {smile_au['narrative_valence'].notna().sum():,}")
print(f"  Duchenne:           {smile_au['is_duchenne'].sum():,} "
      f"({smile_au['is_duchenne'].mean()*100:.1f}%)")
print(f"\nProfile breakdown:")
print(smile_au['au_profile'].value_counts().to_string())
