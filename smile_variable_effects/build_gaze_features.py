"""
Enrich sentence table with per-sentence gaze features from eyegaze_localized,
and extract frame-level gaze data around each smile for temporal analyses.

Outputs:
  sentence_table_gaze.pkl   — sentence table + gaze columns
  smile_trajectories.pkl    — per-smile gaze_sd_level at ±1.5s around onset
  smile_gaze_shifts.pkl     — per-smile gaze_x shift (last third − first third)
"""

import json, os, time, collections, pickle
import numpy as np
import pandas as pd
from pathlib import Path

BASE          = Path("/Users/marcus/Desktop/usc/VoiceOver")
LOCALIZED_DIR = BASE / "data/eyegaze_localized"
SMILES_FILE   = BASE / "data/detected_smiles.json"
INPUT_PKL     = BASE / "smile_variable_effects/sentence_table.pkl"
OUTPUT_DIR    = BASE / "smile_variable_effects"

ONSET_WINDOW  = 45  # ±45 frames = ±1.5s at 30fps
N_OFFSETS     = 2 * ONSET_WINDOW + 1  # 91

# ── 1. Load inputs ────────────────────────────────────────────────────────────
print("Loading sentence table...", flush=True)
df = pd.read_pickle(INPUT_PKL)

print("Loading smiles...", flush=True)
raw_smiles = json.loads(SMILES_FILE.read_text())["smiles"]
smiles_by_video = collections.defaultdict(list)
for sm in raw_smiles:
    smiles_by_video[sm["video_id"]].append(sm)

# ── 2. Prepare output arrays ─────────────────────────────────────────────────
n = len(df)
gaze_mean_sd     = np.full(n, np.nan, dtype=np.float32)
gaze_mean_abs_x  = np.full(n, np.nan, dtype=np.float32)
gaze_x_std_arr   = np.full(n, np.nan, dtype=np.float32)
gaze_frac_at_int = np.full(n, np.nan, dtype=np.float32)

# Smile-level collectors
traj_records = []   # dicts with 'subject', 'narrative_valence', 'trajectory'
shift_records = []  # dicts with per-smile gaze shift data

# ── 3. Process per video ─────────────────────────────────────────────────────
print("Processing videos...", flush=True)
t0 = time.time()

video_groups = {}
for video_id, grp in df.groupby("video_id"):
    video_groups[video_id] = grp.index.values

videos = sorted(video_groups.keys())
n_vids = len(videos)

for vi, video_id in enumerate(videos):
    loc_path = LOCALIZED_DIR / f"{video_id}.csv"
    if not loc_path.exists():
        continue

    # Load only needed columns
    gdf = pd.read_csv(loc_path, usecols=["timestamp", "gaze_x", "gaze_x_sd_level"])
    ts  = gdf["timestamp"].values            # seconds
    gx  = gdf["gaze_x"].values
    gsd = gdf["gaze_x_sd_level"].values

    row_idxs = video_groups[video_id]
    fwm_arr  = df.loc[row_idxs, "first_word_ms"].values
    lwm_arr  = df.loc[row_idxs, "last_word_ms"].values

    # ── Sentence-level gaze features ──────────────────────────────────────
    for i, idx in enumerate(row_idxs):
        fwm, lwm = fwm_arr[i], lwm_arr[i]
        if np.isnan(fwm) or np.isnan(lwm):
            continue
        s0, s1 = fwm / 1000.0, lwm / 1000.0
        i0 = np.searchsorted(ts, s0)
        i1 = np.searchsorted(ts, s1, side="right")
        if i1 - i0 < 3:
            continue

        chunk_sd = gsd[i0:i1]
        chunk_x  = gx[i0:i1]
        valid_sd = chunk_sd[~np.isnan(chunk_sd)]
        valid_x  = chunk_x[~np.isnan(chunk_x)]

        if len(valid_sd) >= 3:
            gaze_mean_sd[idx]     = valid_sd.mean()
            gaze_frac_at_int[idx] = (valid_sd <= 1.0).mean()
        if len(valid_x) >= 3:
            gaze_mean_abs_x[idx]  = np.abs(valid_x).mean()
            gaze_x_std_arr[idx]   = valid_x.std()

    # ── Smile-level data ──────────────────────────────────────────────────
    # Build sentence lookup for this video (sorted by time)
    sent_sub = df.loc[row_idxs].sort_values("first_word_ms")
    sent_starts = sent_sub["first_word_ms"].values / 1000.0
    sent_ends   = sent_sub["last_word_ms"].values / 1000.0
    sent_vals   = sent_sub["narrative_valence"].values
    sent_subjs  = sent_sub["subject"].values

    # Mask out NaN sentence times
    sent_valid = ~(np.isnan(sent_starts) | np.isnan(sent_ends))

    for sm in smiles_by_video.get(video_id, []):
        sm_start = sm["start_ts"]
        sm_end   = sm["end_ts"]
        sm_mid   = (sm_start + sm_end) / 2.0

        # Match smile midpoint to sentence
        if not sent_valid.any():
            continue
        match = sent_valid & (sent_starts <= sm_mid) & (sent_ends >= sm_mid)
        if not match.any():
            continue
        si = np.where(match)[0][0]
        valence = sent_vals[si]
        subject = sent_subjs[si]

        # ── C: Frame trajectory around onset ──────────────────────────────
        onset_fi = np.searchsorted(ts, sm_start)
        traj = np.full(N_OFFSETS, np.nan, dtype=np.float32)
        lo = max(0, onset_fi - ONSET_WINDOW)
        hi = min(len(gsd), onset_fi + ONSET_WINDOW + 1)
        for fi in range(lo, hi):
            oi = fi - onset_fi + ONSET_WINDOW
            traj[oi] = gsd[fi]
        traj_records.append({
            "subject": subject,
            "narrative_valence": valence,
            "trajectory": traj,
        })

        # ── E: Gaze shift during smile ────────────────────────────────────
        fi0 = np.searchsorted(ts, sm_start)
        fi1 = np.searchsorted(ts, sm_end, side="right")
        if fi1 - fi0 < 6:
            continue
        smile_gx = gx[fi0:fi1]
        valid = ~np.isnan(smile_gx)
        nv = valid.sum()
        if nv < 6:
            continue
        vals_clean = smile_gx[valid]
        third = nv // 3
        first_third = vals_clean[:third].mean()
        last_third  = vals_clean[-third:].mean()
        shift_records.append({
            "subject":           subject,
            "video_id":          video_id,
            "narrative_valence": valence,
            "smile_score":       sm["score"],
            "gaze_shift":        float(last_third - first_third),
            "gaze_first_third":  float(first_third),
            "gaze_last_third":   float(last_third),
            "n_frames":          int(nv),
        })

    if (vi + 1) % 500 == 0:
        print(f"  {vi+1}/{n_vids} videos ({time.time()-t0:.1f}s)", flush=True)

# ── 4. Save enriched sentence table ──────────────────────────────────────────
print(f"Saving ({time.time()-t0:.1f}s)...", flush=True)
df["gaze_mean_sd"]     = gaze_mean_sd
df["gaze_mean_abs_x"]  = gaze_mean_abs_x
df["gaze_x_std"]       = gaze_x_std_arr
df["gaze_frac_at_int"] = gaze_frac_at_int

out1 = OUTPUT_DIR / "sentence_table_gaze.pkl"
df.to_pickle(out1)

# ── 5. Save smile-level data ─────────────────────────────────────────────────
# Trajectories: store as dict of arrays for compactness
traj_subjects  = [r["subject"] for r in traj_records]
traj_valences  = [r["narrative_valence"] for r in traj_records]
traj_matrix    = np.array([r["trajectory"] for r in traj_records], dtype=np.float32)

out2 = OUTPUT_DIR / "smile_trajectories.pkl"
with open(out2, "wb") as f:
    pickle.dump({
        "subjects":  traj_subjects,
        "valences":  traj_valences,
        "matrix":    traj_matrix,
        "onset_window": ONSET_WINDOW,
    }, f)

shift_df = pd.DataFrame(shift_records)
out3 = OUTPUT_DIR / "smile_gaze_shifts.pkl"
shift_df.to_pickle(out3)

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s")
print(f"  Sentence table: {out1}  ({df['gaze_mean_sd'].notna().mean()*100:.1f}% gaze coverage)")
print(f"  Trajectories:   {out2}  ({len(traj_records):,} smiles × {N_OFFSETS} frames)")
print(f"  Gaze shifts:    {out3}  ({len(shift_records):,} smiles)")
