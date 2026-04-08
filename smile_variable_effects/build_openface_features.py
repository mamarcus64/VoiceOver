"""
Extract per-sentence gaze dynamics from raw OpenFace result.csv files.

Features (raw, then per-subject z-scored):
  gaze_path_rate   — total Euclidean gaze angle path / sentence duration (rad/s)
                     captures how much the eyes move, independent of where they point
  blink_rate       — blink onsets (AU45_r rising edge ≥ 0.5) / sentence duration (blinks/s)

Per-subject normalization: for each subject, all tapes are pooled to compute
the subject's mean and SD, then each sentence is z-scored against that baseline.
This removes baseline differences between subjects (e.g., naturally restless
vs. steady gazers).

Output: smile_variable_effects/sentence_table_openface.pkl
"""

import os, time
import numpy as np
import pandas as pd
from pathlib import Path

BASE        = Path("/Users/marcus/Desktop/usc/VoiceOver")
OF_DIR      = Path("/Users/marcus/Desktop/usc/openface_results")
INPUT_PKL   = BASE / "smile_variable_effects/sentence_table.pkl"
OUTPUT_PKL  = BASE / "smile_variable_effects/sentence_table_openface.pkl"

BLINK_THRESH   = 0.5      # AU45_r threshold for blink onset detection
MAX_FRAME_GAP  = 0.10     # skip gaze diff across gaps > 100ms (scene cuts / tracking loss)
MIN_VALID_FRAC = 0.30     # sentence needs ≥30% valid-gaze frames, else NaN

# ── 1. Load sentence table ────────────────────────────────────────────────────
print("Loading sentence table...", flush=True)
df = pd.read_pickle(INPUT_PKL)
n  = len(df)

# Pre-allocate output arrays
raw_path  = np.full(n, np.nan, dtype=np.float32)   # gaze path (rad/s)
raw_blink = np.full(n, np.nan, dtype=np.float32)   # blink rate (blinks/s)

# ── 2. Group sentences by video ───────────────────────────────────────────────
print("Processing videos...", flush=True)
t0 = time.time()

video_groups = {}
for vid, grp in df.groupby("video_id"):
    video_groups[str(vid)] = grp.index.values

videos  = sorted(video_groups.keys())
n_vids  = len(videos)
skipped = 0

for vi, vid in enumerate(videos):
    of_path = OF_DIR / vid / "result.csv"
    if not of_path.exists():
        skipped += 1
        continue

    # ── Load OpenFace result ─────────────────────────────────────────────────
    of = pd.read_csv(of_path)
    of.columns = [c.strip() for c in of.columns]

    ts   = of["timestamp"].values.astype(np.float64)   # seconds
    ax   = of["gaze_angle_x"].values.astype(np.float64)
    ay   = of["gaze_angle_y"].values.astype(np.float64)
    au45 = of["AU45_r"].values.astype(np.float64)
    gx0  = of["gaze_0_x"].values.astype(np.float64)
    gy0  = of["gaze_0_y"].values.astype(np.float64)

    # Valid tracking mask (OpenFace zeros gaze vectors when tracking fails)
    valid = ~((gx0 == 0.0) & (gy0 == 0.0))

    # ── Precompute frame-to-frame gaze path increments ───────────────────────
    dt   = np.empty(len(ts)); dt[0] = 0.0; dt[1:] = np.diff(ts)
    dax  = np.empty(len(ts)); dax[0] = 0.0; dax[1:] = np.diff(ax)
    day  = np.empty(len(ts)); day[0] = 0.0; day[1:] = np.diff(ay)

    step_dist = np.sqrt(dax**2 + day**2)
    # Zero out invalid steps: tracking failure, time gap (cut), or first frame
    bad_step = ~valid | (dt > MAX_FRAME_GAP)
    bad_step[0] = True
    step_dist[bad_step] = 0.0
    # Also zero out steps that cross a tracking failure (either neighbour invalid)
    bad_either        = np.zeros(len(ts), dtype=bool)
    bad_either[1:]   |= ~valid[:-1]   # previous frame invalid
    step_dist[bad_either] = 0.0

    # ── Precompute blink onset events (rising edge of AU45_r) ─────────────────
    # Rising edge: AU45_r[i] >= thresh and AU45_r[i-1] < thresh, on valid frame
    blink_onset        = np.zeros(len(ts), dtype=np.float32)
    rising             = (au45[1:] >= BLINK_THRESH) & (au45[:-1] < BLINK_THRESH)
    blink_onset[1:]    = rising.astype(np.float32)
    blink_onset[~valid] = 0.0

    # ── Sentence-level aggregation ────────────────────────────────────────────
    row_idxs = video_groups[vid]
    fwm_arr  = df.loc[row_idxs, "first_word_ms"].values
    lwm_arr  = df.loc[row_idxs, "last_word_ms"].values

    for i, idx in enumerate(row_idxs):
        fwm, lwm = fwm_arr[i], lwm_arr[i]
        if np.isnan(fwm) or np.isnan(lwm):
            continue
        dur_s = (lwm - fwm) / 1000.0
        if dur_s < 0.3:          # too short to be meaningful
            continue

        s0 = fwm / 1000.0
        s1 = lwm / 1000.0
        i0 = int(np.searchsorted(ts, s0))
        i1 = int(np.searchsorted(ts, s1, side="right"))
        if i1 - i0 < 3:
            continue

        n_valid = int(valid[i0:i1].sum())
        n_total = i1 - i0
        if n_valid < max(3, n_total * MIN_VALID_FRAC):
            continue

        raw_path[idx]  = float(step_dist[i0:i1].sum()) / dur_s
        raw_blink[idx] = float(blink_onset[i0:i1].sum()) / dur_s

    if (vi + 1) % 500 == 0:
        print(f"  {vi+1}/{n_vids}  ({time.time()-t0:.0f}s)", flush=True)

print(f"Done extracting ({time.time()-t0:.0f}s). Skipped {skipped} missing OF files.", flush=True)
cov = (~np.isnan(raw_path)).sum()
print(f"Coverage: {cov:,}/{n:,} ({cov/n*100:.1f}%)", flush=True)

# ── 3. Attach raw features to df ─────────────────────────────────────────────
df["gaze_path_rate_raw"]  = raw_path
df["blink_rate_raw"]      = raw_blink

# ── 4. Per-subject z-scoring ──────────────────────────────────────────────────
# Pool all sentences across a subject's tapes; compute mean/SD; z-score each row.
print("Computing per-subject z-scores...", flush=True)

for feat in ["gaze_path_rate", "blink_rate"]:
    raw_col = f"{feat}_raw"
    z_col   = f"{feat}_z"
    s_mean  = df.groupby("subject")[raw_col].transform("mean")
    s_std   = df.groupby("subject")[raw_col].transform("std")
    df[z_col] = (df[raw_col] - s_mean) / s_std.replace(0, np.nan)

# ── 5. Save ───────────────────────────────────────────────────────────────────
df.to_pickle(OUTPUT_PKL)
print(f"\nSaved → {OUTPUT_PKL}")
print(df[["gaze_path_rate_raw","blink_rate_raw","gaze_path_rate_z","blink_rate_z"]].describe().to_string())
