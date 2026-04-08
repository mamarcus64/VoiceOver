"""
Build per-smile feature table using the smile window as the exact unit of analysis.

For each detected smile [start_ts, end_ts]:
  - Extract OpenFace frames strictly within that window
  - Compute gaze_path_rate (rad/s) and blink_rate (blinks/s)
  - Optionally remove head-pose contribution from gaze path (CORRECT_HEAD_POSE)
  - Look up narrative context (valence, domain, topic) from sentence_table
  - Assign neg_cluster for negative-valence smiles

Normalization:
  Z-score each smile's raw features against that SUBJECT's non-smile sentence
  distribution WITHIN THE SAME NARRATIVE VALENCE. So z=+1 means "eyes moved
  1 SD more than this person's typical sentence of the same valence context."
  This eliminates both inter-subject baseline differences and valence confounds.

Output: smile_variable_effects/smile_window_features.pkl
"""

import json, os, time, collections
import numpy as np
import pandas as pd
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
CORRECT_HEAD_POSE = False   # toggle: subtract head rotation from gaze path
BLINK_THRESH      = 0.5     # AU45_r rising-edge threshold
MAX_FRAME_GAP     = 0.10    # ignore gaze diff across gaps > 100 ms
MIN_SMILE_FRAMES  = 10      # skip smiles with fewer valid OpenFace frames

BASE      = Path("/Users/marcus/Desktop/usc/VoiceOver")
OF_DIR    = Path("/Users/marcus/Desktop/usc/openface_results")
INPUT_PKL = BASE / "smile_variable_effects/sentence_table_openface.pkl"
OUT_PKL   = BASE / "smile_variable_effects/smile_window_features.pkl"

# ── Negative-valence topic clusters ───────────────────────────────────────────
TOPIC_CLUSTER = {
    "Captivity":                  "imprisonment/captivity",
    "Forced labor":               "imprisonment/captivity",
    "Daily life (imprisonment)":  "imprisonment/captivity",
    "Feelings and thoughts":      "emotional reflection",
    "Health":                     "physical suffering",
    "Parents":                    "family/loss",
    "Government":                 "persecution/politics",
    "Refugee experiences":        "displacement/flight",
    "Post-conflict":              "post-war aftermath",
    "Daily life (childhood)":     "pre-war life",
    "Liberation":                 "liberation",
}

def primary_topic(t):
    if not t or not isinstance(t, str):
        return None
    parts = [p.strip() for p in t.split(",")]
    return parts[0] if parts[0] else None

# ── 1. Load sentence table ─────────────────────────────────────────────────────
print("Loading sentence table...", flush=True)
df = pd.read_pickle(INPUT_PKL)

# Build per-subject × per-valence non-smile baseline stats (raw sentence-level)
# Used later for z-scoring smile windows
print("Computing per-subject × valence non-smile baselines...", flush=True)
no_smile = df[~df["has_smile"] & df["gaze_path_rate_raw"].notna() & df["blink_rate_raw"].notna()]
baselines = (
    no_smile.groupby(["subject", "narrative_valence"])[["gaze_path_rate_raw", "blink_rate_raw"]]
    .agg(["mean", "std"])
    .rename(columns={"mean": "m", "std": "s"})
)
# Flatten multi-level columns: (feat, stat) → feat_stat
baselines.columns = ["_".join(c) for c in baselines.columns]
baselines = baselines.reset_index()
print(f"  Baselines computed for {len(baselines):,} subject×valence combinations", flush=True)

# ── 2. Build sentence lookup for narrative context ────────────────────────────
# For each (video_id, frame second) → find containing sentence
# Keyed: video_id → sorted arrays of (start_s, end_s, valence, domain, topic, subject)
print("Building sentence lookup...", flush=True)
sent_lookup = {}
timing_ok = df["first_word_ms"].notna() & df["last_word_ms"].notna()
for vid, grp in df[timing_ok].groupby("video_id"):
    grp_s = grp.sort_values("first_word_ms")
    sent_lookup[str(vid)] = {
        "starts":  (grp_s["first_word_ms"].values / 1000.0).astype(np.float32),
        "ends":    (grp_s["last_word_ms"].values  / 1000.0).astype(np.float32),
        "valence": grp_s["narrative_valence"].values,
        "domain":  grp_s["content_domain"].values,
        "topics":  grp_s["topics"].values,
        "subject": grp_s["subject"].values,
    }

def lookup_context(vid, mid_ts):
    """Return (subject, valence, domain, topic) for the sentence containing mid_ts."""
    sl = sent_lookup.get(str(vid))
    if sl is None:
        return None, None, None, None
    idx = np.searchsorted(sl["ends"], mid_ts)
    # Search a small window in case of slight misalignment
    for i in range(max(0, idx - 1), min(len(sl["starts"]), idx + 3)):
        if sl["starts"][i] <= mid_ts <= sl["ends"][i]:
            return sl["subject"][i], sl["valence"][i], sl["domain"][i], sl["topics"][i]
    return None, None, None, None

# ── 3. Load smiles ────────────────────────────────────────────────────────────
print("Loading smiles...", flush=True)
smiles_raw = json.loads((BASE / "data/detected_smiles.json").read_text())["smiles"]
by_video   = collections.defaultdict(list)
for s in smiles_raw:
    by_video[s["video_id"]].append(s)
print(f"  {len(smiles_raw):,} smiles across {len(by_video):,} videos", flush=True)

# ── 4. Process per video ──────────────────────────────────────────────────────
print(f"Extracting smile-window features (CORRECT_HEAD_POSE={CORRECT_HEAD_POSE})...", flush=True)
t0      = time.time()
records = []
videos  = sorted(by_video.keys())

for vi, vid in enumerate(videos):
    of_path = OF_DIR / vid / "result.csv"
    if not of_path.exists():
        continue

    of = pd.read_csv(of_path)
    of.columns = [c.strip() for c in of.columns]

    ts   = of["timestamp"].values.astype(np.float64)
    ax   = of["gaze_angle_x"].values.astype(np.float64)
    ay   = of["gaze_angle_y"].values.astype(np.float64)
    au45 = of["AU45_r"].values.astype(np.float64)
    gx0  = of["gaze_0_x"].values.astype(np.float64)
    gy0  = of["gaze_0_y"].values.astype(np.float64)

    valid = ~((gx0 == 0.0) & (gy0 == 0.0))

    # Frame-to-frame gaze path increments (world-space or head-corrected)
    dt  = np.empty(len(ts)); dt[0]  = 0.0; dt[1:]  = np.diff(ts)
    dax = np.empty(len(ts)); dax[0] = 0.0; dax[1:] = np.diff(ax)
    day = np.empty(len(ts)); day[0] = 0.0; day[1:] = np.diff(ay)

    if CORRECT_HEAD_POSE:
        ry  = of["pose_Ry"].values.astype(np.float64)
        rx  = of["pose_Rx"].values.astype(np.float64)
        dry = np.empty(len(ts)); dry[0] = 0.0; dry[1:] = np.diff(ry)
        drx = np.empty(len(ts)); drx[0] = 0.0; drx[1:] = np.diff(rx)
        dax = dax - dry   # subtract head yaw contribution
        day = day - drx   # subtract head pitch contribution

    step_dist = np.sqrt(dax**2 + day**2)
    bad = ~valid | (dt > MAX_FRAME_GAP)
    bad[0] = True
    bad[1:] |= ~valid[:-1]   # either neighbor invalid
    step_dist[bad] = 0.0

    # Blink onset (rising edge on valid frames)
    blink_onset       = np.zeros(len(ts), dtype=np.float32)
    rising            = (au45[1:] >= BLINK_THRESH) & (au45[:-1] < BLINK_THRESH)
    blink_onset[1:]   = rising.astype(np.float32)
    blink_onset[~valid] = 0.0

    for sm in by_video[vid]:
        s0  = sm["start_ts"]
        s1  = sm["end_ts"]
        dur = s1 - s0
        if dur < 0.2:
            continue

        i0 = int(np.searchsorted(ts, s0))
        i1 = int(np.searchsorted(ts, s1, side="right"))
        if i1 - i0 < 2:
            continue

        n_valid = int(valid[i0:i1].sum())
        if n_valid < MIN_SMILE_FRAMES:
            continue

        gpr = float(step_dist[i0:i1].sum()) / dur
        br  = float(blink_onset[i0:i1].sum()) / dur

        mid = (s0 + s1) / 2.0
        subj, valence, domain, topic = lookup_context(vid, mid)
        if subj is None:
            continue

        ptopic  = primary_topic(topic)
        cluster = TOPIC_CLUSTER.get(ptopic, "other") if valence == "negative" else None

        records.append({
            "video_id":          vid,
            "subject":           subj,
            "start_ts":          s0,
            "end_ts":            s1,
            "duration":          dur,
            "smile_score":       sm["score"],
            "n_valid_frames":    n_valid,
            "gaze_path_rate":    gpr,
            "blink_rate":        br,
            "narrative_valence": valence,
            "content_domain":    domain,
            "primary_topic":     ptopic,
            "neg_cluster":       cluster,
        })

    if (vi + 1) % 500 == 0:
        print(f"  {vi+1}/{len(videos)}  ({time.time()-t0:.0f}s)", flush=True)

print(f"Done extracting ({time.time()-t0:.0f}s). {len(records):,} smile windows.", flush=True)

# ── 5. Build DataFrame and z-score ────────────────────────────────────────────
sw = pd.DataFrame(records)
print(f"Smile window table: {sw.shape}", flush=True)

# Merge baselines and compute z-scores
sw = sw.merge(
    baselines.rename(columns={
        "gaze_path_rate_raw_m": "bl_gpr_m",
        "gaze_path_rate_raw_s": "bl_gpr_s",
        "blink_rate_raw_m":     "bl_br_m",
        "blink_rate_raw_s":     "bl_br_s",
    }),
    on=["subject", "narrative_valence"],
    how="left",
)

# Z-score: (smile_raw - subject_valence_mean) / subject_valence_std
sw["gaze_path_rate_z"] = (sw["gaze_path_rate"] - sw["bl_gpr_m"]) / sw["bl_gpr_s"].replace(0, np.nan)
sw["blink_rate_z"]     = (sw["blink_rate"]     - sw["bl_br_m"])  / sw["bl_br_s"].replace(0, np.nan)

# Drop rows where z-score couldn't be computed (no baseline)
before = len(sw)
sw = sw.dropna(subset=["gaze_path_rate_z", "blink_rate_z"])
print(f"  After dropping missing baselines: {len(sw):,} / {before:,} smiles", flush=True)

# ── 6. Save ───────────────────────────────────────────────────────────────────
sw.to_pickle(OUT_PKL)
print(f"\nSaved → {OUT_PKL}")
print(sw[["gaze_path_rate","blink_rate","gaze_path_rate_z","blink_rate_z"]].describe().round(4).to_string())
print(f"\nValence breakdown:")
print(sw["narrative_valence"].value_counts())
print(f"\nNeg cluster breakdown (negative smiles only):")
print(sw[sw["narrative_valence"]=="negative"]["neg_cluster"].value_counts())
