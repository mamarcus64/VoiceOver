"""
Build a comprehensive per-sentence table merging:
  - LLM annotations (llm_annotated_eyegaze)
  - Smile detections (detected_smiles.json)
  - Audio VAD valence (audio_vad)
  - Eyegaze VAD valence (eyegaze_vad)

Output: smile_variable_effects/sentence_table.pkl
"""

import json, os, pickle, collections, time
import numpy as np
import pandas as pd
from pathlib import Path

BASE          = Path("/Users/marcus/Desktop/usc/VoiceOver")
EYEGAZE_DIR   = BASE / "data/llm_annotated_eyegaze"
AUDIO_VAD_DIR = BASE / "data/audio_vad"
EG_VAD_DIR    = BASE / "data/eyegaze_vad"
SMILES_FILE   = BASE / "data/detected_smiles.json"
OUTPUT_DIR    = BASE / "smile_variable_effects"

# ── 1. Load smiles, group by video ────────────────────────────────────────────
print("Loading smiles...", flush=True)
raw_smiles = json.loads(SMILES_FILE.read_text())["smiles"]
smiles_by_video: dict[str, list] = collections.defaultdict(list)
for sm in raw_smiles:
    smiles_by_video[sm["video_id"]].append(sm)

# ── 2. Process each video ─────────────────────────────────────────────────────
print("Processing videos...", flush=True)
t0 = time.time()
records = []
files = sorted(f for f in os.listdir(EYEGAZE_DIR) if f.endswith(".json"))

for fi, fname in enumerate(files):
    vid     = fname.replace(".json", "")
    subject = vid.rsplit(".", 1)[0]
    tape    = float(vid.rsplit(".", 1)[1])

    with open(EYEGAZE_DIR / fname) as fh:
        llm = json.load(fh)
    sents = llm["sentences"]
    if not sents:
        continue

    # ── Pre-load audio VAD as numpy arrays ────────────────────────────────
    audio_starts = audio_ends = audio_vals = np.empty(0)
    audio_path = AUDIO_VAD_DIR / f"{vid}.json"
    if audio_path.exists():
        asegs = json.load(open(audio_path)).get("segments", [])
        if asegs:
            audio_starts = np.array([s["start"] * 1000 for s in asegs])
            audio_ends   = np.array([s["end"]   * 1000 for s in asegs])
            audio_vals   = np.array([s["valence"] for s in asegs])

    # ── Pre-load eyegaze VAD as numpy arrays ──────────────────────────────
    eg_ts = eg_vals = np.empty(0)
    eg_path = EG_VAD_DIR / f"{vid}.csv"
    if eg_path.exists():
        eg_df  = pd.read_csv(eg_path)
        eg_ts  = eg_df["timestamp"].values * 1000          # seconds → ms
        eg_vals = eg_df["valence"].values

    # ── Pre-load smiles as numpy arrays ───────────────────────────────────
    vsm = smiles_by_video.get(vid, [])
    sm_starts = sm_ends = sm_scores = np.empty(0)
    if vsm:
        sm_starts = np.array([s["start_ts"] * 1000 for s in vsm])
        sm_ends   = np.array([s["end_ts"]   * 1000 for s in vsm])
        sm_scores = np.array([s["score"] for s in vsm])

    # ── Process each sentence ─────────────────────────────────────────────
    for si, sent in enumerate(sents):
        fwm = sent.get("first_word_ms")
        lwm = sent.get("last_word_ms")
        has_time = fwm is not None and lwm is not None

        # Smile overlap
        n_smiles  = 0
        max_score = np.nan
        if has_time and len(sm_starts) > 0:
            mask = (sm_starts < lwm) & (sm_ends > fwm)
            n_smiles = int(mask.sum())
            if n_smiles:
                max_score = float(sm_scores[mask].max())

        # Audio VAD (overlap-weighted average)
        audio_val = np.nan
        if has_time and len(audio_starts) > 0:
            mask = (audio_starts < lwm) & (audio_ends > fwm)
            if mask.any():
                overlaps = (np.minimum(lwm, audio_ends[mask])
                          - np.maximum(fwm, audio_starts[mask]))
                if overlaps.sum() > 0:
                    audio_val = float(np.average(audio_vals[mask], weights=overlaps))

        # Eyegaze VAD (mean of frames in window)
        eg_val = np.nan
        if has_time and len(eg_ts) > 0:
            mask = (eg_ts >= fwm) & (eg_ts <= lwm)
            if mask.any():
                eg_val = float(eg_vals[mask].mean())

        records.append((
            subject, vid, tape,
            sent["segment_idx"], sent["sentence_idx_in_seg"], si,
            fwm, lwm,
            sent["memory_type"], sent["content_domain"], sent["temporal_syntax"],
            sent["narrative_structure"], sent["narrative_valence"],
            sent["present_day_valence"],
            ",".join(sent.get("topics", [])),
            n_smiles, max_score, n_smiles > 0,
            audio_val, eg_val,
        ))

    if (fi + 1) % 500 == 0:
        print(f"  {fi+1}/{len(files)} videos ({time.time()-t0:.1f}s)", flush=True)

# ── 3. Build DataFrame ────────────────────────────────────────────────────────
print(f"Building DataFrame ({time.time()-t0:.1f}s)...", flush=True)
COLS = [
    "subject", "video_id", "tape",
    "segment_idx", "sentence_idx_in_seg", "sent_idx_in_video",
    "first_word_ms", "last_word_ms",
    "memory_type", "content_domain", "temporal_syntax",
    "narrative_structure", "narrative_valence", "present_day_valence",
    "topics",
    "n_smiles", "max_smile_score", "has_smile",
    "audio_valence", "eyegaze_valence",
]
df = pd.DataFrame(records, columns=COLS)

# ── 4. Add positional columns ─────────────────────────────────────────────────
tape_info = (df[["subject", "tape"]].drop_duplicates()
             .sort_values(["subject", "tape"]))
tape_info["tape_rank"] = tape_info.groupby("subject").cumcount()
n_tapes = tape_info.groupby("subject")["tape"].count().rename("n_tapes")
tape_info = tape_info.join(n_tapes, on="subject")

df = df.merge(tape_info[["subject", "tape", "tape_rank", "n_tapes"]],
              on=["subject", "tape"], how="left")

max_seg = df.groupby("video_id")["segment_idx"].max().rename("max_seg")
df = df.merge(max_seg, on="video_id", how="left")
df["within_tape_pos"] = df["segment_idx"] / df["max_seg"].clip(lower=1)
df["concat_pos"] = (df["tape_rank"] + df["within_tape_pos"]) / df["n_tapes"]

df = df.sort_values(["subject", "tape_rank", "segment_idx", "sentence_idx_in_seg"])
df["subj_sent_idx"] = df.groupby("subject").cumcount()
df = df.reset_index(drop=True)

# ── 5. Save ───────────────────────────────────────────────────────────────────
out = OUTPUT_DIR / "sentence_table.pkl"
df.to_pickle(out)
elapsed = time.time() - t0

print(f"\nSaved {len(df):,} sentences to {out}  ({elapsed:.1f}s)")
print(f"  Subjects:                {df['subject'].nunique():,}")
print(f"  Sentences with smile:    {df['has_smile'].sum():,}")
print(f"  Audio valence coverage:  {df['audio_valence'].notna().mean()*100:.1f}%")
print(f"  Eyegaze valence coverage:{df['eyegaze_valence'].notna().mean()*100:.1f}%")
print(f"  Negative-valence sents:  {(df['narrative_valence']=='negative').sum():,}")
print(f"  Neg + smile:             {(df['has_smile'] & (df['narrative_valence']=='negative')).sum():,}")
