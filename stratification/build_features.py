#!/usr/bin/env python3
"""Build a comprehensive feature table for all detected smiles.

Joins smile detections with speaking labels, audio/gaze affect,
OpenFace AUs, and VHA demographics. Outputs smile_features.csv.
"""

import json
import csv
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from bisect import bisect_right

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OPENFACE = BASE.parent / "openface_results"
OUT = Path(__file__).resolve().parent


def load_smiles():
    with open(DATA / "detected_smiles.json") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw["smiles"])
    df["subject_id"] = df["video_id"].apply(lambda x: int(x.split(".")[0]))
    df["tape_num"] = df["video_id"].apply(lambda x: int(x.split(".")[1]))
    df["duration"] = df["end_ts"] - df["start_ts"]
    return df, raw["threshold"]


def load_video_durations():
    durations = {}
    seg_dir = DATA / "smiling_segments"
    if not seg_dir.exists():
        return durations
    for f in seg_dir.iterdir():
        if f.suffix == ".json":
            try:
                with open(f) as fh:
                    d = json.load(fh)
                durations[d["video_id"]] = d["total_duration_sec"]
            except Exception:
                pass
    return durations


def load_speaking_labels(video_id):
    path = DATA / "speaking_labels" / f"{video_id}.csv"
    if not path.exists():
        return None
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                (
                    int(row["start_ms"]) / 1000.0,
                    int(row["end_ms"]) / 1000.0,
                    row["label"],
                )
            )
    rows.sort(key=lambda x: x[0])
    return rows


def extract_speaking_features(smile_start, smile_end, labels):
    """Return (context_at_midpoint, seconds_since_last_interviewer_end, during_own_speech)."""
    if labels is None:
        return None, np.nan, np.nan

    mid = (smile_start + smile_end) / 2.0

    context = "unknown"
    for s, e, lab in labels:
        if s <= mid <= e:
            context = lab
            break

    best_dist = np.nan
    for s, e, lab in labels:
        if "interviewer" in lab:
            gap = smile_start - e
            if gap >= 0 and (np.isnan(best_dist) or gap < best_dist):
                best_dist = gap

    during_own = 1 if context == "interviewee_speaking" else 0
    return context, best_dist, during_own


def load_audio_vad(video_id):
    path = DATA / "audio_vad" / f"{video_id}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        return d.get("segments", [])
    except Exception:
        return None


def extract_audio_affect(smile_start, smile_end, segments):
    if not segments:
        return np.nan, np.nan, np.nan

    mid = (smile_start + smile_end) / 2.0
    best, best_dist = None, 1e9

    for seg in segments:
        seg_mid = (seg["start"] + seg["end"]) / 2.0
        d = abs(mid - seg_mid)
        if d < best_dist:
            best_dist = d
            best = seg

    if best is None or best_dist > 30:
        return np.nan, np.nan, np.nan
    return best.get("valence", np.nan), best.get("arousal", np.nan), best.get("dominance", np.nan)


def load_eyegaze_vad(video_id):
    path = DATA / "eyegaze_vad" / f"{video_id}.csv"
    if not path.exists():
        return None, None
    try:
        df = pd.read_csv(path)
        ts = df["timestamp"].values
        return df, ts
    except Exception:
        return None, None


def extract_gaze_affect(smile_start, smile_end, gaze_df, gaze_ts):
    if gaze_df is None or len(gaze_df) == 0:
        return np.nan, np.nan, np.nan

    mid = (smile_start + smile_end) / 2.0
    idx = np.argmin(np.abs(gaze_ts - mid))
    if abs(gaze_ts[idx] - mid) > 30:
        return np.nan, np.nan, np.nan
    row = gaze_df.iloc[idx]
    return row["valence"], row["arousal"], row["dominance"]


def load_openface(video_id):
    path = OPENFACE / video_id / "result.csv"
    if not path.exists():
        return None, None
    try:
        cols = ["timestamp", "AU04_r", "AU06_r", "AU09_r", "AU12_r"]
        df = pd.read_csv(path, usecols=cols)
        ts = df["timestamp"].values
        return df, ts
    except Exception:
        return None, None


def extract_aus(smile_start, smile_end, of_df, of_ts):
    if of_df is None:
        return np.nan, np.nan, np.nan, np.nan, 0

    mask = (of_ts >= smile_start) & (of_ts <= smile_end)
    n = mask.sum()
    if n == 0:
        mask = (of_ts >= smile_start - 0.5) & (of_ts <= smile_end + 0.5)
        n = mask.sum()
        if n == 0:
            return np.nan, np.nan, np.nan, np.nan, 0

    window = of_df.iloc[mask] if isinstance(mask, np.ndarray) else of_df[mask]
    return (
        window["AU06_r"].mean(),
        window["AU12_r"].mean(),
        window["AU09_r"].mean(),
        window["AU04_r"].mean(),
        n,
    )


def load_vha_demographics():
    meta_dir = DATA / "vha_metadata"
    if not meta_dir.exists():
        return {}
    demographics = {}
    for f in meta_dir.iterdir():
        if not f.name.endswith(".xml"):
            continue
        intcode_str = f.name.replace("intcode-", "").replace(".xml", "")
        try:
            intcode = int(intcode_str)
        except ValueError:
            continue
        try:
            tree = ET.parse(f)
            root = tree.getroot()
            bio = root.find("BiographicalInformation")
            if bio is None:
                continue

            gender_el = bio.find(".//format[@modifier='Interviewee Gender']")
            gender = gender_el.text.strip() if gender_el is not None and gender_el.text else None

            dob_el = bio.find(".//created[@modifier='Interviewee Date of Birth']")
            birth_year = None
            if dob_el is not None and dob_el.text:
                try:
                    birth_year = int(dob_el.text.strip().split("/")[0])
                except (ValueError, IndexError):
                    pass

            country_el = bio.find(".//response[@questionlabel='Country of Birth']")
            country = country_el.text.strip() if country_el is not None and country_el.text else None

            demographics[intcode] = {
                "gender": gender,
                "birth_year": birth_year,
                "country_of_birth": country,
            }
        except Exception:
            continue
    return demographics


def main():
    t0 = time.time()
    print("=" * 60)
    print("  Building Smile Feature Table")
    print("=" * 60)

    # ── Phase 1: base data ──────────────────────────────────────
    print("\n[1/5] Loading smile data ...")
    df, threshold = load_smiles()
    n = len(df)
    print(f"  {n:,} smiles | {df['video_id'].nunique():,} videos | {df['subject_id'].nunique():,} subjects")

    # ── Phase 2: video-level metadata ───────────────────────────
    print("\n[2/5] Computing positional & density features ...")
    durations = load_video_durations()
    df["video_duration"] = df["video_id"].map(durations)

    df["position_in_tape"] = (df["start_ts"] / df["video_duration"]).clip(0, 1)

    max_tapes = df.groupby("subject_id")["tape_num"].max()
    df["max_tape"] = df["subject_id"].map(max_tapes)
    df["tape_frac"] = (df["tape_num"] - 1) / (df["max_tape"] - 1).clip(lower=1)
    df["tape_frac"] = df["tape_frac"].fillna(0).clip(0, 1)

    def phase_label(frac):
        if frac < 0.34:
            return "early"
        elif frac < 0.67:
            return "middle"
        return "late"

    df["interview_phase"] = df["tape_frac"].apply(phase_label)

    vid_counts = df.groupby("video_id").size()
    df["n_smiles_video"] = df["video_id"].map(vid_counts)
    df["smile_density_per_min"] = df["n_smiles_video"] / (df["video_duration"] / 60)

    subj_counts = df.groupby("subject_id").size()
    df["n_smiles_subject"] = df["subject_id"].map(subj_counts)

    df = df.sort_values(["video_id", "start_ts"]).reset_index(drop=True)
    df["smile_order"] = df.groupby("video_id").cumcount()
    prev_end = df.groupby("video_id")["end_ts"].shift(1)
    df["inter_smile_interval"] = df["start_ts"] - prev_end

    # ── Phase 3: per-video feature extraction ───────────────────
    print("\n[3/5] Per-video extraction (speaking, audio, gaze, AUs) ...")
    groups = df.groupby("video_id")
    videos = list(groups.groups.keys())
    n_videos = len(videos)

    speaking_ctx = [None] * n
    time_to_int = np.full(n, np.nan)
    own_speech = np.full(n, np.nan)
    a_val = np.full(n, np.nan)
    a_aro = np.full(n, np.nan)
    a_dom = np.full(n, np.nan)
    g_val = np.full(n, np.nan)
    g_aro = np.full(n, np.nan)
    g_dom = np.full(n, np.nan)
    au06 = np.full(n, np.nan)
    au12 = np.full(n, np.nan)
    au09 = np.full(n, np.nan)
    au04 = np.full(n, np.nan)
    au_nf = np.full(n, np.nan)

    t_phase3 = time.time()
    for vi, vid in enumerate(videos):
        if vi % 500 == 0:
            elapsed = time.time() - t_phase3
            eta = (elapsed / max(vi, 1)) * (n_videos - vi)
            print(f"  [{vi:>5}/{n_videos}]  elapsed={elapsed:.0f}s  eta={eta:.0f}s  ({vid})")

        idxs = groups.groups[vid].tolist()

        labels = load_speaking_labels(vid)
        audio_segs = load_audio_vad(vid)
        gaze_df, gaze_ts = load_eyegaze_vad(vid)
        of_df, of_ts = load_openface(vid)

        for idx in idxs:
            ss = df.at[idx, "start_ts"]
            se = df.at[idx, "end_ts"]

            ctx, dist, own = extract_speaking_features(ss, se, labels)
            speaking_ctx[idx] = ctx
            time_to_int[idx] = dist
            own_speech[idx] = own

            av, aa, ad = extract_audio_affect(ss, se, audio_segs)
            a_val[idx] = av
            a_aro[idx] = aa
            a_dom[idx] = ad

            gv, ga, gd = extract_gaze_affect(ss, se, gaze_df, gaze_ts)
            g_val[idx] = gv
            g_aro[idx] = ga
            g_dom[idx] = gd

            v6, v12, v9, v4, nf = extract_aus(ss, se, of_df, of_ts)
            au06[idx] = v6
            au12[idx] = v12
            au09[idx] = v9
            au04[idx] = v4
            au_nf[idx] = nf

    df["speaking_context"] = speaking_ctx
    df["time_to_interviewer"] = time_to_int
    df["during_own_speech"] = own_speech
    df["audio_valence"] = a_val
    df["audio_arousal"] = a_aro
    df["audio_dominance"] = a_dom
    df["gaze_valence"] = g_val
    df["gaze_arousal"] = g_aro
    df["gaze_dominance"] = g_dom
    df["au06_during"] = au06
    df["au12_during"] = au12
    df["au09_during"] = au09
    df["au04_during"] = au04
    df["au_n_frames"] = au_nf

    # Duchenne: AU06 and AU12 both clearly activated
    has_au = df["au06_during"].notna() & df["au12_during"].notna()
    df["duchenne"] = np.nan
    df.loc[has_au, "duchenne"] = (
        (df.loc[has_au, "au06_during"] > 0.7) & (df.loc[has_au, "au12_during"] > 1.0)
    ).astype(float)

    # AU z-scores relative to video baseline
    au_base = pd.read_csv(DATA / "video_au_baselines.csv")
    au_base = au_base.drop_duplicates(subset="video_id", keep="first").set_index("video_id")
    for au_name in ["AU06", "AU12"]:
        col_during = f"au{au_name[-2:]}_during"
        mean_col = f"{au_name}_r_mean"
        std_col = f"{au_name}_r_std"
        if mean_col in au_base.columns:
            bmean = df["video_id"].map(au_base[mean_col])
            bstd = df["video_id"].map(au_base[std_col]).clip(lower=0.01)
            df[f"au{au_name[-2:]}_zscore"] = (df[col_during] - bmean) / bstd

    print(f"  Phase 3 done in {time.time() - t_phase3:.0f}s")

    # ── Phase 4: demographics ───────────────────────────────────
    print("\n[4/5] Parsing VHA demographics ...")
    demo = load_vha_demographics()
    df["gender"] = df["subject_id"].map(lambda x: demo.get(x, {}).get("gender"))
    df["birth_year"] = df["subject_id"].map(lambda x: demo.get(x, {}).get("birth_year"))
    df["country_of_birth"] = df["subject_id"].map(lambda x: demo.get(x, {}).get("country_of_birth"))
    matched = df["gender"].notna().sum()
    print(f"  Demographics matched for {matched:,}/{n:,} smiles ({100*matched/n:.1f}%)")

    # ── Phase 5: save ──────────────────────────────────────────
    print("\n[5/5] Saving ...")
    out_path = OUT / "smile_features.csv"
    df.to_csv(out_path, index=False)
    print(f"  {len(df):,} rows x {len(df.columns)} cols → {out_path}")

    elapsed_total = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Done in {elapsed_total:.0f}s")
    print(f"{'=' * 60}")
    print(f"  speaking_context  : {df['speaking_context'].notna().sum():>8,}")
    print(f"  audio_valence     : {df['audio_valence'].notna().sum():>8,}")
    print(f"  gaze_valence      : {df['gaze_valence'].notna().sum():>8,}")
    print(f"  au06_during       : {df['au06_during'].notna().sum():>8,}")
    print(f"  duchenne (True)   : {(df['duchenne'] == 1).sum():>8,}")
    print(f"  gender            : {df['gender'].notna().sum():>8,}")


if __name__ == "__main__":
    main()
