#!/usr/bin/env python3
"""Extract contextual features for smile segments from all data sources.

Outputs:
  data/video_au_baselines.csv   — per-video AU mean/std (reusable for any segment set)
  data/labeled_segment_features.csv — full feature matrix for all labeled segments

Data sources:
  1. OpenFace raw AUs: before_10s / during / after_10s windows + z-scores vs video baseline
  2. Eyegaze VAD: valence/arousal/dominance for the three windows
  3. Audio VAD: valence/arousal/dominance for the three windows + speaking fraction
  4. VHA metadata: keyword-derived semantic features for the segment's 1-min window
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OPENFACE_DIR = Path(os.environ.get(
    "OPENFACE_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "openface_results"),
))
EYEGAZE_DIR = DATA_DIR / "eyegaze_vad"
AUDIO_DIR = DATA_DIR / "audio_vad"
META_DIR = DATA_DIR / "vha_metadata"

WINDOW_SEC = 10.0

AU_NAMES = ["AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
            "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
            "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]

# ── Keyword categories ─────────────────────────────────────────────────────────

POSITIVE_STEMS = [
    "family", "photograph", "children", "wedding", "humor", "laugh", "friend",
    "music", "danc", "love", "baby", "reunion", "grandchild", "grandson",
    "granddaugh", "grandfather", "grandmother", "birthday", "holiday", "food",
    "play", "joy", "happy", "liberat", "cultural activ", "religious",
    "school", "sport", "pet",
]
NEGATIVE_STEMS = [
    "death", "murder", "kill", "camp", "ghetto", "deportat", "torture",
    "starvat", "disease", "selection", "forced labor", "march", "shoot",
    "gas chamber", "cremator", "abuse", "violence", "punish", "hunger",
    "persecution", "anti-jewish", "antisemit", "pogrom", "massacre",
    "mass murder", "hiding", "arrest",
]


def classify_keyword(text: str) -> str:
    t = text.lower()
    if any(s in t for s in POSITIVE_STEMS):
        return "positive"
    if any(s in t for s in NEGATIVE_STEMS):
        return "negative"
    return "neutral"


# ── Label converters ───────────────────────────────────────────────────────────

def binary_label_recall(entry: dict) -> str | None:
    lab = entry.get("label")
    return lab if lab in ("smile", "not_a_smile") else None


def binary_label_pilot(entry: dict) -> str | None:
    lab = entry.get("label")
    if lab in ("genuine", "polite", "masking"):
        return "smile"
    if lab == "not_a_smile":
        return "not_a_smile"
    return None


def binary_label_main(entry: dict) -> str | None:
    if entry.get("not_a_smile") or entry.get("label") == "not_a_smile":
        return "not_a_smile"
    if entry.get("label") in ("felt", "false", "miserable"):
        return "smile"
    return None


# ── Data loading ───────────────────────────────────────────────────────────────

def majority_label(labels: list[str]) -> str:
    smiles = labels.count("smile")
    return "smile" if smiles * 2 >= len(labels) else "not_a_smile"


def load_labeled_segments() -> list[dict]:
    """Load all labeled segments from three annotation sources, producing consensus labels."""
    segments: list[dict] = []

    def collect(ann_dir: Path, manifest_path: Path, label_fn, source: str,
                start_key: str = "segment_start", end_key: str = "segment_end"):
        if not manifest_path.is_file() or not ann_dir.is_dir():
            return
        manifest = json.load(open(manifest_path))
        task_by_num: dict[str, dict] = {str(t["task_number"]): t for t in manifest["tasks"]}

        per_task: dict[str, list[str]] = {}
        for f in ann_dir.glob("*.json"):
            data = json.load(open(f))
            for tk, entry in data.get("annotations", {}).items():
                lab = label_fn(entry)
                if lab:
                    per_task.setdefault(tk, []).append(lab)

        for tk, labels in per_task.items():
            task = task_by_num.get(tk)
            if not task:
                continue
            seg = {
                "video_id": task["video_id"],
                "segment_start": task[start_key],
                "segment_end": task[end_key],
                "label": majority_label(labels),
                "source": source,
                "n_annotators": len(labels),
                "logistic_score": task.get("logistic_score"),
                "mean_r": task.get("mean_r"),
                "peak_r": task.get("peak_r"),
                "bin": task.get("bin"),
            }
            segments.append(seg)

    collect(DATA_DIR / "recall_annotations", DATA_DIR / "recall_task_manifest.json",
            binary_label_recall, "recall")
    collect(DATA_DIR / "smile_annotations", DATA_DIR / "smile_task_manifest.json",
            binary_label_main, "main", start_key="smile_start", end_key="smile_end")
    collect(DATA_DIR / "pilot_smile_annotations", DATA_DIR / "pilot_smile_task_manifest.json",
            binary_label_pilot, "pilot", start_key="smile_start", end_key="smile_end")

    return segments


# ── OpenFace baseline computation ──────────────────────────────────────────────

def compute_all_baselines(video_ids: set[str] | None = None) -> pd.DataFrame:
    """Compute per-video AU mean/std. If video_ids is None, processes all available."""
    if video_ids is None:
        video_ids = set()
        if OPENFACE_DIR.is_dir():
            for d in OPENFACE_DIR.iterdir():
                if d.is_dir() and (d / "result.csv").is_file():
                    video_ids.add(d.name)

    cols_to_read = ["timestamp"] + AU_NAMES
    rows = []
    skipped = 0
    for vid in sorted(video_ids):
        csv_path = OPENFACE_DIR / vid / "result.csv"
        if not csv_path.is_file():
            skipped += 1
            continue
        try:
            df = pd.read_csv(csv_path, usecols=cols_to_read)
            means = df[AU_NAMES].mean()
            stds = df[AU_NAMES].std()
            n_frames = len(df)
            duration = float(df["timestamp"].iloc[-1]) if n_frames > 0 else 0.0
            row = {"video_id": vid, "n_frames": n_frames, "duration_s": round(duration, 1)}
            for au in AU_NAMES:
                row[f"{au}_mean"] = round(float(means[au]), 6)
                row[f"{au}_std"] = round(float(stds[au]), 6)
            rows.append(row)
        except Exception as e:
            print(f"  WARN: {vid}: {e}", file=sys.stderr)
            skipped += 1

    if skipped:
        print(f"  Skipped {skipped} video_ids (no CSV or read error)", file=sys.stderr)
    return pd.DataFrame(rows)


# ── Per-segment feature extraction ─────────────────────────────────────────────

def extract_au_window_features(
    df: pd.DataFrame, timestamps: np.ndarray, start: float, end: float
) -> dict[str, float]:
    """Extract mean AU values for before_10s / during / after_10s windows."""
    features: dict[str, float] = {}

    before_mask = (timestamps >= start - WINDOW_SEC) & (timestamps < start)
    during_mask = (timestamps >= start) & (timestamps <= end)
    after_mask = (timestamps > end) & (timestamps <= end + WINDOW_SEC)

    for prefix, mask in [("before10s", before_mask), ("during", during_mask), ("after10s", after_mask)]:
        sub = df.loc[mask, AU_NAMES]
        if len(sub) > 0:
            means = sub.mean()
            for au in AU_NAMES:
                features[f"au_{prefix}_{au}"] = float(means[au])
        else:
            for au in AU_NAMES:
                features[f"au_{prefix}_{au}"] = float("nan")
        features[f"au_{prefix}_n_frames"] = int(mask.sum())

    return features


def extract_au_derived_features(
    window_feats: dict[str, float],
    baseline_row: dict[str, float] | None,
) -> dict[str, float]:
    """Z-scores vs video baseline and temporal deltas."""
    features: dict[str, float] = {}

    for au in AU_NAMES:
        during_val = window_feats.get(f"au_during_{au}", float("nan"))

        # z-score vs video baseline
        if baseline_row:
            mu = baseline_row.get(f"{au}_mean", float("nan"))
            sigma = baseline_row.get(f"{au}_std", float("nan"))
            if sigma and sigma > 1e-6 and not math.isnan(during_val) and not math.isnan(mu):
                features[f"au_zscore_{au}"] = (during_val - mu) / sigma
            else:
                features[f"au_zscore_{au}"] = float("nan")
        else:
            features[f"au_zscore_{au}"] = float("nan")

        # temporal deltas
        before_val = window_feats.get(f"au_before10s_{au}", float("nan"))
        after_val = window_feats.get(f"au_after10s_{au}", float("nan"))
        features[f"au_delta_before_{au}"] = during_val - before_val if not (math.isnan(during_val) or math.isnan(before_val)) else float("nan")
        features[f"au_delta_after_{au}"] = during_val - after_val if not (math.isnan(during_val) or math.isnan(after_val)) else float("nan")

    return features


def extract_eyegaze_features(video_id: str, start: float, end: float) -> dict[str, float]:
    """Extract eyegaze VAD features for before/during/after windows."""
    features: dict[str, float] = {}
    csv_path = EYEGAZE_DIR / f"{video_id}.csv"
    if not csv_path.is_file():
        for prefix in ("before10s", "during", "after10s"):
            for dim in ("valence", "arousal", "dominance"):
                features[f"gaze_{prefix}_{dim}"] = float("nan")
            features[f"gaze_{prefix}_n"] = 0
        for dim in ("valence", "arousal", "dominance"):
            features[f"gaze_delta_before_{dim}"] = float("nan")
            features[f"gaze_delta_after_{dim}"] = float("nan")
        return features

    df = pd.read_csv(csv_path)
    ts = df["timestamp"].values

    for prefix, lo, hi in [
        ("before10s", start - WINDOW_SEC, start),
        ("during", start, end),
        ("after10s", end, end + WINDOW_SEC),
    ]:
        mask = (ts >= lo) & (ts <= hi)
        n = int(mask.sum())
        features[f"gaze_{prefix}_n"] = n
        for dim in ("valence", "arousal", "dominance"):
            if n > 0:
                features[f"gaze_{prefix}_{dim}"] = float(df.loc[mask, dim].mean())
            else:
                features[f"gaze_{prefix}_{dim}"] = float("nan")

    for dim in ("valence", "arousal", "dominance"):
        d = features.get(f"gaze_during_{dim}", float("nan"))
        b = features.get(f"gaze_before10s_{dim}", float("nan"))
        a = features.get(f"gaze_after10s_{dim}", float("nan"))
        features[f"gaze_delta_before_{dim}"] = d - b if not (math.isnan(d) or math.isnan(b)) else float("nan")
        features[f"gaze_delta_after_{dim}"] = d - a if not (math.isnan(d) or math.isnan(a)) else float("nan")

    return features


def extract_audio_features(video_id: str, start: float, end: float) -> dict[str, float]:
    """Extract audio VAD features for before/during/after windows."""
    features: dict[str, float] = {}
    json_path = AUDIO_DIR / f"{video_id}.json"
    empty_dims = ("valence", "arousal", "dominance")

    if not json_path.is_file():
        for prefix in ("before10s", "during", "after10s"):
            for dim in empty_dims:
                features[f"audio_{prefix}_{dim}"] = float("nan")
            features[f"audio_{prefix}_n_segs"] = 0
            features[f"audio_{prefix}_coverage"] = 0.0
        for dim in empty_dims:
            features[f"audio_delta_before_{dim}"] = float("nan")
            features[f"audio_delta_after_{dim}"] = float("nan")
        return features

    data = json.load(open(json_path))
    segs = data.get("segments", [])

    for prefix, win_start, win_end in [
        ("before10s", start - WINDOW_SEC, start),
        ("during", start, end),
        ("after10s", end, end + WINDOW_SEC),
    ]:
        win_dur = max(win_end - win_start, 1e-6)
        overlapping = []
        total_overlap = 0.0
        for s in segs:
            s_start, s_end = s["start"], s["end"]
            overlap_start = max(s_start, win_start)
            overlap_end = min(s_end, win_end)
            if overlap_end > overlap_start:
                weight = overlap_end - overlap_start
                overlapping.append((s, weight))
                total_overlap += weight

        features[f"audio_{prefix}_n_segs"] = len(overlapping)
        features[f"audio_{prefix}_coverage"] = round(total_overlap / win_dur, 4)

        if overlapping:
            for dim in empty_dims:
                weighted_sum = sum(s[dim] * w for s, w in overlapping)
                features[f"audio_{prefix}_{dim}"] = weighted_sum / total_overlap
        else:
            for dim in empty_dims:
                features[f"audio_{prefix}_{dim}"] = float("nan")

    for dim in empty_dims:
        d = features.get(f"audio_during_{dim}", float("nan"))
        b = features.get(f"audio_before10s_{dim}", float("nan"))
        a = features.get(f"audio_after10s_{dim}", float("nan"))
        features[f"audio_delta_before_{dim}"] = d - b if not (math.isnan(d) or math.isnan(b)) else float("nan")
        features[f"audio_delta_after_{dim}"] = d - a if not (math.isnan(d) or math.isnan(a)) else float("nan")

    return features


# ── VHA keyword cache ──────────────────────────────────────────────────────────

_keyword_cache: dict[str, list[dict]] = {}


def _load_keywords_for_intcode(intcode: str) -> list[dict]:
    """Return list of {tape, in_sec, out_sec, keywords: [str], kw_positive, kw_negative, kw_neutral}."""
    if intcode in _keyword_cache:
        return _keyword_cache[intcode]

    xml_path = META_DIR / f"intcode-{intcode}.xml"
    result: list[dict] = []
    if not xml_path.is_file():
        _keyword_cache[intcode] = result
        return result

    try:
        tree = ET.parse(xml_path)
    except Exception:
        _keyword_cache[intcode] = result
        return result

    def parse_time(t: str) -> float:
        parts = t.split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

    for seg in tree.iter("segment"):
        tape = seg.attrib.get("InFile", "")
        in_time = seg.attrib.get("InTime", "00:00:00:00")
        out_time = seg.attrib.get("OutTime", "00:01:00:00")
        kws = [kw.text.strip() for kw in seg.iter("keyword") if kw.text and kw.text.strip()]
        cats = [classify_keyword(k) for k in kws]
        result.append({
            "tape": tape,
            "in_sec": parse_time(in_time),
            "out_sec": parse_time(out_time),
            "keywords": kws,
            "kw_positive": cats.count("positive"),
            "kw_negative": cats.count("negative"),
            "kw_neutral": cats.count("neutral"),
            "kw_total": len(kws),
        })

    _keyword_cache[intcode] = result
    return result


def extract_keyword_features(video_id: str, start: float, _end: float) -> dict[str, Any]:
    """Find the VHA metadata 1-min segment covering this timestamp and extract keyword features."""
    intcode = video_id.split(".")[0]
    tape = video_id.split(".")[1] if "." in video_id else "1"
    segs = _load_keywords_for_intcode(intcode)

    features: dict[str, Any] = {
        "kw_positive": 0, "kw_negative": 0, "kw_neutral": 0, "kw_total": 0,
        "kw_valence_ratio": float("nan"),
        "kw_raw": "",
    }

    for seg in segs:
        if seg["tape"] == tape and seg["in_sec"] <= start < seg["out_sec"]:
            features["kw_positive"] = seg["kw_positive"]
            features["kw_negative"] = seg["kw_negative"]
            features["kw_neutral"] = seg["kw_neutral"]
            features["kw_total"] = seg["kw_total"]
            denom = seg["kw_positive"] + seg["kw_negative"]
            if denom > 0:
                features["kw_valence_ratio"] = seg["kw_positive"] / denom
            elif seg["kw_total"] > 0:
                features["kw_valence_ratio"] = 0.5
            features["kw_raw"] = "|".join(seg["keywords"])
            break

    return features


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines-only", action="store_true",
                        help="Only compute video AU baselines, skip segment features")
    parser.add_argument("--all-videos", action="store_true",
                        help="Compute baselines for ALL videos, not just those with labeled data")
    args = parser.parse_args()

    print("Loading labeled segments...")
    segments = load_labeled_segments()
    print(f"  {len(segments)} labeled segments from {len(set(s['video_id'] for s in segments))} videos")
    print(f"  Sources: {Counter(s['source'] for s in segments)}")
    print(f"  Labels: {Counter(s['label'] for s in segments)}")

    # Group segments by video_id
    by_video: dict[str, list[dict]] = {}
    for seg in segments:
        by_video.setdefault(seg["video_id"], []).append(seg)

    needed_vids = set(by_video.keys()) if not args.all_videos else None

    # ── Step 1: Compute baselines ──
    print(f"\nComputing AU baselines for {'all' if args.all_videos else len(by_video)} videos...")
    t0 = time.time()
    baselines_df = compute_all_baselines(needed_vids)
    elapsed = time.time() - t0
    print(f"  Done: {len(baselines_df)} videos in {elapsed:.1f}s")
    baselines_path = DATA_DIR / "video_au_baselines.csv"
    baselines_df.to_csv(baselines_path, index=False)
    print(f"  Saved → {baselines_path}")

    if args.baselines_only:
        return

    # Index baselines by video_id
    baseline_lookup: dict[str, dict] = {}
    for _, row in baselines_df.iterrows():
        baseline_lookup[row["video_id"]] = row.to_dict()

    # ── Step 2: Extract features per segment ──
    print(f"\nExtracting features for {len(segments)} segments...")
    t0 = time.time()
    feature_rows: list[dict] = []
    processed_videos = 0
    total_videos = len(by_video)

    for vid, vid_segments in sorted(by_video.items()):
        processed_videos += 1
        if processed_videos % 100 == 0 or processed_videos == total_videos:
            elapsed = time.time() - t0
            rate = processed_videos / elapsed if elapsed > 0 else 0
            eta = (total_videos - processed_videos) / rate if rate > 0 else 0
            print(f"  [{processed_videos}/{total_videos}] {vid} ({rate:.0f} vid/s, ETA {eta:.0f}s)")

        # Load OpenFace once per video
        of_path = OPENFACE_DIR / vid / "result.csv"
        of_df = None
        of_timestamps = None
        if of_path.is_file():
            try:
                of_df = pd.read_csv(of_path, usecols=["timestamp"] + AU_NAMES)
                of_timestamps = of_df["timestamp"].values
            except Exception:
                pass

        baseline_row = baseline_lookup.get(vid)

        for seg in vid_segments:
            row: dict[str, Any] = {
                "video_id": seg["video_id"],
                "segment_start": seg["segment_start"],
                "segment_end": seg["segment_end"],
                "label": seg["label"],
                "label_binary": 1 if seg["label"] == "smile" else 0,
                "source": seg["source"],
                "n_annotators": seg["n_annotators"],
                "manifest_logistic_score": seg.get("logistic_score"),
                "manifest_mean_r": seg.get("mean_r"),
                "manifest_peak_r": seg.get("peak_r"),
                "manifest_bin": seg.get("bin"),
            }

            # AU window features
            if of_df is not None and of_timestamps is not None:
                au_window = extract_au_window_features(
                    of_df, of_timestamps, seg["segment_start"], seg["segment_end"]
                )
                row.update(au_window)
                row.update(extract_au_derived_features(au_window, baseline_row))
            else:
                for prefix in ("before10s", "during", "after10s"):
                    for au in AU_NAMES:
                        row[f"au_{prefix}_{au}"] = float("nan")
                    row[f"au_{prefix}_n_frames"] = 0
                for au in AU_NAMES:
                    row[f"au_zscore_{au}"] = float("nan")
                    row[f"au_delta_before_{au}"] = float("nan")
                    row[f"au_delta_after_{au}"] = float("nan")

            # Eyegaze features
            row.update(extract_eyegaze_features(vid, seg["segment_start"], seg["segment_end"]))

            # Audio features
            row.update(extract_audio_features(vid, seg["segment_start"], seg["segment_end"]))

            # Keyword features
            row.update(extract_keyword_features(vid, seg["segment_start"], seg["segment_end"]))

            feature_rows.append(row)

    elapsed = time.time() - t0
    print(f"  Done: {len(feature_rows)} feature rows in {elapsed:.1f}s")

    # Save
    out_path = DATA_DIR / "labeled_segment_features.csv"
    out_df = pd.DataFrame(feature_rows)
    out_df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")
    print(f"  Columns: {len(out_df.columns)}")
    print(f"  Feature columns (excluding metadata): {len([c for c in out_df.columns if c.startswith(('au_', 'gaze_', 'audio_', 'kw_'))])}")

    # Quick summary
    print("\n=== Feature coverage ===")
    for col in out_df.columns:
        if col.startswith(("au_", "gaze_", "audio_", "kw_")) and col != "kw_raw":
            non_nan = out_df[col].notna().sum()
            print(f"  {col}: {non_nan}/{len(out_df)} non-null ({non_nan/len(out_df)*100:.0f}%)")


if __name__ == "__main__":
    main()
