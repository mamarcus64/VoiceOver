#!/usr/bin/env python3
"""Generate the smile-why annotation task manifest.

Reads smile_features.csv, filters to downloaded videos, stratifies by
time_to_interviewer (narrative >=20s vs reactive <20s) x score tertiles,
and outputs data/smile_why_task_manifest.json.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
FEATURES_CSV = Path(__file__).resolve().parent / "smile_features.csv"
VIDEO_DIR = DATA / "videos"
FALLBACK_VIDEO_DIR = Path(
    os.environ.get("VOICEOVER_VIDEO_FALLBACK", "/home/mjma/voices/test_data/videos")
)
OUT_PATH = DATA / "smile_why_task_manifest.json"

SEED = 42
TOTAL_TASKS = 500
NARRATIVE_FRAC = 0.70
REACTIVE_THRESHOLD_SEC = 20.0


def downloaded_video_ids() -> set[str]:
    ids: set[str] = set()
    for d in (VIDEO_DIR, FALLBACK_VIDEO_DIR):
        if d.is_dir():
            for f in d.iterdir():
                if f.suffix == ".mp4":
                    ids.add(f.stem)
    return ids


def sample_from_pool(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample n rows from pool, balanced across score tertiles."""
    if len(pool) == 0:
        return pool.iloc[:0]

    terciles = pd.qcut(pool["score"], q=3, labels=["low", "medium", "high"], duplicates="drop")
    pool = pool.copy()
    pool["score_tier"] = terciles

    per_tier = max(1, n // 3)
    remainder = n - per_tier * 3
    parts = []
    for tier in ["low", "medium", "high"]:
        tier_df = pool[pool["score_tier"] == tier]
        take = per_tier + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        take = min(take, len(tier_df))
        if take > 0:
            parts.append(tier_df.sample(n=take, random_state=rng.integers(1 << 31)))
    if not parts:
        return pool.iloc[:0]
    return pd.concat(parts, ignore_index=True)


def main():
    if not FEATURES_CSV.exists():
        print(f"ERROR: {FEATURES_CSV} not found. Run build_features.py first.")
        sys.exit(1)

    print("Loading smile features ...")
    df = pd.read_csv(FEATURES_CSV, low_memory=False, dtype={"video_id": str})
    print(f"  {len(df):,} total smiles")

    df = df.dropna(subset=["video_duration"])
    print(f"  {len(df):,} with video duration")

    available = downloaded_video_ids()
    print(f"  {len(available):,} downloaded videos found on disk")
    df = df[df["video_id"].isin(available)]
    print(f"  {len(df):,} smiles in downloaded videos")

    if len(df) == 0:
        print("WARNING: No smiles in downloaded videos. Writing empty manifest.")
        manifest = {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "study": "smile_why",
            "params": {
                "total_target": TOTAL_TASKS,
                "narrative_frac": NARRATIVE_FRAC,
                "reactive_threshold_sec": REACTIVE_THRESHOLD_SEC,
                "seed": SEED,
            },
            "total_tasks": 0,
            "tasks": [],
        }
        with open(OUT_PATH, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  Wrote {OUT_PATH}")
        return

    narrative = df[(df["time_to_interviewer"] >= REACTIVE_THRESHOLD_SEC) | df["time_to_interviewer"].isna()]
    reactive = df[df["time_to_interviewer"] < REACTIVE_THRESHOLD_SEC]
    print(f"  Narrative pool (>=20s or NaN): {len(narrative):,}")
    print(f"  Reactive pool (<20s):          {len(reactive):,}")

    rng = np.random.default_rng(SEED)

    n_narrative = min(int(TOTAL_TASKS * NARRATIVE_FRAC), len(narrative))
    n_reactive = min(TOTAL_TASKS - n_narrative, len(reactive))
    if n_reactive < int(TOTAL_TASKS * (1 - NARRATIVE_FRAC)):
        n_narrative = min(TOTAL_TASKS - n_reactive, len(narrative))
    print(f"  Sampling: {n_narrative} narrative + {n_reactive} reactive = {n_narrative + n_reactive}")

    sampled_narr = sample_from_pool(narrative, n_narrative, rng)
    sampled_narr["stratum"] = "narrative"

    sampled_react = sample_from_pool(reactive, n_reactive, rng)
    sampled_react["stratum"] = "reactive"

    sampled = pd.concat([sampled_narr, sampled_react], ignore_index=True)
    sampled = sampled.sample(frac=1, random_state=rng.integers(1 << 31)).reset_index(drop=True)

    tasks = []
    for i, row in sampled.iterrows():
        tasks.append({
            "task_number": i + 1,
            "video_id": row["video_id"],
            "smile_start": round(float(row["start_ts"]), 3),
            "smile_end": round(float(row["end_ts"]), 3),
            "score": round(float(row["score"]), 4),
            "time_to_interviewer": round(float(row["time_to_interviewer"]), 1) if pd.notna(row["time_to_interviewer"]) else None,
            "stratum": row["stratum"],
            "score_tier": row["score_tier"],
        })

    manifest = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "study": "smile_why",
        "params": {
            "total_target": TOTAL_TASKS,
            "narrative_frac": NARRATIVE_FRAC,
            "reactive_threshold_sec": REACTIVE_THRESHOLD_SEC,
            "seed": SEED,
            "source_smiles": len(df),
            "downloaded_videos": len(available),
        },
        "total_tasks": len(tasks),
        "tasks": tasks,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Wrote {len(tasks)} tasks to {OUT_PATH}")

    strata = sampled["stratum"].value_counts()
    tiers = sampled["score_tier"].value_counts()
    print(f"  Strata: {dict(strata)}")
    print(f"  Score tiers: {dict(tiers)}")


if __name__ == "__main__":
    main()
