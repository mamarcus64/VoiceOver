#!/usr/bin/env python3
"""Generate the smile-valence annotation task manifest.

Reads detected_smiles.json, filters to smiles at or above the file's threshold,
sorts by score descending, caps at MAX_PER_VIDEO smiles per video_id, restricts
to downloaded videos, and optionally caps the total at MAX_TASKS.

Outputs data/smile_valence_task_manifest.json.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
DETECTED_SMILES_PATH = DATA / "detected_smiles.json"
VIDEO_DIR = DATA / "videos"
FALLBACK_VIDEO_DIR = Path(
    os.environ.get("VOICEOVER_VIDEO_FALLBACK", "/home/mjma/voices/test_data/videos")
)
OUT_PATH = DATA / "smile_valence_task_manifest.json"

MAX_PER_VIDEO = 10
MAX_TASKS = None  # set to an int to hard-cap total tasks


def downloaded_video_ids() -> set[str]:
    ids: set[str] = set()
    for d in (VIDEO_DIR, FALLBACK_VIDEO_DIR):
        if d.is_dir():
            for f in d.iterdir():
                if f.suffix == ".mp4":
                    ids.add(f.stem)
    return ids


def main():
    if not DETECTED_SMILES_PATH.exists():
        print(f"ERROR: {DETECTED_SMILES_PATH} not found.")
        sys.exit(1)

    print("Loading detected smiles ...")
    with open(DETECTED_SMILES_PATH) as f:
        raw = json.load(f)

    threshold: float = raw["threshold"]
    smiles: list[dict] = raw["smiles"]
    print(f"  {len(smiles):,} total detected smiles")
    print(f"  Threshold: {threshold}")

    # Filter to at-or-above threshold, then keep only 95th percentile and above
    smiles = [s for s in smiles if s["score"] >= threshold]
    print(f"  {len(smiles):,} at or above threshold")

    p95 = float(np.percentile([s["score"] for s in smiles], 95))
    smiles = [s for s in smiles if s["score"] >= p95]
    print(f"  {len(smiles):,} at or above 95th percentile (score >= {p95:.4f})")

    # Sort by score descending
    smiles.sort(key=lambda s: s["score"], reverse=True)

    available = downloaded_video_ids()
    print(f"  {len(available):,} downloaded videos")

    # Build task list: walk sorted smiles, cap per video, restrict to downloaded
    per_video_count: dict[str, int] = {}
    tasks = []
    task_number = 1
    for smile in smiles:
        vid = smile["video_id"]
        if vid not in available:
            continue
        count = per_video_count.get(vid, 0)
        if count >= MAX_PER_VIDEO:
            continue
        per_video_count[vid] = count + 1
        tasks.append({
            "task_number": task_number,
            "video_id": vid,
            "smile_start": smile["start_ts"],
            "smile_end": smile["end_ts"],
            "score": round(smile["score"], 4),
            "peak_r": round(smile.get("peak_r", 0), 4),
        })
        task_number += 1
        if MAX_TASKS is not None and task_number > MAX_TASKS:
            break

    print(f"  {len(tasks):,} tasks after per-video cap of {MAX_PER_VIDEO}")
    print(f"  {len(per_video_count):,} unique videos")

    manifest = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "study": "smile_valence",
        "params": {
            "threshold": threshold,
            "p95_score": round(p95, 4),
            "max_per_video": MAX_PER_VIDEO,
            "max_tasks": MAX_TASKS,
            "source_smiles": len(raw["smiles"]),
            "downloaded_videos": len(available),
        },
        "total_tasks": len(tasks),
        "tasks": tasks,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {OUT_PATH} ({len(tasks):,} tasks)")


if __name__ == "__main__":
    main()
