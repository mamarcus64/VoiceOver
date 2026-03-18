#!/usr/bin/env python3
"""
Generate a deterministic smile-annotation task manifest.

Walks data/annotation_sample.json in order, applies filter/merge logic
to each video's smiling segments, caps at --max-per-video, and assigns
sequential global task numbers.

Can be run standalone or imported by the backend.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = Path(os.environ.get("VOICEOVER_DATA_DIR", str(PROJECT_DIR / "data")))

DEFAULT_PARAMS = {
    "intensityThreshold": 1.5,
    "mergeDistance": 1.0,
    "minDuration": 0.5,
    "maxPerVideo": 10,
}


def filter_and_merge(
    segments: list[dict[str, Any]],
    intensity_threshold: float,
    merge_distance: float,
    min_duration: float,
) -> list[dict[str, Any]]:
    """Port of the frontend filterAndMerge logic."""
    filtered = [s for s in segments if s["mean_r"] >= intensity_threshold]
    filtered.sort(key=lambda s: s["start_ts"])

    merged: list[dict[str, Any]] = []
    for seg in filtered:
        if merged and seg["start_ts"] - merged[-1]["end_ts"] <= merge_distance:
            last = merged[-1]
            last["end_ts"] = max(last["end_ts"], seg["end_ts"])
            last["peak_r"] = max(last["peak_r"], seg["peak_r"])
            last["mean_r"] = (last["mean_r"] + seg["mean_r"]) / 2
        else:
            merged.append({
                "start_ts": seg["start_ts"],
                "end_ts": seg["end_ts"],
                "peak_r": seg["peak_r"],
                "mean_r": seg["mean_r"],
            })

    return [s for s in merged if s["end_ts"] - s["start_ts"] >= min_duration]


def select_evenly_spaced(items: list, max_count: int) -> list:
    """Pick max_count evenly-spaced items by index."""
    if len(items) <= max_count:
        return items
    n = len(items) - 1
    indices = [round(i * n / (max_count - 1)) for i in range(max_count)]
    return [items[i] for i in indices]


def build_tasks(
    params: dict[str, Any] | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Build the full task list. Returns the manifest dict (not written to disk)."""
    if data_dir is None:
        data_dir = DATA_DIR
    if params is None:
        params = dict(DEFAULT_PARAMS)

    intensity_threshold = params["intensityThreshold"]
    merge_distance = params["mergeDistance"]
    min_duration = params["minDuration"]
    max_per_video = params["maxPerVideo"]

    sample_path = data_dir / "annotation_sample.json"
    smiling_dir = data_dir / "smiling_segments"

    with open(sample_path) as f:
        sample = json.load(f)

    tasks: list[dict[str, Any]] = []
    task_number = 1
    videos_with_tasks = 0

    for entry in sample:
        video_id = entry["id"]
        seg_path = smiling_dir / f"{video_id}.json"
        if not seg_path.is_file():
            continue

        with open(seg_path) as f:
            seg_data = json.load(f)

        moments = filter_and_merge(
            seg_data.get("segments", []),
            intensity_threshold,
            merge_distance,
            min_duration,
        )

        if not moments:
            continue

        moments = select_evenly_spaced(moments, max_per_video)
        videos_with_tasks += 1

        for m in moments:
            tasks.append({
                "task_number": task_number,
                "video_id": video_id,
                "smile_start": round(m["start_ts"], 3),
                "smile_end": round(m["end_ts"], 3),
                "peak_r": round(m["peak_r"], 4),
                "mean_r": round(m["mean_r"], 4),
            })
            task_number += 1

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "params": {
            "intensityThreshold": intensity_threshold,
            "mergeDistance": merge_distance,
            "minDuration": min_duration,
            "maxPerVideo": max_per_video,
        },
        "total_tasks": len(tasks),
        "videos_with_tasks": videos_with_tasks,
        "tasks": tasks,
    }


def preview_stats(params: dict[str, Any] | None = None, data_dir: Path | None = None) -> dict[str, Any]:
    """Return summary stats without writing anything."""
    manifest = build_tasks(params, data_dir)
    tasks = manifest["tasks"]

    per_video: dict[str, int] = {}
    for t in tasks:
        per_video[t["video_id"]] = per_video.get(t["video_id"], 0) + 1

    counts = list(per_video.values())
    return {
        "total_tasks": manifest["total_tasks"],
        "videos_with_tasks": manifest["videos_with_tasks"],
        "tasks_per_video_mean": round(statistics.mean(counts), 2) if counts else 0,
        "tasks_per_video_median": round(statistics.median(counts)) if counts else 0,
        "tasks_per_video_max": max(counts) if counts else 0,
        "params": manifest["params"],
    }


def generate_and_write(
    params: dict[str, Any] | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Build tasks and write the manifest file. Returns summary stats."""
    if data_dir is None:
        data_dir = DATA_DIR
    manifest = build_tasks(params, data_dir)
    out_path = data_dir / "smile_task_manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {manifest['total_tasks']} tasks to {out_path}")
    return {
        "total_tasks": manifest["total_tasks"],
        "videos_with_tasks": manifest["videos_with_tasks"],
        "output_path": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate smile-annotation task manifest")
    parser.add_argument("--intensity-threshold", type=float, default=DEFAULT_PARAMS["intensityThreshold"])
    parser.add_argument("--merge-distance", type=float, default=DEFAULT_PARAMS["mergeDistance"])
    parser.add_argument("--min-duration", type=float, default=DEFAULT_PARAMS["minDuration"])
    parser.add_argument("--max-per-video", type=int, default=DEFAULT_PARAMS["maxPerVideo"])
    parser.add_argument("--preview", action="store_true", help="Print stats only, don't write")
    args = parser.parse_args()

    params = {
        "intensityThreshold": args.intensity_threshold,
        "mergeDistance": args.merge_distance,
        "minDuration": args.min_duration,
        "maxPerVideo": args.max_per_video,
    }

    if args.preview:
        stats = preview_stats(params)
        print(json.dumps(stats, indent=2))
    else:
        result = generate_and_write(params)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
