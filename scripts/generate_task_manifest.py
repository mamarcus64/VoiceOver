#!/usr/bin/env python3
"""
Generate a deterministic smile-annotation task manifest.

Walks data/annotation_sample.json in order, applies filter/merge logic
to each video's smiling segments, optionally applies a logistic regression
filter over all 17 AU means, caps at --max-per-video, and assigns
sequential global task numbers.

Can be run standalone or imported by the backend.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = Path(os.environ.get("VOICEOVER_DATA_DIR", str(PROJECT_DIR / "data")))
OPENFACE_DIR = Path(os.environ.get(
    "OPENFACE_DIR",
    str(PROJECT_DIR.parent / "openface_results"),
))

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


# ── Logistic filter ──────────────────────────────────────────────────────────

def load_logistic_model(model_path: str | Path) -> dict:
    with open(model_path) as f:
        return json.load(f)


_openface_cache: dict[str, tuple[np.ndarray, list[str], np.ndarray] | None] = {}


def _load_openface(video_id: str, openface_dir: Path) -> tuple[np.ndarray, list[str], np.ndarray] | None:
    if video_id in _openface_cache:
        return _openface_cache[video_id]
    csv_path = openface_dir / video_id / "result.csv"
    if not csv_path.is_file():
        _openface_cache[video_id] = None
        return None
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    timestamps = df["timestamp"].to_numpy(dtype=np.float64)
    au_r_cols = sorted(c for c in df.columns if c.startswith("AU") and c.endswith("_r"))
    au_data = df[au_r_cols].to_numpy(dtype=np.float64)
    result = (timestamps, au_r_cols, au_data)
    _openface_cache[video_id] = result
    return result


def apply_logistic_filter(
    segments: list[dict[str, Any]],
    video_id: str,
    model: dict,
    threshold: float,
    openface_dir: Path,
) -> list[dict[str, Any]] | None:
    """Score segments with the logistic model and keep those >= threshold.

    Returns None if the OpenFace CSV is unavailable for this video.
    """
    of_data = _load_openface(video_id, openface_dir)
    if of_data is None:
        return None
    timestamps, csv_au_cols, au_data = of_data

    model_au_cols = model["au_columns"]
    csv_col_map = {c.replace("_r_mean", "_r"): i for i, c in enumerate(model_au_cols)}
    col_indices = []
    for csv_col in csv_au_cols:
        if csv_col in csv_col_map:
            col_indices.append((csv_au_cols.index(csv_col), csv_col_map[csv_col]))

    mean_arr = np.array(model["mean"])
    std_arr = np.array(model["std"])
    coef = np.array(model["logistic_coef"])
    intercept = model["logistic_intercept"]

    kept = []
    for seg in segments:
        mask = (timestamps >= seg["start_ts"]) & (timestamps <= seg["end_ts"])
        if not np.any(mask):
            continue
        slice_data = au_data[mask]
        au_means = np.zeros(len(model_au_cols))
        for csv_idx, model_idx in col_indices:
            au_means[model_idx] = np.mean(slice_data[:, csv_idx])
        z = (au_means - mean_arr) / std_arr
        logit = float(np.dot(z, coef) + intercept)
        score = 1.0 / (1.0 + np.exp(-logit))
        if score >= threshold:
            seg["logistic_score"] = round(score, 4)
            kept.append(seg)
    return kept


def build_tasks(
    params: dict[str, Any] | None = None,
    data_dir: Path | None = None,
    logistic_model: dict | None = None,
    logistic_threshold: float | None = None,
    openface_dir: Path | None = None,
) -> dict[str, Any]:
    """Build the full task list. Returns the manifest dict (not written to disk)."""
    if data_dir is None:
        data_dir = DATA_DIR
    if params is None:
        params = dict(DEFAULT_PARAMS)
    if openface_dir is None:
        openface_dir = OPENFACE_DIR

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
    videos_skipped_no_csv = 0
    pre_logistic_count = 0

    total_videos = len(sample)
    for vi, entry in enumerate(sample, 1):
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

        pre_logistic_count += len(moments)

        if logistic_model is not None and logistic_threshold is not None:
            moments = apply_logistic_filter(
                moments, video_id, logistic_model, logistic_threshold, openface_dir,
            )
            if moments is None:
                videos_skipped_no_csv += 1
                continue

        if not moments:
            continue

        moments = select_evenly_spaced(moments, max_per_video)
        videos_with_tasks += 1

        for m in moments:
            task_entry = {
                "task_number": task_number,
                "video_id": video_id,
                "smile_start": round(m["start_ts"], 3),
                "smile_end": round(m["end_ts"], 3),
                "peak_r": round(m["peak_r"], 4),
                "mean_r": round(m["mean_r"], 4),
            }
            if "logistic_score" in m:
                task_entry["logistic_score"] = m["logistic_score"]
            tasks.append(task_entry)
            task_number += 1

        if vi % 500 == 0 or vi == total_videos:
            print(f"  [{vi}/{total_videos}] {len(tasks)} tasks so far, "
                  f"{videos_with_tasks} videos with tasks", file=sys.stderr)

    manifest_params: dict[str, Any] = {
        "intensityThreshold": intensity_threshold,
        "mergeDistance": merge_distance,
        "minDuration": min_duration,
        "maxPerVideo": max_per_video,
    }
    if logistic_model is not None:
        manifest_params["logisticThreshold"] = logistic_threshold
        manifest_params["logisticModel"] = "pilot_logistic_model.json"

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "params": manifest_params,
        "total_tasks": len(tasks),
        "videos_with_tasks": videos_with_tasks,
        "pre_logistic_events": pre_logistic_count,
        "videos_skipped_no_csv": videos_skipped_no_csv,
        "tasks": tasks,
    }


def preview_stats(
    params: dict[str, Any] | None = None,
    data_dir: Path | None = None,
    logistic_model: dict | None = None,
    logistic_threshold: float | None = None,
) -> dict[str, Any]:
    """Return summary stats without writing anything."""
    manifest = build_tasks(params, data_dir, logistic_model, logistic_threshold)
    tasks = manifest["tasks"]

    per_video: dict[str, int] = {}
    for t in tasks:
        per_video[t["video_id"]] = per_video.get(t["video_id"], 0) + 1

    counts = list(per_video.values())
    return {
        "total_tasks": manifest["total_tasks"],
        "videos_with_tasks": manifest["videos_with_tasks"],
        "pre_logistic_events": manifest.get("pre_logistic_events", 0),
        "videos_skipped_no_csv": manifest.get("videos_skipped_no_csv", 0),
        "tasks_per_video_mean": round(statistics.mean(counts), 2) if counts else 0,
        "tasks_per_video_median": round(statistics.median(counts)) if counts else 0,
        "tasks_per_video_max": max(counts) if counts else 0,
        "params": manifest["params"],
    }


def generate_and_write(
    params: dict[str, Any] | None = None,
    data_dir: Path | None = None,
    logistic_model: dict | None = None,
    logistic_threshold: float | None = None,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Build tasks and write the manifest file. Returns summary stats."""
    if data_dir is None:
        data_dir = DATA_DIR
    manifest = build_tasks(params, data_dir, logistic_model, logistic_threshold)
    if output_path is None:
        output_path = data_dir / "smile_task_manifest.json"
    else:
        output_path = Path(output_path)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {manifest['total_tasks']} tasks to {output_path}")
    return {
        "total_tasks": manifest["total_tasks"],
        "videos_with_tasks": manifest["videos_with_tasks"],
        "pre_logistic_events": manifest.get("pre_logistic_events", 0),
        "output_path": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate smile-annotation task manifest")
    parser.add_argument("--intensity-threshold", type=float, default=DEFAULT_PARAMS["intensityThreshold"])
    parser.add_argument("--merge-distance", type=float, default=DEFAULT_PARAMS["mergeDistance"])
    parser.add_argument("--min-duration", type=float, default=DEFAULT_PARAMS["minDuration"])
    parser.add_argument("--max-per-video", type=int, default=DEFAULT_PARAMS["maxPerVideo"])
    parser.add_argument("--logistic-model", type=str, default=None,
                        help="Path to pilot_logistic_model.json for 17-AU filtering")
    parser.add_argument("--logistic-threshold", type=float, default=0.636,
                        help="Logistic score threshold (default: 0.636)")
    parser.add_argument("--openface-dir", type=str, default=None,
                        help="Path to OpenFace results (default: $OPENFACE_DIR)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: data/smile_task_manifest.json)")
    parser.add_argument("--preview", action="store_true", help="Print stats only, don't write")
    args = parser.parse_args()

    params = {
        "intensityThreshold": args.intensity_threshold,
        "mergeDistance": args.merge_distance,
        "minDuration": args.min_duration,
        "maxPerVideo": args.max_per_video,
    }

    model = None
    threshold = None
    if args.logistic_model:
        model = load_logistic_model(args.logistic_model)
        threshold = args.logistic_threshold
        print(f"Logistic filter: {args.logistic_model} @ θ={threshold}")
    if args.openface_dir:
        global OPENFACE_DIR
        OPENFACE_DIR = Path(args.openface_dir)

    if args.preview:
        stats = preview_stats(params, logistic_model=model, logistic_threshold=threshold)
        print(json.dumps(stats, indent=2))
    else:
        result = generate_and_write(
            params, logistic_model=model, logistic_threshold=threshold,
            output_path=args.output,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
