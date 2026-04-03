#!/usr/bin/env python3
"""Estimate the interviewer's angular position for each video.

For each video that has both eyegaze vectors and speaking labels, collects
gaze_angle_x / gaze_angle_y during interviewer_speaking segments and takes
the median as the interviewer direction.  When a video has too few usable
frames, falls back to the subject-level aggregate (all tape segments pooled).

Output
------
data/interviewer_positions.csv
    video_id, interviewer_angle_x, interviewer_angle_y,
    n_frames, stdev_x, stdev_y, method   (per_video | per_subject | none)
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
GAZE_DIR = DATA / "eyegaze_vectors"
SPEAK_DIR = DATA / "speaking_labels"
OUT_PATH = DATA / "interviewer_positions.csv"

MIN_FRAMES = 30  # ~1 s at 30 fps


def load_interviewer_segments(path: Path) -> list[tuple[float, float]]:
    """Return [(start_ms, end_ms), ...] for interviewer_speaking."""
    segs = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["label"] == "interviewer_speaking":
                segs.append((int(row["start_ms"]), int(row["end_ms"])))
    return segs


def load_gaze(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def gaze_during_interviewer(
    gaze_rows: list[dict],
    int_segs: list[tuple[float, float]],
) -> tuple[list[float], list[float]]:
    """Return (angle_x_list, angle_y_list) for valid frames inside int_segs."""
    if not int_segs:
        return [], []

    ax_list: list[float] = []
    ay_list: list[float] = []

    seg_starts = np.array([s for s, _ in int_segs])
    seg_ends = np.array([e for _, e in int_segs])

    for row in gaze_rows:
        if float(row["gaze_0_x"]) == 0.0 and float(row["gaze_0_y"]) == 0.0:
            continue
        ts_ms = float(row["timestamp"]) * 1000.0
        hits = (ts_ms >= seg_starts) & (ts_ms <= seg_ends)
        if hits.any():
            ax_list.append(float(row["gaze_angle_x"]))
            ay_list.append(float(row["gaze_angle_y"]))

    return ax_list, ay_list


def summarise(ax: list[float], ay: list[float]) -> dict:
    return {
        "angle_x": statistics.median(ax),
        "angle_y": statistics.median(ay),
        "n": len(ax),
        "std_x": statistics.stdev(ax) if len(ax) > 1 else 0.0,
        "std_y": statistics.stdev(ay) if len(ay) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-frames", type=int, default=MIN_FRAMES,
        help="Minimum gaze frames during interviewer speech for per-video estimate",
    )
    args = parser.parse_args()

    gaze_ids = {p.stem for p in GAZE_DIR.glob("*.csv")}
    speak_ids = {p.stem for p in SPEAK_DIR.glob("*.csv")}
    common = sorted(gaze_ids & speak_ids)
    print(f"Videos with both gaze + speaking labels: {len(common)}", file=sys.stderr)

    # ── Pass 1: per-video estimates ──────────────────────────────
    per_video: dict[str, dict] = {}
    subject_pool: dict[str, tuple[list[float], list[float]]] = {}  # subject_id -> (ax, ay)

    t0 = time.time()
    for i, vid in enumerate(common):
        gaze_rows = load_gaze(GAZE_DIR / f"{vid}.csv")
        int_segs = load_interviewer_segments(SPEAK_DIR / f"{vid}.csv")
        ax, ay = gaze_during_interviewer(gaze_rows, int_segs)

        subj = vid.split(".")[0]
        if subj not in subject_pool:
            subject_pool[subj] = ([], [])
        subject_pool[subj][0].extend(ax)
        subject_pool[subj][1].extend(ay)

        if len(ax) >= args.min_frames:
            per_video[vid] = {**summarise(ax, ay), "method": "per_video"}
        else:
            per_video[vid] = None  # resolve in pass 2

        if (i + 1) % 500 == 0 or (i + 1) == len(common):
            elapsed = time.time() - t0
            print(
                f"  pass 1: {i + 1}/{len(common)}  ({elapsed:.1f}s)",
                file=sys.stderr,
            )

    # ── Pass 2: fallback to subject-level aggregate ──────────────
    fallback_count = 0
    none_count = 0
    for vid in common:
        if per_video[vid] is not None:
            continue
        subj = vid.split(".")[0]
        sax, say = subject_pool[subj]
        if len(sax) >= args.min_frames:
            per_video[vid] = {**summarise(sax, say), "method": "per_subject"}
            fallback_count += 1
        else:
            per_video[vid] = {
                "angle_x": float("nan"),
                "angle_y": float("nan"),
                "n": len(sax),
                "std_x": float("nan"),
                "std_y": float("nan"),
                "method": "none",
            }
            none_count += 1

    # ── Write output ─────────────────────────────────────────────
    fieldnames = [
        "video_id", "interviewer_angle_x", "interviewer_angle_y",
        "n_frames", "stdev_x", "stdev_y", "method",
    ]
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for vid in common:
            info = per_video[vid]
            w.writerow({
                "video_id": vid,
                "interviewer_angle_x": f"{info['angle_x']:.6f}" if not np.isnan(info["angle_x"]) else "",
                "interviewer_angle_y": f"{info['angle_y']:.6f}" if not np.isnan(info["angle_y"]) else "",
                "n_frames": info["n"],
                "stdev_x": f"{info['std_x']:.6f}" if not np.isnan(info["std_x"]) else "",
                "stdev_y": f"{info['std_y']:.6f}" if not np.isnan(info["std_y"]) else "",
                "method": info["method"],
            })

    per_video_ok = sum(1 for v in per_video.values() if v["method"] == "per_video")
    print(f"\nDone in {time.time() - t0:.1f}s", file=sys.stderr)
    print(f"  per_video : {per_video_ok}", file=sys.stderr)
    print(f"  per_subject: {fallback_count}", file=sys.stderr)
    print(f"  none       : {none_count}", file=sys.stderr)
    print(f"  output     : {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
