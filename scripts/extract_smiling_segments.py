#!/usr/bin/env python3
"""
Extract smiling segments from OpenFace AU12_r data for all videos.

Pipeline per video:
  1. Read AU12_r from result.csv
  2. Gaussian-smooth with sigma = 0.133s (≈4 frames at 30fps)
  3. Threshold at AU12_r > 1.0, find consecutive runs >= 0.5s
  4. Save segments with start_ts, end_ts, peak_r, mean_r, mass_r
"""

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

OPENFACE_DIR = os.environ.get("OPENFACE_DIR", os.path.join(os.path.dirname(PROJECT_DIR), "threadward_results"))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "smiling_segments")

FPS = 30.0
SIGMA_SEC = 0.133
SIGMA_FRAMES = SIGMA_SEC * FPS  # ~4.0 frames
THRESHOLD = 1.0
MIN_DURATION_SEC = 0.5
MIN_DURATION_FRAMES = int(MIN_DURATION_SEC * FPS)


def process_video(video_id):
    """Process a single video. Returns (video_id, num_segments, total_smile_sec) or (video_id, None, error_msg)."""
    csv_path = os.path.join(OPENFACE_DIR, video_id, "result.csv")
    out_path = os.path.join(OUTPUT_DIR, f"{video_id}.json")

    try:
        au12 = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                au12.append(float(row["AU12_r"]))

        if not au12:
            return video_id, None, "empty csv"

        au12 = np.array(au12, dtype=np.float64)
        total_frames = len(au12)
        duration = total_frames / FPS

        smoothed = gaussian_filter1d(au12, sigma=SIGMA_FRAMES)

        above = smoothed > THRESHOLD

        segments = []
        i = 0
        while i < len(above):
            if above[i]:
                start = i
                while i < len(above) and above[i]:
                    i += 1
                end = i
                if (end - start) >= MIN_DURATION_FRAMES:
                    raw_slice = au12[start:end]
                    seg_duration = (end - start) / FPS
                    segments.append({
                        "start_ts": round(start / FPS, 3),
                        "end_ts": round(end / FPS, 3),
                        "peak_r": round(float(np.max(raw_slice)), 4),
                        "mean_r": round(float(np.mean(raw_slice)), 4),
                        "mass_r": round(float(np.sum(raw_slice) / FPS), 4),
                    })
            else:
                i += 1

        total_smile_sec = sum(s["end_ts"] - s["start_ts"] for s in segments)

        result = {
            "video_id": video_id,
            "fps": FPS,
            "total_frames": total_frames,
            "total_duration_sec": round(duration, 3),
            "smoothing_sigma_sec": SIGMA_SEC,
            "threshold": THRESHOLD,
            "min_duration_sec": MIN_DURATION_SEC,
            "num_segments": len(segments),
            "total_smile_sec": round(total_smile_sec, 3),
            "segments": segments,
        }

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        return video_id, len(segments), total_smile_sec

    except Exception as e:
        return video_id, None, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(64, cpu_count()))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    video_ids = sorted(
        d for d in os.listdir(OPENFACE_DIR)
        if os.path.isfile(os.path.join(OPENFACE_DIR, d, "result.csv"))
    )

    if args.limit:
        video_ids = video_ids[:args.limit]

    print(f"Processing {len(video_ids)} videos with {args.workers} workers...")
    t0 = time.time()

    total_segments = 0
    total_smile_sec = 0.0
    errors = 0
    segment_counts = []

    with Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_video, video_ids), 1):
            vid, n_seg, extra = result
            if n_seg is None:
                errors += 1
                if i <= 5 or errors <= 5:
                    print(f"  ERROR {vid}: {extra}")
            else:
                total_segments += n_seg
                total_smile_sec += extra
                segment_counts.append(n_seg)

            if i % 500 == 0:
                elapsed = time.time() - t0
                print(f"  [{i}/{len(video_ids)}] {elapsed:.1f}s elapsed")

    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print("SMILING SEGMENTS EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"  Videos processed:    {len(video_ids) - errors}")
    print(f"  Errors:              {errors}")
    print(f"  Total segments:      {total_segments}")
    print(f"  Total smile time:    {total_smile_sec:.1f}s ({total_smile_sec/3600:.2f} hours)")
    if segment_counts:
        arr = np.array(segment_counts)
        print(f"  Segments per video:  mean={arr.mean():.1f}, median={np.median(arr):.0f}, "
              f"min={arr.min()}, max={arr.max()}")
        print(f"  Videos with 0 segs:  {np.sum(arr == 0)}")
    print(f"  Time elapsed:        {elapsed:.1f}s")
    print(f"  Output:              {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
