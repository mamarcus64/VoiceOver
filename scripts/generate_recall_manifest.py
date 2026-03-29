#!/usr/bin/env python3
"""
Generate the stratified-sampling manifest for recall estimation.

Population: all AU12 > 1.0 candidate segments from smiling_segments/,
merged at 1.0s gap, min duration 0.5s. Each segment is scored by the
17-AU logistic model.  Segments are binned into K quantile bins by
logistic score, sampled uniformly within each bin (fixed seed), and
globally shuffled for blinded presentation.

See Section 4.4 of the ICMI 2026 paper for the study design.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = Path(os.environ.get("VOICEOVER_DATA_DIR", str(PROJECT_DIR / "data")))
OPENFACE_DIR = Path(os.environ.get(
    "OPENFACE_DIR",
    str(PROJECT_DIR.parent / "openface_results"),
))
MODEL_PATH = PROJECT_DIR / "pilot_analysis" / "pilot_logistic_model.json"

MERGE_DISTANCE = 1.0
MIN_DURATION = 0.5
NUM_BINS = 10
SAMPLES_PER_BIN = 75
SEED = 42


def load_logistic_model(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def merge_segments(segments: list[dict], merge_distance: float, min_duration: float) -> list[dict]:
    segments = sorted(segments, key=lambda s: s["start_ts"])
    merged: list[dict] = []
    for seg in segments:
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


def score_segment(seg: dict, timestamps: np.ndarray, au_data: np.ndarray,
                  csv_au_cols: list[str], model: dict) -> float | None:
    """Compute logistic score for a single segment."""
    mask = (timestamps >= seg["start_ts"]) & (timestamps <= seg["end_ts"])
    if not np.any(mask):
        return None
    slice_data = au_data[mask]

    model_au_cols = model["au_columns"]
    mean_arr = np.array(model["mean"])
    std_arr = np.array(model["std"])
    coef = np.array(model["logistic_coef"])
    intercept = model["logistic_intercept"]

    csv_col_map = {c.replace("_r_mean", "_r"): i for i, c in enumerate(model_au_cols)}
    au_means = np.zeros(len(model_au_cols))
    for csv_col in csv_au_cols:
        if csv_col in csv_col_map:
            col_idx = csv_au_cols.index(csv_col)
            au_means[csv_col_map[csv_col]] = np.mean(slice_data[:, col_idx])

    z = (au_means - mean_arr) / std_arr
    logit = float(np.dot(z, coef) + intercept)
    return 1.0 / (1.0 + np.exp(-logit))


def load_openface(video_id: str) -> tuple[np.ndarray, list[str], np.ndarray] | None:
    csv_path = OPENFACE_DIR / video_id / "result.csv"
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    timestamps = df["timestamp"].to_numpy(dtype=np.float64)
    au_r_cols = sorted(c for c in df.columns if c.startswith("AU") and c.endswith("_r"))
    au_data = df[au_r_cols].to_numpy(dtype=np.float64)
    return timestamps, au_r_cols, au_data


def main():
    parser = argparse.ArgumentParser(description="Generate recall-estimation manifest")
    parser.add_argument("--samples-per-bin", type=int, default=SAMPLES_PER_BIN)
    parser.add_argument("--num-bins", type=int, default=NUM_BINS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    model = load_logistic_model(MODEL_PATH)
    smiling_dir = DATA_DIR / "smiling_segments"
    seg_files = sorted(smiling_dir.glob("*.json"))
    print(f"Found {len(seg_files)} smiling-segment files", file=sys.stderr)

    all_segments: list[dict] = []
    skipped_no_csv = 0

    for i, seg_file in enumerate(seg_files):
        video_id = seg_file.stem
        with open(seg_file) as f:
            seg_data = json.load(f)

        segments = merge_segments(seg_data.get("segments", []), MERGE_DISTANCE, MIN_DURATION)
        if not segments:
            continue

        of_data = load_openface(video_id)
        if of_data is None:
            skipped_no_csv += 1
            continue
        timestamps, csv_au_cols, au_data = of_data

        for seg in segments:
            score = score_segment(seg, timestamps, au_data, csv_au_cols, model)
            if score is None:
                continue
            all_segments.append({
                "video_id": video_id,
                "segment_start": round(seg["start_ts"], 3),
                "segment_end": round(seg["end_ts"], 3),
                "peak_r": round(seg["peak_r"], 4),
                "mean_r": round(seg["mean_r"], 4),
                "logistic_score": round(score, 6),
            })

        if (i + 1) % 500 == 0 or (i + 1) == len(seg_files):
            print(f"  [{i+1}/{len(seg_files)}] {len(all_segments)} segments scored", file=sys.stderr)

    print(f"\nTotal scored segments: {len(all_segments)}", file=sys.stderr)
    print(f"Videos skipped (no OpenFace CSV): {skipped_no_csv}", file=sys.stderr)

    scores = np.array([s["logistic_score"] for s in all_segments])
    quantiles = np.quantile(scores, np.linspace(0, 1, args.num_bins + 1))
    quantiles[0] = -0.001  # ensure all segments included
    quantiles[-1] = 1.001

    bins_info = []
    rng = np.random.default_rng(args.seed)

    sampled_tasks: list[dict] = []
    for k in range(args.num_bins):
        lo, hi = quantiles[k], quantiles[k + 1]
        in_bin = [s for s in all_segments if lo < s["logistic_score"] <= hi]
        n_sample = min(args.samples_per_bin, len(in_bin))
        chosen_indices = rng.choice(len(in_bin), size=n_sample, replace=False)
        chosen = [in_bin[j] for j in sorted(chosen_indices)]

        bins_info.append({
            "bin": k,
            "score_min": round(float(quantiles[k + 0]), 6) if k > 0 else 0.0,
            "score_max": round(float(quantiles[k + 1]), 6) if k < args.num_bins - 1 else 1.0,
            "population": len(in_bin),
            "sampled": n_sample,
        })

        for seg in chosen:
            sampled_tasks.append({**seg, "bin": k})

    rng2 = np.random.default_rng(args.seed + 1)
    perm = rng2.permutation(len(sampled_tasks))
    sampled_tasks = [sampled_tasks[j] for j in perm]

    for i, task in enumerate(sampled_tasks, 1):
        task["task_number"] = i

    manifest = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "study": "recall_estimation",
        "params": {
            "population_threshold": 1.0,
            "merge_distance": MERGE_DISTANCE,
            "min_duration": MIN_DURATION,
            "num_bins": args.num_bins,
            "samples_per_bin": args.samples_per_bin,
            "seed": args.seed,
            "logistic_model": "pilot_logistic_model.json",
        },
        "population_size": len(all_segments),
        "videos_skipped_no_csv": skipped_no_csv,
        "bins": bins_info,
        "total_tasks": len(sampled_tasks),
        "tasks": sampled_tasks,
    }

    output_path = Path(args.output) if args.output else DATA_DIR / "recall_task_manifest.json"
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(sampled_tasks)} tasks to {output_path}", file=sys.stderr)

    print(f"\n{'Bin':>4s}  {'Score Range':>20s}  {'Pop':>7s}  {'Sampled':>7s}")
    for b in bins_info:
        print(f"{b['bin']:4d}  {b['score_min']:.4f} – {b['score_max']:.4f}  "
              f"{b['population']:7d}  {b['sampled']:7d}")


if __name__ == "__main__":
    main()
