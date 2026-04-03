#!/usr/bin/env python3
"""Compute gaze-relative-to-interviewer features for every detected smile.

For each smile, looks up the eyegaze vectors during [start_ts, end_ts],
computes how far the subject's gaze deviates from the estimated interviewer
direction, and outputs a feature CSV that can be joined downstream.

Output
------
data/smile_gaze_features.csv
    video_id, start_ts, end_ts,
    gaze_deviation_x, gaze_deviation_y, gaze_deviation_euc,
    frac_looking_at_interviewer, gaze_stability_x, gaze_stability_y,
    n_gaze_frames, n_total_frames, gaze_available
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
GAZE_DIR = DATA / "eyegaze_vectors"
POS_PATH = DATA / "interviewer_positions.csv"
SMILES_PATH = DATA / "detected_smiles.json"
OUT_PATH = DATA / "smile_gaze_features.csv"

LOOK_THRESHOLD_RAD = 0.10  # ~5.7°


def load_positions() -> dict[str, tuple[float, float]]:
    pos = {}
    with open(POS_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] == "none" or not row["interviewer_angle_x"]:
                continue
            pos[row["video_id"]] = (
                float(row["interviewer_angle_x"]),
                float(row["interviewer_angle_y"]),
            )
    return pos


def load_gaze_arrays(path: Path):
    """Return (timestamps, angle_x, angle_y, valid_mask) as numpy arrays."""
    ts, ax, ay, valid = [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            t = float(row["timestamp"])
            gx = float(row["gaze_angle_x"])
            gy = float(row["gaze_angle_y"])
            is_valid = not (float(row["gaze_0_x"]) == 0.0 and float(row["gaze_0_y"]) == 0.0)
            ts.append(t)
            ax.append(gx)
            ay.append(gy)
            valid.append(is_valid)
    return np.array(ts), np.array(ax), np.array(ay), np.array(valid, dtype=bool)


def extract_features(
    start_ts: float,
    end_ts: float,
    gaze_ts: np.ndarray,
    gaze_ax: np.ndarray,
    gaze_ay: np.ndarray,
    gaze_valid: np.ndarray,
    int_x: float,
    int_y: float,
    threshold: float,
) -> dict:
    mask = (gaze_ts >= start_ts) & (gaze_ts <= end_ts)
    n_total = int(mask.sum())
    mask_valid = mask & gaze_valid
    n_valid = int(mask_valid.sum())

    if n_valid == 0:
        return {
            "gaze_deviation_x": "",
            "gaze_deviation_y": "",
            "gaze_deviation_euc": "",
            "frac_looking_at_interviewer": "",
            "gaze_stability_x": "",
            "gaze_stability_y": "",
            "n_gaze_frames": 0,
            "n_total_frames": n_total,
            "gaze_available": False,
        }

    dx = gaze_ax[mask_valid] - int_x
    dy = gaze_ay[mask_valid] - int_y
    euc = np.sqrt(dx**2 + dy**2)

    med_dx = float(np.median(dx))
    med_dy = float(np.median(dy))
    med_euc = float(np.median(euc))
    frac_look = float((euc < threshold).sum()) / n_valid

    std_x = float(np.std(gaze_ax[mask_valid])) if n_valid > 1 else 0.0
    std_y = float(np.std(gaze_ay[mask_valid])) if n_valid > 1 else 0.0

    return {
        "gaze_deviation_x": f"{med_dx:.6f}",
        "gaze_deviation_y": f"{med_dy:.6f}",
        "gaze_deviation_euc": f"{med_euc:.6f}",
        "frac_looking_at_interviewer": f"{frac_look:.4f}",
        "gaze_stability_x": f"{std_x:.6f}",
        "gaze_stability_y": f"{std_y:.6f}",
        "n_gaze_frames": n_valid,
        "n_total_frames": n_total,
        "gaze_available": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold", type=float, default=LOOK_THRESHOLD_RAD,
        help="Angular threshold (rad) for 'looking at interviewer'",
    )
    args = parser.parse_args()

    positions = load_positions()
    print(f"Interviewer positions loaded: {len(positions)}", file=sys.stderr)

    with open(SMILES_PATH) as f:
        smiles = json.load(f)["smiles"]
    print(f"Total smiles: {len(smiles)}", file=sys.stderr)

    fieldnames = [
        "video_id", "start_ts", "end_ts",
        "gaze_deviation_x", "gaze_deviation_y", "gaze_deviation_euc",
        "frac_looking_at_interviewer",
        "gaze_stability_x", "gaze_stability_y",
        "n_gaze_frames", "n_total_frames", "gaze_available",
    ]

    smiles_by_video: dict[str, list[dict]] = {}
    for s in smiles:
        smiles_by_video.setdefault(s["video_id"], []).append(s)

    eligible_vids = sorted(set(smiles_by_video.keys()) & set(positions.keys()))
    skip_vids = set(smiles_by_video.keys()) - set(positions.keys())
    print(f"Videos with position + smiles: {len(eligible_vids)}", file=sys.stderr)
    print(f"Videos skipped (no position): {len(skip_vids)}", file=sys.stderr)

    t0 = time.time()
    rows_written = 0
    no_gaze_count = 0

    with open(OUT_PATH, "w", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=fieldnames)
        w.writeheader()

        for vi, vid in enumerate(eligible_vids):
            gaze_path = GAZE_DIR / f"{vid}.csv"
            if not gaze_path.exists():
                for s in smiles_by_video[vid]:
                    w.writerow({
                        "video_id": vid,
                        "start_ts": s["start_ts"],
                        "end_ts": s["end_ts"],
                        "gaze_deviation_x": "", "gaze_deviation_y": "",
                        "gaze_deviation_euc": "",
                        "frac_looking_at_interviewer": "",
                        "gaze_stability_x": "", "gaze_stability_y": "",
                        "n_gaze_frames": 0, "n_total_frames": 0,
                        "gaze_available": False,
                    })
                    no_gaze_count += 1
                    rows_written += 1
                continue

            gaze_ts, gaze_ax, gaze_ay, gaze_valid = load_gaze_arrays(gaze_path)
            int_x, int_y = positions[vid]

            for s in smiles_by_video[vid]:
                feats = extract_features(
                    s["start_ts"], s["end_ts"],
                    gaze_ts, gaze_ax, gaze_ay, gaze_valid,
                    int_x, int_y, args.threshold,
                )
                if not feats["gaze_available"]:
                    no_gaze_count += 1
                w.writerow({
                    "video_id": vid,
                    "start_ts": s["start_ts"],
                    "end_ts": s["end_ts"],
                    **feats,
                })
                rows_written += 1

            if (vi + 1) % 500 == 0 or (vi + 1) == len(eligible_vids):
                elapsed = time.time() - t0
                print(
                    f"  {vi + 1}/{len(eligible_vids)} videos  "
                    f"({rows_written:,} smiles, {elapsed:.1f}s)",
                    file=sys.stderr,
                )

        for vid in sorted(skip_vids):
            for s in smiles_by_video[vid]:
                w.writerow({
                    "video_id": vid,
                    "start_ts": s["start_ts"],
                    "end_ts": s["end_ts"],
                    "gaze_deviation_x": "", "gaze_deviation_y": "",
                    "gaze_deviation_euc": "",
                    "frac_looking_at_interviewer": "",
                    "gaze_stability_x": "", "gaze_stability_y": "",
                    "n_gaze_frames": 0, "n_total_frames": 0,
                    "gaze_available": False,
                })
                rows_written += 1
                no_gaze_count += 1

    elapsed = time.time() - t0
    gaze_ok = rows_written - no_gaze_count
    print(f"\nDone in {elapsed:.1f}s", file=sys.stderr)
    print(f"  rows written   : {rows_written:,}", file=sys.stderr)
    print(f"  with gaze data : {gaze_ok:,}", file=sys.stderr)
    print(f"  no gaze        : {no_gaze_count:,}", file=sys.stderr)
    print(f"  output         : {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
