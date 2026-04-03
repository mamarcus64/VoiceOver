#!/usr/bin/env python3
"""Generate timeline strip plots to visually verify interviewer position estimates.

For a sample of videos, plots gaze_angle_x over time with speaking-context
background shading and the estimated interviewer angle as a horizontal line.
Saves a multi-panel figure to stratification/figures/.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
GAZE_DIR = DATA / "eyegaze_vectors"
SPEAK_DIR = DATA / "speaking_labels"
POS_PATH = DATA / "interviewer_positions.csv"
OUT_DIR = PROJECT / "stratification" / "figures"

CONTEXT_COLORS = {
    "interviewer_speaking": "#3B82F6",
    "interviewee_speaking": "#F97316",
    "pause_question":       "#93C5FD",
    "pause_answered":       "#FDBA74",
    "pause_narrative":      "#D1D5DB",
}


def load_positions() -> dict[str, dict]:
    pos = {}
    with open(POS_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] == "none":
                continue
            pos[row["video_id"]] = {
                "angle_x": float(row["interviewer_angle_x"]),
                "angle_y": float(row["interviewer_angle_y"]),
                "std_x": float(row["stdev_x"]) if row["stdev_x"] else 0.0,
                "method": row["method"],
            }
    return pos


def load_gaze_ts(vid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts, ax, ay = [], [], []
    with open(GAZE_DIR / f"{vid}.csv", newline="") as f:
        for row in csv.DictReader(f):
            if float(row["gaze_0_x"]) == 0.0 and float(row["gaze_0_y"]) == 0.0:
                continue
            ts.append(float(row["timestamp"]))
            ax.append(float(row["gaze_angle_x"]))
            ay.append(float(row["gaze_angle_y"]))
    return np.array(ts), np.array(ax), np.array(ay)


def load_speaking(vid: str) -> list[tuple[float, float, str]]:
    segs = []
    with open(SPEAK_DIR / f"{vid}.csv", newline="") as f:
        for row in csv.DictReader(f):
            segs.append((
                int(row["start_ms"]) / 1000.0,
                int(row["end_ms"]) / 1000.0,
                row["label"],
            ))
    return segs


def plot_video(ax_obj, vid: str, pos: dict, gaze_ts, gaze_ax, speaking):
    t_max = gaze_ts[-1] if len(gaze_ts) else 1.0

    for s, e, lab in speaking:
        colour = CONTEXT_COLORS.get(lab, "#E5E7EB")
        ax_obj.axvspan(s, e, alpha=0.25, color=colour, linewidth=0)

    ax_obj.scatter(gaze_ts, gaze_ax, s=0.15, alpha=0.4, color="#1F2937", rasterized=True)

    int_x = pos["angle_x"]
    std_x = pos["std_x"]
    ax_obj.axhline(int_x, color="#DC2626", linewidth=1.2, label="interviewer")
    ax_obj.axhspan(int_x - std_x, int_x + std_x, alpha=0.12, color="#DC2626")
    ax_obj.axhspan(int_x - 0.10, int_x + 0.10, alpha=0.06, color="#DC2626")

    ax_obj.set_xlim(0, t_max)
    ax_obj.set_ylabel("gaze_x (rad)", fontsize=7)
    ax_obj.set_title(f"{vid}  (int_x={int_x:+.3f}, method={pos['method']})", fontsize=8)
    ax_obj.tick_params(labelsize=6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=15, help="Number of videos to sample")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default=None, help="Output path (default: auto)")
    args = parser.parse_args()

    positions = load_positions()
    eligible = sorted(positions.keys())
    print(f"Videos with interviewer position: {len(eligible)}", file=sys.stderr)

    random.seed(args.seed)
    sample = random.sample(eligible, min(args.n, len(eligible)))
    sample.sort()

    n = len(sample)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.0 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for i, vid in enumerate(sample):
        print(f"  plotting {vid} ({i+1}/{n})", file=sys.stderr)
        gaze_ts, gaze_ax, _ = load_gaze_ts(vid)
        speaking = load_speaking(vid)
        plot_video(axes[i], vid, positions[vid], gaze_ts, gaze_ax, speaking)

    axes[-1].set_xlabel("Time (s)")

    legend_patches = [
        mpatches.Patch(color=c, alpha=0.4, label=lab.replace("_", " "))
        for lab, c in CONTEXT_COLORS.items()
    ]
    legend_patches.append(mpatches.Patch(color="#DC2626", alpha=0.5, label="interviewer angle ± 1σ"))
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=7, frameon=False)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUT_DIR / "21_interviewer_gaze_verify.png"
    fig.savefig(out, dpi=180)
    print(f"\nSaved: {out}", file=sys.stderr)
    plt.close(fig)


if __name__ == "__main__":
    main()
