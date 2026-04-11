#!/usr/bin/env python3
"""Interactive threshold tuner for audio/eyegaze → valence labels.

Loads human annotations once, then lets you type new threshold pairs and
immediately see accuracy + confusion matrices.

Usage:
    python scripts/tune_valence_thresholds.py
"""

import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
MANIFEST_PATH  = DATA / "smile_valence_task_manifest.json"
ANN_DIR        = DATA / "smile_valence_annotations"
AUDIO_DIR      = DATA / "audio_vad"
EYEGAZE_DIR    = DATA / "eyegaze_vad"

VALENCE_ORD  = {"negative": -1, "neutral": 0, "positive": 1}
ORD_VALENCE  = {-1: "neg", 0: "neu", 1: "pos"}
CLASSES      = [-1, 0, 1]

# ── ANSI colours ──────────────────────────────────────────────────────────────
RED, YLW, GRN, CYN, BLD, DIM, RST = (
    "\033[91m", "\033[93m", "\033[92m", "\033[96m",
    "\033[1m",  "\033[2m",  "\033[0m",
)

def col(v: int) -> str:
    return {-1: RED, 0: YLW, 1: GRN}[v]


# ── Feature extraction ────────────────────────────────────────────────────────

def audio_valence(video_id, smile_start, smile_end):
    fp = AUDIO_DIR / f"{video_id}.json"
    if not fp.is_file():
        return None
    with open(fp) as f:
        segs = json.load(f)["segments"]
    tw = tv = 0.0
    for s in segs:
        ov = min(smile_end, s["end"]) - max(smile_start, s["start"])
        if ov > 0:
            tw += ov; tv += ov * s["valence"]
    return tv / tw if tw > 0 else None


def eyegaze_valence(video_id, smile_start, smile_end, ctx=5.0):
    fp = EYEGAZE_DIR / f"{video_id}.csv"
    if not fp.is_file():
        return None
    df = pd.read_csv(fp)
    win = df[(df["timestamp"] >= smile_start - ctx) & (df["timestamp"] <= smile_end + ctx)]
    if win.empty:
        return None
    w = np.where(
        (win["timestamp"] >= smile_start) & (win["timestamp"] <= smile_end), 1.0,
        np.maximum(0.0, 1.0 - np.minimum(
            np.abs(win["timestamp"] - smile_start),
            np.abs(win["timestamp"] - smile_end)) / ctx)
    )
    return float(np.average(win["valence"].values, weights=w)) if w.sum() > 0 else None


# ── Discretise ───────────────────────────────────────────────────────────────

def discretise(value: float, t_lo: float, t_hi: float) -> int:
    if value < t_lo: return -1
    if value < t_hi: return 0
    return 1


# ── Display helpers ───────────────────────────────────────────────────────────

def bar(v: float, width: int = 30) -> str:
    filled = round(v * width)
    return "█" * filled + "░" * (width - filled)


def print_results(label: str, y_true, y_pred, modality_vals=None,
                  t_lo=None, t_hi=None, show_dist=False):
    n = len(y_true)
    correct = sum(a == b for a, b in zip(y_true, y_pred))
    acc = correct / n
    color = GRN if acc >= 0.65 else (YLW if acc >= 0.45 else RED)

    print(f"\n  {BLD}{label}{RST}")
    print(f"  Accuracy  {color}{BLD}{acc:.1%}{RST}  ({correct}/{n})")

    # Per-class
    parts = []
    for c in CLASSES:
        idx = [i for i, y in enumerate(y_true) if y == c]
        if not idx: continue
        hit = sum(y_pred[i] == c for i in idx)
        ca = hit / len(idx)
        parts.append(f"  {col(c)}{ORD_VALENCE[c]}{RST} {ca:.0%}(n={len(idx)})")
    print("  Per-class " + "  ".join(parts))

    # Confusion matrix
    header = f"  {DIM}{'':>6}{'neg':>6}{'neu':>6}{'pos':>6}  ← predicted{RST}"
    print(header)
    for t in CLASSES:
        row_counts = [sum(yt == t and yp == p for yt, yp in zip(y_true, y_pred)) for p in CLASSES]
        row_total = sum(yt == t for yt in y_true)
        if row_total == 0: continue
        cells = ""
        for p, cnt in zip(CLASSES, row_counts):
            cell_col = GRN if (t == p and cnt > 0) else (RED if cnt > 0 else DIM)
            cells += f"  {cell_col}{cnt:>4}{RST}"
        print(f"  {col(t)}{ORD_VALENCE[t]:>5}{RST}{cells}  (n={row_total})")

    # Optional: value distribution
    if show_dist and modality_vals and t_lo is not None:
        vals = [v for v in modality_vals if v is not None]
        mn, mx = min(vals), max(vals)
        span = mx - mn or 1
        print(f"\n  {DIM}Value distribution  [min={mn:.3f}  max={mx:.3f}  mean={np.mean(vals):.3f}]{RST}")
        bins = 20
        hist, edges = np.histogram(vals, bins=bins, range=(mn, mx))
        peak = max(hist)
        for i in range(bins):
            lo, hi = edges[i], edges[i+1]
            mid = (lo + hi) / 2
            bucket_col = RED if mid < t_lo else (YLW if mid < t_hi else GRN)
            bar_w = int(hist[i] / peak * 25) if peak else 0
            marker = "◄ neg/neu" if abs(mid - t_lo) < (mx - mn) / bins else (
                     "◄ neu/pos" if abs(mid - t_hi) < (mx - mn) / bins else "")
            print(f"  {bucket_col}{lo:5.3f}{RST} {'█'*bar_w:<25} {marker}")
        print()


def print_separator():
    print(f"\n  {DIM}{'─'*58}{RST}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BLD}Loading data...{RST}", end=" ", flush=True)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    task_map = {t["task_number"]: t for t in manifest["tasks"]}

    records = []
    if ANN_DIR.is_dir():
        for fp in sorted(ANN_DIR.glob("*.json")):
            if fp.stem.endswith(".bak"): continue
            with open(fp) as f:
                ann = json.load(f)
            for tn_str, entry in ann["annotations"].items():
                tn = int(tn_str)
                if tn not in task_map or entry.get("not_a_smile"): continue
                nv = entry.get("narrative_valence")
                sv = entry.get("speaker_valence")
                if nv not in VALENCE_ORD or sv not in VALENCE_ORD: continue
                task = task_map[tn]
                records.append({
                    "task_number": tn,
                    "video_id": task["video_id"],
                    "smile_start": task["smile_start"],
                    "smile_end": task["smile_end"],
                    "human_nv": VALENCE_ORD[nv],
                    "human_sv": VALENCE_ORD[sv],
                    "llm_nv": VALENCE_ORD.get(task.get("llm_narrative_valence", ""), 0),
                    "llm_sv": VALENCE_ORD.get(task.get("llm_speaker_valence", ""), 0),
                })

    if not records:
        print("No annotations found."); sys.exit(1)

    for r in records:
        r["audio_v"]   = audio_valence(r["video_id"], r["smile_start"], r["smile_end"])
        r["eyegaze_v"] = eyegaze_valence(r["video_id"], r["smile_start"], r["smile_end"])

    print(f"done  ({len(records)} annotations from "
          f"{len({r['video_id'] for r in records})} videos)\n")

    audio_vals   = [r["audio_v"]   for r in records]
    eyegaze_vals = [r["eyegaze_v"] for r in records]
    a_valid  = [r for r in records if r["audio_v"]   is not None]
    e_valid  = [r for r in records if r["eyegaze_v"] is not None]

    print(f"  Audio coverage:   {len(a_valid)}/{len(records)}  "
          f"range [{min(v for v in audio_vals if v):.3f}, "
          f"{max(v for v in audio_vals if v):.3f}]")
    print(f"  Eyegaze coverage: {len(e_valid)}/{len(records)}  "
          f"range [{min(v for v in eyegaze_vals if v):.3f}, "
          f"{max(v for v in eyegaze_vals if v):.3f}]")

    # Default thresholds from the best-fit analysis
    audio_t_lo, audio_t_hi     = 0.35, 0.45
    eyegaze_t_lo, eyegaze_t_hi = 0.256, 0.258

    show_dist = False

    def run_display():
        print_separator()
        print(f"\n  {BLD}Thresholds:{RST}"
              f"  audio  neg<{BLD}{audio_t_lo:.3f}{RST}≤neu<{BLD}{audio_t_hi:.3f}{RST}≤pos"
              f"  |  eyegaze  neg<{BLD}{eyegaze_t_lo:.3f}{RST}≤neu<{BLD}{eyegaze_t_hi:.3f}{RST}≤pos")

        for target_field, target_label in [("human_sv", "Speaker's Current Valence"),
                                            ("human_nv", "Narrative Valence")]:
            print(f"\n  {BLD}{CYN}── {target_label} ──{RST}")

            # LLM (reference, always shown)
            llm_field = "llm_sv" if target_field == "human_sv" else "llm_nv"
            y_true = [r[target_field] for r in records]
            y_llm  = [r[llm_field]    for r in records]
            print_results("LLM (reference)", y_true, y_llm)

            # Audio
            if a_valid:
                y_true_a = [r[target_field] for r in a_valid]
                y_audio  = [discretise(r["audio_v"], audio_t_lo, audio_t_hi) for r in a_valid]
                print_results("Audio VAD", y_true_a, y_audio,
                              modality_vals=[r["audio_v"] for r in a_valid],
                              t_lo=audio_t_lo, t_hi=audio_t_hi, show_dist=show_dist)

            # Eyegaze
            y_true_e = [r[target_field] for r in e_valid]
            y_eye    = [discretise(r["eyegaze_v"], eyegaze_t_lo, eyegaze_t_hi) for r in e_valid]
            print_results("Eyegaze VAD", y_true_e, y_eye,
                          modality_vals=[r["eyegaze_v"] for r in e_valid],
                          t_lo=eyegaze_t_lo, t_hi=eyegaze_t_hi, show_dist=show_dist)

        print_separator()

    run_display()

    print(f"""
{BLD}Commands:{RST}
  {GRN}a T1 T2{RST}       set audio thresholds     e.g. {DIM}a 0.35 0.45{RST}
  {GRN}e T1 T2{RST}       set eyegaze thresholds   e.g. {DIM}e 0.30 0.50{RST}
  {GRN}both A1 A2 E1 E2{RST}  set both at once
  {GRN}dist{RST}          toggle value histograms
  {GRN}q{RST}             quit
""")

    while True:
        try:
            raw = input(f"  {BLD}>{RST} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye."); break

        if not raw:
            run_display()
            continue

        parts = raw.split()
        cmd = parts[0]

        if cmd in ("q", "quit", "exit"):
            print("Bye."); break

        elif cmd == "dist":
            show_dist = not show_dist
            print(f"  Histograms {'ON' if show_dist else 'OFF'}")
            run_display()

        elif cmd == "a" and len(parts) == 3:
            try:
                audio_t_lo, audio_t_hi = float(parts[1]), float(parts[2])
                run_display()
            except ValueError:
                print("  Usage: a T1 T2  (e.g. a 0.35 0.45)")

        elif cmd == "e" and len(parts) == 3:
            try:
                eyegaze_t_lo, eyegaze_t_hi = float(parts[1]), float(parts[2])
                run_display()
            except ValueError:
                print("  Usage: e T1 T2  (e.g. e 0.30 0.50)")

        elif cmd == "both" and len(parts) == 5:
            try:
                audio_t_lo, audio_t_hi = float(parts[1]), float(parts[2])
                eyegaze_t_lo, eyegaze_t_hi = float(parts[3]), float(parts[4])
                run_display()
            except ValueError:
                print("  Usage: both A1 A2 E1 E2")

        else:
            # Try parsing as two bare numbers → update audio
            try:
                vals = [float(x) for x in parts[:2]]
                if len(vals) == 2:
                    audio_t_lo, audio_t_hi = vals
                    run_display()
            except ValueError:
                print(f"  Unknown command '{raw}'. Type q to quit.")


if __name__ == "__main__":
    main()
