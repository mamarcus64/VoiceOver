#!/usr/bin/env python3
"""Generate the smile-valence annotation task manifest.

Filters pipeline:
  1. score >= threshold (from detected_smiles.json)
  2. score >= 95th percentile of remaining smiles
  3. Video has a downloaded .mp4
  4. Smile window overlaps with at least one LLM-annotated sentence in
     llm_annotated_eyegaze/{video_id}.json with a valid valence label
  5. Per-video cap of MAX_PER_VIDEO (applied in score-descending order)

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
LLM_EYEGAZE_DIR = DATA / "llm_annotated_eyegaze"
OUT_PATH = DATA / "smile_valence_task_manifest.json"

MAX_PER_VIDEO = 10
MAX_TASKS = None  # set to an int to hard-cap total tasks
VALENCE_LABELS = {"negative", "neutral", "positive"}


def downloaded_video_ids() -> set[str]:
    ids: set[str] = set()
    for d in (VIDEO_DIR, FALLBACK_VIDEO_DIR):
        if d.is_dir():
            for f in d.iterdir():
                if f.suffix == ".mp4":
                    ids.add(f.stem)
    return ids


def _best_sentence(sentences: list[dict], start_ms: float, end_ms: float) -> dict | None:
    """Return the sentence with the most temporal overlap with [start_ms, end_ms]."""
    smile_mid = (start_ms + end_ms) / 2
    best = None
    best_score: tuple[float, float] = (-1.0, float("inf"))
    for s in sentences:
        fw = s.get("first_word_ms")
        lw = s.get("last_word_ms")
        if fw is None or lw is None:
            continue
        fw, lw = float(fw), float(lw)
        overlap = min(end_ms, lw) - max(start_ms, fw)
        if overlap < 0:
            continue
        # Point-sentences (fw == lw) that fall inside the window
        if overlap == 0 and fw >= start_ms and lw <= end_ms:
            overlap = 1.0
        if overlap <= 0:
            continue
        mid_dist = abs((fw + lw) / 2 - smile_mid)
        score: tuple[float, float] = (overlap, -mid_dist)
        if score > best_score:
            best_score = score
            best = s
    return best


def build_llm_coverage(video_ids: set[str]) -> dict[str, list[dict]]:
    """Load sentences for each video that has an eyegaze file. Returns {video_id: sentences}."""
    print("  Loading LLM eyegaze sentences ...")
    coverage: dict[str, list[dict]] = {}
    for vid in video_ids:
        fp = LLM_EYEGAZE_DIR / f"{vid}.json"
        if not fp.is_file():
            continue
        try:
            with open(fp) as f:
                data = json.load(f)
            coverage[vid] = data.get("sentences", [])
        except Exception:
            pass
    print(f"  {len(coverage):,} videos have LLM eyegaze annotations")
    return coverage


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

    # 1. Filter to at-or-above threshold
    smiles = [s for s in smiles if s["score"] >= threshold]
    print(f"  {len(smiles):,} at or above threshold")

    # 2. Keep only 95th percentile and above
    p95 = float(np.percentile([s["score"] for s in smiles], 95))
    smiles = [s for s in smiles if s["score"] >= p95]
    print(f"  {len(smiles):,} at or above 95th percentile (score >= {p95:.4f})")

    # Sort by score descending
    smiles.sort(key=lambda s: s["score"], reverse=True)

    # 3. Restrict to downloaded videos
    available = downloaded_video_ids()
    print(f"  {len(available):,} downloaded videos")
    smiles = [s for s in smiles if s["video_id"] in available]
    print(f"  {len(smiles):,} smiles in downloaded videos")

    # 4. Load LLM coverage for all candidate videos
    candidate_videos = {s["video_id"] for s in smiles}
    llm_sentences = build_llm_coverage(candidate_videos)
    llm_videos = set(llm_sentences.keys())

    # 5. Build task list: walk sorted smiles, require LLM match, apply per-video cap
    per_video_count: dict[str, int] = {}
    tasks = []
    task_number = 1
    skipped_no_llm = 0
    skipped_cap = 0

    for smile in smiles:
        vid = smile["video_id"]

        if vid not in llm_videos:
            skipped_no_llm += 1
            continue

        count = per_video_count.get(vid, 0)
        if count >= MAX_PER_VIDEO:
            skipped_cap += 1
            continue

        start_ms = smile["start_ts"] * 1000
        end_ms = smile["end_ts"] * 1000
        sent = _best_sentence(llm_sentences[vid], start_ms, end_ms)

        if sent is None:
            skipped_no_llm += 1
            continue

        nv = sent.get("narrative_valence")
        sv = sent.get("present_day_valence")
        if nv not in VALENCE_LABELS or sv not in VALENCE_LABELS:
            skipped_no_llm += 1
            continue

        per_video_count[vid] = count + 1
        tasks.append({
            "task_number": task_number,
            "video_id": vid,
            "smile_start": smile["start_ts"],
            "smile_end": smile["end_ts"],
            "score": round(smile["score"], 4),
            "peak_r": round(smile.get("peak_r", 0), 4),
            # Embed the LLM labels for quick alignment lookup
            "llm_narrative_valence": nv,
            "llm_speaker_valence": sv,
        })
        task_number += 1

        if MAX_TASKS is not None and task_number > MAX_TASKS:
            break

    print(f"  {skipped_no_llm:,} smiles skipped (no LLM match)")
    print(f"  {skipped_cap:,} smiles skipped (per-video cap)")
    print(f"  {len(tasks):,} final tasks across {len(per_video_count):,} videos")

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
            "llm_coverage_videos": len(llm_videos & candidate_videos),
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
