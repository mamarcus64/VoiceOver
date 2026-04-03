"""API endpoint that selects contrasting smile pairs for gaze-direction verification.

For a set of subjects, returns pairs of smiles: one where the subject was
looking at the interviewer and one where they were not.  The frontend renders
these side-by-side so a human can visually confirm the interviewer-position
estimate makes sense.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from fastapi import APIRouter, Query

from services.video_manager import video_manager

router = APIRouter()

DATA = Path(video_manager.data_dir)
GAZE_PATH = DATA / "smile_gaze_features.csv"
SMILES_PATH = DATA / "detected_smiles.json"
POSITIONS_PATH = DATA / "interviewer_positions.csv"

MIN_SMILE_DURATION = 0.8
MIN_GAZE_FRAMES = 8


def _load_gaze_index() -> dict[tuple[str, float], dict]:
    """Return {(video_id, start_ts): row_dict} from smile_gaze_features.csv."""
    idx = {}
    with open(GAZE_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if row["gaze_available"] != "True":
                continue
            key = (row["video_id"], float(row["start_ts"]))
            idx[key] = row
    return idx


def _load_interviewer_positions() -> dict[str, dict]:
    pos = {}
    with open(POSITIONS_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] == "none" or not row["interviewer_angle_x"]:
                continue
            pos[row["video_id"]] = {
                "angle_x": float(row["interviewer_angle_x"]),
                "angle_y": float(row["interviewer_angle_y"]),
            }
    return pos


def _select_pairs(n_subjects: int = 10, seed: int = 42):
    with open(SMILES_PATH) as f:
        all_smiles = json.load(f)["smiles"]

    gaze_idx = _load_gaze_index()
    positions = _load_interviewer_positions()

    downloaded = set()
    for vid_id in {s["video_id"] for s in all_smiles}:
        if video_manager.get_video_path(vid_id) is not None:
            downloaded.add(vid_id)

    by_subject: dict[str, list[dict]] = {}
    for s in all_smiles:
        vid = s["video_id"]
        if vid not in downloaded or vid not in positions:
            continue
        dur = s["end_ts"] - s["start_ts"]
        if dur < MIN_SMILE_DURATION:
            continue
        key = (vid, s["start_ts"])
        gaze = gaze_idx.get(key)
        if not gaze or int(gaze["n_gaze_frames"]) < MIN_GAZE_FRAMES:
            continue

        subj = vid.split(".")[0]
        by_subject.setdefault(subj, []).append({
            "video_id": vid,
            "start_ts": s["start_ts"],
            "end_ts": s["end_ts"],
            "score": s.get("score", 0),
            "frac_looking": float(gaze["frac_looking_at_interviewer"]),
            "deviation_euc": float(gaze["gaze_deviation_euc"]),
            "deviation_x": float(gaze["gaze_deviation_x"]),
            "deviation_y": float(gaze["gaze_deviation_y"]),
        })

    eligible_subjects = [
        subj for subj, smiles in by_subject.items()
        if any(s["frac_looking"] >= 0.8 for s in smiles)
        and any(s["frac_looking"] <= 0.2 for s in smiles)
    ]

    rng = random.Random(seed)
    rng.shuffle(eligible_subjects)
    selected = eligible_subjects[:n_subjects]

    pairs = []
    for subj in sorted(selected):
        smiles = by_subject[subj]
        looking = [s for s in smiles if s["frac_looking"] >= 0.8]
        away = [s for s in smiles if s["frac_looking"] <= 0.2]
        looking.sort(key=lambda s: s["score"], reverse=True)
        away.sort(key=lambda s: s["deviation_euc"], reverse=True)
        pos = positions.get(looking[0]["video_id"], {"angle_x": 0, "angle_y": 0})
        pairs.append({
            "subject_id": subj,
            "interviewer_angle_x": pos["angle_x"],
            "interviewer_angle_y": pos["angle_y"],
            "looking_at": looking[0],
            "looking_away": away[0],
        })

    return pairs


@router.get("/gaze-verify-pairs")
async def gaze_verify_pairs(
    n: int = Query(10, ge=1, le=50),
    seed: int = Query(42),
):
    if not GAZE_PATH.exists() or not SMILES_PATH.exists():
        return {"pairs": [], "error": "Required data files not found."}

    pairs = _select_pairs(n_subjects=n, seed=seed)
    return {"pairs": pairs, "n": len(pairs)}
