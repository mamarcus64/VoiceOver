import csv
import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

DATA_DIR = Path(os.environ.get(
    "VOICEOVER_DATA_DIR",
    Path(__file__).resolve().parent.parent.parent / "data",
))

router = APIRouter()


@router.get("/videos/{video_id:path}/audio-emotion")
async def get_audio_emotion(video_id: str):
    path = DATA_DIR / "audio_vad" / f"{video_id}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Audio emotion data not found for video {video_id}")
    with open(path) as f:
        return json.load(f)


@router.get("/videos/{video_id:path}/eyegaze-emotion")
async def get_eyegaze_emotion(video_id: str):
    path = DATA_DIR / "eyegaze_vad" / f"{video_id}.csv"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Eyegaze emotion data not found for video {video_id}")
    segments = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            segments.append({
                "timestamp": float(row["timestamp"]),
                "valence": float(row["valence"]),
                "arousal": float(row["arousal"]),
                "dominance": float(row["dominance"]),
            })
    return {"video_id": video_id, "segments": segments}
