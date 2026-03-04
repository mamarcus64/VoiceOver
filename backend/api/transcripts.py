import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

DATA_DIR = Path(os.environ.get(
    "VOICEOVER_DATA_DIR",
    Path(__file__).resolve().parent.parent.parent / "data",
))

router = APIRouter()


@router.get("/videos/{video_id:path}/transcript")
async def get_transcript(video_id: str):
    path = DATA_DIR / "transcripts_llm" / f"{video_id}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Transcript not found for video {video_id} (checked transcripts_llm)")
    with open(path) as f:
        return json.load(f)
