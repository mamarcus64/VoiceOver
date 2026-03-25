"""Read-only access to archived pilot smile annotation data."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

DATA_DIR = Path(
    os.environ.get(
        "VOICEOVER_DATA_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "data"),
    )
)

MANIFEST_PATH = DATA_DIR / "pilot_smile_task_manifest.json"
ANNOTATIONS_DIR = DATA_DIR / "pilot_smile_annotations"
VIDEO_DIR = DATA_DIR / "videos"
FALLBACK_VIDEO_DIR = Path(os.environ.get("VOICEOVER_VIDEO_FALLBACK", "/home/mjma/voices/test_data/videos"))

router = APIRouter()


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise HTTPException(status_code=404, detail="Pilot manifest not found.")
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def _video_is_downloaded(video_id: str) -> bool:
    mp4 = f"{video_id}.mp4"
    return (VIDEO_DIR / mp4).is_file() or (FALLBACK_VIDEO_DIR / mp4).is_file()


def _downloaded_video_ids() -> set[str]:
    ids: set[str] = set()
    for d in (VIDEO_DIR, FALLBACK_VIDEO_DIR):
        if d.is_dir():
            for f in d.iterdir():
                if f.suffix == ".mp4":
                    ids.add(f.stem)
    return ids


def _load_annotations(annotator: str) -> dict[str, Any]:
    path = ANNOTATIONS_DIR / f"{annotator}.json"
    if path.is_file():
        with open(path) as f:
            return json.load(f)
    return {"annotator": annotator, "annotations": {}}


@router.get("/pilot-smile-tasks/next-incomplete")
async def next_incomplete(annotator: str = Query(...)):
    manifest = _load_manifest()
    ann_data = _load_annotations(annotator)
    completed = ann_data.get("annotations", {})
    for task in manifest["tasks"]:
        if str(task["task_number"]) not in completed:
            return {"task_number": task["task_number"]}
    return {"task_number": None}


@router.get("/pilot-smile-tasks/count")
async def task_count():
    manifest = _load_manifest()
    downloaded = _downloaded_video_ids()
    return {
        "total_tasks": manifest["total_tasks"],
        "available_tasks": sum(1 for t in manifest["tasks"] if t["video_id"] in downloaded),
    }


@router.get("/pilot-smile-tasks/{task_number}")
async def get_task(task_number: int):
    manifest = _load_manifest()
    if task_number < 1 or task_number > len(manifest["tasks"]):
        raise HTTPException(status_code=404, detail=f"Task {task_number} not found")
    task = manifest["tasks"][task_number - 1]
    downloaded = _downloaded_video_ids()
    return {
        **task,
        "total_tasks": manifest["total_tasks"],
        "available_tasks": sum(1 for t in manifest["tasks"] if t["video_id"] in downloaded),
        "video_downloaded": _video_is_downloaded(task["video_id"]),
    }


@router.get("/pilot-smile-annotations/{annotator}")
async def get_annotations(annotator: str):
    return _load_annotations(annotator)
