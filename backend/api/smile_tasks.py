from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

DATA_DIR = Path(
    os.environ.get(
        "VOICEOVER_DATA_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "data"),
    )
)

MANIFEST_PATH = DATA_DIR / "smile_task_manifest.json"
ANNOTATIONS_DIR = DATA_DIR / "smile_annotations"
VIDEO_DIR = DATA_DIR / "videos"
FALLBACK_VIDEO_DIR = Path(os.environ.get("VOICEOVER_VIDEO_FALLBACK", "/home/mjma/voices/test_data/videos"))

router = APIRouter()


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise HTTPException(status_code=404, detail="Task manifest not generated yet. Use the config page to generate it.")
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def _video_is_downloaded(video_id: str) -> bool:
    mp4 = f"{video_id}.mp4"
    return (VIDEO_DIR / mp4).is_file() or (FALLBACK_VIDEO_DIR / mp4).is_file()


def _downloaded_video_ids() -> set[str]:
    """Return set of all video IDs that have a downloaded .mp4."""
    ids: set[str] = set()
    for d in (VIDEO_DIR, FALLBACK_VIDEO_DIR):
        if d.is_dir():
            for f in d.iterdir():
                if f.suffix == ".mp4":
                    ids.add(f.stem)
    return ids


def _count_available_tasks(manifest: dict[str, Any]) -> int:
    downloaded = _downloaded_video_ids()
    return sum(1 for t in manifest["tasks"] if t["video_id"] in downloaded)


def _annotations_path(annotator: str) -> Path:
    return ANNOTATIONS_DIR / f"{annotator}.json"


def _load_annotations(annotator: str) -> dict[str, Any]:
    path = _annotations_path(annotator)
    if path.is_file():
        with open(path) as f:
            return json.load(f)
    return {"annotator": annotator, "annotations": {}}


def _save_annotations(annotator: str, data: dict[str, Any]) -> None:
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = _annotations_path(annotator)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


@router.get("/smile-tasks/count")
async def task_count():
    manifest = _load_manifest()
    return {
        "total_tasks": manifest["total_tasks"],
        "available_tasks": _count_available_tasks(manifest),
    }


@router.get("/smile-tasks/next-incomplete")
async def next_incomplete(annotator: str = Query(...)):
    manifest = _load_manifest()
    ann_data = _load_annotations(annotator)
    completed = ann_data.get("annotations", {})
    for task in manifest["tasks"]:
        if str(task["task_number"]) not in completed:
            return {"task_number": task["task_number"]}
    return {"task_number": None}


@router.get("/smile-tasks/{task_number}")
async def get_task(task_number: int):
    manifest = _load_manifest()
    if task_number < 1 or task_number > len(manifest["tasks"]):
        raise HTTPException(status_code=404, detail=f"Task {task_number} not found")
    task = manifest["tasks"][task_number - 1]
    return {
        **task,
        "total_tasks": manifest["total_tasks"],
        "available_tasks": _count_available_tasks(manifest),
        "video_downloaded": _video_is_downloaded(task["video_id"]),
    }


@router.get("/smile-annotations/{annotator}")
async def get_annotations(annotator: str):
    return _load_annotations(annotator)


class AnnotateBody(BaseModel):
    annotator: str
    task_number: int
    label: str
    notes: str = ""
    runner_up: str = ""


VALID_LABELS = {"genuine", "polite", "masking", "not_a_smile"}


@router.post("/smile-annotations")
async def save_annotation(body: AnnotateBody):
    if body.label not in VALID_LABELS:
        raise HTTPException(status_code=400, detail=f"Invalid label '{body.label}'. Must be one of: {', '.join(sorted(VALID_LABELS))}")

    manifest = _load_manifest()
    if body.task_number < 1 or body.task_number > len(manifest["tasks"]):
        raise HTTPException(status_code=404, detail=f"Task {body.task_number} not found")

    data = _load_annotations(body.annotator)
    entry: dict[str, Any] = {
        "label": body.label,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if body.notes.strip():
        entry["notes"] = body.notes.strip()
    if body.runner_up:
        if body.runner_up not in VALID_LABELS:
            raise HTTPException(status_code=400, detail=f"Invalid runner_up '{body.runner_up}'")
        entry["runner_up"] = body.runner_up
    data["annotations"][str(body.task_number)] = entry
    _save_annotations(body.annotator, data)
    return {"ok": True, "task_number": body.task_number, "label": body.label}
