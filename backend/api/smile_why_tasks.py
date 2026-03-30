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

MANIFEST_PATH = DATA_DIR / "smile_why_task_manifest.json"
ANNOTATIONS_DIR = DATA_DIR / "smile_why_annotations"
VIDEO_DIR = DATA_DIR / "videos"
FALLBACK_VIDEO_DIR = Path(
    os.environ.get("VOICEOVER_VIDEO_FALLBACK", "/home/mjma/voices/test_data/videos")
)

router = APIRouter()


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        return {"study": "smile_why", "total_tasks": 0, "tasks": []}
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
    with open(_annotations_path(annotator), "w") as f:
        json.dump(data, f, indent=2)


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get("/smile-why-tasks/count")
async def count_tasks():
    manifest = _load_manifest()
    return {
        "total_tasks": manifest["total_tasks"],
        "available_tasks": _count_available_tasks(manifest),
    }


@router.get("/smile-why-tasks/next-incomplete")
async def next_incomplete(annotator: str = Query(...)):
    manifest = _load_manifest()
    ann_data = _load_annotations(annotator)
    completed = ann_data.get("annotations", {})
    for task in manifest["tasks"]:
        if str(task["task_number"]) not in completed:
            return {"task_number": task["task_number"]}
    return {"task_number": None}


@router.get("/smile-why-tasks/{task_number}")
async def get_task(task_number: int):
    manifest = _load_manifest()
    task_map = {t["task_number"]: t for t in manifest["tasks"]}
    if task_number not in task_map:
        raise HTTPException(status_code=404, detail=f"Task {task_number} not found")
    task = task_map[task_number]
    return {
        "task_number": task["task_number"],
        "video_id": task["video_id"],
        "smile_start": task["smile_start"],
        "smile_end": task["smile_end"],
        "score": task.get("score"),
        "stratum": task.get("stratum"),
        "score_tier": task.get("score_tier"),
        "total_tasks": manifest["total_tasks"],
        "available_tasks": _count_available_tasks(manifest),
        "video_downloaded": _video_is_downloaded(task["video_id"]),
    }


@router.get("/smile-why-annotations/{annotator}")
async def get_annotations(annotator: str):
    return _load_annotations(annotator)


class SmileWhyAnnotateBody(BaseModel):
    annotator: str
    task_number: int
    response: str = ""
    not_a_smile: bool = False


@router.post("/smile-why-annotations")
async def save_annotation(body: SmileWhyAnnotateBody):
    if not body.response.strip() and not body.not_a_smile:
        raise HTTPException(status_code=400, detail="Either provide a response or mark as not_a_smile")

    manifest = _load_manifest()
    task_numbers = {t["task_number"] for t in manifest["tasks"]}
    if body.task_number not in task_numbers:
        raise HTTPException(status_code=404, detail=f"Task {body.task_number} not found")

    data = _load_annotations(body.annotator)
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if body.not_a_smile:
        entry["not_a_smile"] = True
    if body.response.strip():
        entry["response"] = body.response.strip()

    data["annotations"][str(body.task_number)] = entry
    _save_annotations(body.annotator, data)

    return {"ok": True, "task_number": body.task_number}


@router.get("/smile-why-results")
async def get_results():
    """Return all annotations across all annotators for the results page."""
    manifest = _load_manifest()
    task_map = {t["task_number"]: t for t in manifest["tasks"]}

    results: list[dict[str, Any]] = []
    if ANNOTATIONS_DIR.is_dir():
        for f in sorted(ANNOTATIONS_DIR.glob("*.json")):
            try:
                with open(f) as fp:
                    ann_data = json.load(fp)
            except Exception:
                continue
            annotator = ann_data.get("annotator", f.stem)
            for task_key, entry in ann_data.get("annotations", {}).items():
                task_num = int(task_key)
                task = task_map.get(task_num, {})
                results.append({
                    "task_number": task_num,
                    "annotator": annotator,
                    "response": entry.get("response", ""),
                    "not_a_smile": entry.get("not_a_smile", False),
                    "timestamp": entry.get("timestamp", ""),
                    "video_id": task.get("video_id", ""),
                    "smile_start": task.get("smile_start"),
                    "smile_end": task.get("smile_end"),
                    "stratum": task.get("stratum", ""),
                    "score_tier": task.get("score_tier", ""),
                })

    results.sort(key=lambda r: (r["task_number"], r["annotator"]))
    return {
        "total_annotations": len(results),
        "total_tasks": manifest["total_tasks"],
        "results": results,
    }
