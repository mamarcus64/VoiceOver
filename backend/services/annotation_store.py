from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(
    os.environ.get(
        "VOICEOVER_DATA_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "data"),
    )
)

LABEL_BY_KEY: dict[int, str] = {
    1: "very_happy",
    2: "happy",
    3: "neutral",
    4: "sad",
    5: "very_sad",
}
VALID_LABELS: set[str] = set(LABEL_BY_KEY.values())


def _annotations_dir(video_id: str) -> Path:
    return DATA_DIR / "annotations" / video_id


def _ts_from_filename(name: str) -> str:
    """Extract the ISO-8601 timestamp portion from a filename like 'jordan_2026-03-01T12:00:00Z.json'."""
    stem = Path(name).stem
    return stem.split("_", 1)[1]


def _latest_file(video_id: str, annotator: str) -> Path | None:
    """Return the most recent annotation file for a video+annotator, or None."""
    d = _annotations_dir(video_id)
    if not d.is_dir():
        return None
    prefix = f"{annotator}_"
    candidates = sorted(
        (f for f in d.iterdir() if f.name.startswith(prefix) and f.suffix == ".json"),
        key=lambda f: f.stem.split("_", 1)[1],
    )
    return candidates[-1] if candidates else None


def save(video_id: str, annotator: str, annotations_list: list[dict[str, Any]]) -> dict:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    obj = {
        "video_id": video_id,
        "annotator": annotator,
        "created_at": now,
        "annotations": annotations_list,
    }

    d = _annotations_dir(video_id)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{annotator}_{now}.json"
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return obj


def load(video_id: str, annotator: str) -> dict | None:
    path = _latest_file(video_id, annotator)
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_all(video_id: str) -> list[dict]:
    annotators = list_annotators(video_id)
    results: list[dict] = []
    for ann in annotators:
        data = load(video_id, ann)
        if data is not None:
            results.append(data)
    return results


def list_annotators(video_id: str) -> list[str]:
    d = _annotations_dir(video_id)
    if not d.is_dir():
        return []
    seen: set[str] = set()
    for f in d.iterdir():
        if f.suffix == ".json":
            annotator = f.stem.split("_", 1)[0]
            seen.add(annotator)
    return sorted(seen)
