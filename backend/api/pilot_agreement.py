"""Read-only pilot agreement stats, backed by pilot_smile_annotations/."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from api.smile_agreement import (
    VALID_LABELS,
    LABEL_INDEX,
    COARSE_LABELS,
    _COARSE_MAP,
    MODES,
    _effective_label,
    _compute_mode,
    _fleiss_kappa,
    _cohen_kappa,
    _empty_confusion,
    _fine_to_coarse_confusion,
)

DATA_DIR = Path(
    os.environ.get(
        "VOICEOVER_DATA_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "data"),
    )
)
PILOT_ANNOTATIONS_DIR = DATA_DIR / "pilot_smile_annotations"
PILOT_MANIFEST_PATH = DATA_DIR / "pilot_smile_task_manifest.json"

router = APIRouter()


def _list_pilot_annotators() -> list[str]:
    return sorted(p.stem for p in PILOT_ANNOTATIONS_DIR.glob("*.json"))


def _load_pilot_annotations(annotator: str) -> dict[str, Any]:
    path = PILOT_ANNOTATIONS_DIR / f"{annotator}.json"
    if path.is_file():
        with open(path) as f:
            return json.load(f)
    return {"annotator": annotator, "annotations": {}}


def _pilot_labels_for_tasks(annotators: list[str]) -> dict[str, dict[str, str]]:
    by_task: dict[str, dict[str, str]] = {}
    for name in annotators:
        data = _load_pilot_annotations(name)
        for task_key, entry in data.get("annotations", {}).items():
            eff = _effective_label(entry)
            if eff is not None:
                by_task.setdefault(task_key, {})[name] = eff
    return by_task


@router.get("/pilot-smile-agreement/annotators")
async def pilot_annotators():
    return _list_pilot_annotators()


@router.get("/pilot-smile-agreement/stats")
async def pilot_stats(
    annotators: str = Query(..., description="Comma-separated annotator names"),
):
    names = [s.strip() for s in annotators.split(",") if s.strip()]
    known = set(_list_pilot_annotators())
    for n in names:
        if n not in known:
            raise HTTPException(status_code=400, detail=f"Unknown pilot annotator '{n}'")

    by_task = _pilot_labels_for_tasks(names)

    per_annotator_counts = {}
    for name in names:
        data = _load_pilot_annotations(name)
        per_annotator_counts[name] = len(data.get("annotations", {}))

    multi = {tk: labs for tk, labs in by_task.items() if len(labs) >= 2}
    fully_labeled = {tk: labs for tk, labs in multi.items() if len(labs) == len(names)}

    k = len(VALID_LABELS)
    fleiss = None
    if multi:
        counts_per_subject = []
        for labs in multi.values():
            row = [0] * k
            for lab in labs.values():
                idx = LABEL_INDEX.get(lab)
                if idx is not None:
                    row[idx] += 1
            counts_per_subject.append(row)
        fleiss = _fleiss_kappa(counts_per_subject)

    modes = {key: _compute_mode(key, by_task, names) for key in MODES}

    return {
        "annotators": names,
        "valid_labels": list(VALID_LABELS),
        "coarse_labels": list(COARSE_LABELS),
        "per_annotator_counts": per_annotator_counts,
        "tasks_with_any_label": len(by_task),
        "modes": modes,
        "fleiss_kappa": fleiss,
    }


@router.get("/pilot-smile-agreement/au12-scatter")
async def pilot_au12_scatter(
    annotators: str = Query(..., description="Comma-separated annotator names"),
):
    names = [s.strip() for s in annotators.split(",") if s.strip()]
    if not PILOT_MANIFEST_PATH.is_file():
        raise HTTPException(status_code=500, detail="Pilot manifest not found")
    with open(PILOT_MANIFEST_PATH) as f:
        manifest = json.load(f)
    task_lookup = {str(t["task_number"]): t for t in manifest.get("tasks", [])}

    points: list[dict[str, Any]] = []
    for name in names:
        data = _load_pilot_annotations(name)
        for task_key, entry in data.get("annotations", {}).items():
            eff = _effective_label(entry)
            if eff is None:
                continue
            task_info = task_lookup.get(task_key)
            if task_info is None:
                continue
            points.append({
                "task_number": int(task_key),
                "annotator": name,
                "mean_r": task_info["mean_r"],
                "peak_r": task_info["peak_r"],
                "is_not_a_smile": eff == "not_a_smile",
                "label": eff,
            })
    return {"points": points}
