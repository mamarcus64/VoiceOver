"""Read-only pilot agreement stats, backed by pilot_smile_annotations/.

Uses the original pilot-study labels: genuine / polite / masking / not_a_smile.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from api.smile_agreement import _fleiss_kappa, _cohen_kappa

# ---------------------------------------------------------------------------
# Pilot-study label constants (genuine / polite / masking)
# These are separate from smile_agreement.py, which uses the main-study labels.
# ---------------------------------------------------------------------------
VALID_LABELS = ("genuine", "polite", "masking", "not_a_smile")
LABEL_INDEX = {lab: i for i, lab in enumerate(VALID_LABELS)}

COARSE_LABELS = ("positive", "masking", "not_a_smile")
_COARSE_MAP = (0, 0, 1, 2)

MODES: dict[str, dict[str, Any]] = {
    "fine": {
        "labels": ["genuine", "polite", "masking", "not_a_smile"],
        "map": [0, 1, 2, 3],
        "filter_nas": False,
    },
    "coarse": {
        "labels": ["positive", "masking", "not_a_smile"],
        "map": [0, 0, 1, 2],
        "filter_nas": False,
    },
    "smile_fine": {
        "labels": ["genuine", "polite", "masking"],
        "map": [0, 1, 2, -1],
        "filter_nas": True,
    },
    "smile_coarse": {
        "labels": ["positive", "masking"],
        "map": [0, 0, 1, -1],
        "filter_nas": True,
    },
    "binary": {
        "labels": ["smile", "not_a_smile"],
        "map": [0, 0, 0, 1],
        "filter_nas": False,
    },
}


def _effective_label(entry: dict[str, Any]) -> str | None:
    """Derive the agreement-level label from a pilot annotation entry."""
    label = entry.get("label")
    if label == "not_a_smile":
        return "not_a_smile"
    if entry.get("not_a_smile"):
        return "not_a_smile"
    if label in VALID_LABELS:
        return label
    return None


def _compute_mode(
    mode_key: str,
    by_task: dict[str, dict[str, str]],
    annotators: list[str],
) -> dict[str, Any]:
    mdef = MODES[mode_key]
    mode_labels: list[str] = mdef["labels"]
    label_map: list[int] = mdef["map"]
    filter_nas: bool = mdef["filter_nas"]
    k = len(mode_labels)

    def _map_label(lab: str) -> int | None:
        fi = LABEL_INDEX.get(lab)
        if fi is None:
            return None
        mi = label_map[fi]
        if mi == -1:
            return None
        if filter_nas and lab == "not_a_smile":
            return None
        return mi

    fully_labeled: list[str] = []
    for task_key, labs in by_task.items():
        if not all(a in labs for a in annotators):
            continue
        if all(_map_label(labs[a]) is not None for a in annotators):
            fully_labeled.append(task_key)

    fleiss: float | None = None
    pct_full: float | None = None
    if len(annotators) >= 2 and fully_labeled:
        count_matrix: list[list[int]] = []
        agree = 0
        for task_key in fully_labeled:
            labs = by_task[task_key]
            row = [0] * k
            mapped_set: set[int] = set()
            for a in annotators:
                mi = _map_label(labs[a])
                assert mi is not None
                row[mi] += 1
                mapped_set.add(mi)
            count_matrix.append(row)
            if len(mapped_set) == 1:
                agree += 1
        pct_full = 100.0 * agree / len(fully_labeled)
        fleiss = _fleiss_kappa(count_matrix)

    pairwise: list[dict[str, Any]] = []
    for i, a in enumerate(annotators):
        for b in annotators[i + 1:]:
            conf = [[0] * k for _ in range(k)]
            n_both = 0
            for task_key, labs in by_task.items():
                if a not in labs or b not in labs:
                    continue
                mi_a = _map_label(labs[a])
                mi_b = _map_label(labs[b])
                if mi_a is None or mi_b is None:
                    continue
                n_both += 1
                conf[mi_a][mi_b] += 1
            if n_both == 0:
                pairwise.append({
                    "annotator_a": a,
                    "annotator_b": b,
                    "n_tasks": 0,
                    "cohen_kappa": None,
                    "percent_agreement": None,
                    "confusion": conf,
                })
            else:
                agree_pair = sum(conf[j][j] for j in range(k))
                pct = 100.0 * agree_pair / n_both
                kap = _cohen_kappa(conf)
                pairwise.append({
                    "annotator_a": a,
                    "annotator_b": b,
                    "n_tasks": n_both,
                    "cohen_kappa": kap,
                    "percent_agreement": pct,
                    "confusion": conf,
                })

    return {
        "labels": mode_labels,
        "tasks_fully_labeled": len(fully_labeled),
        "fleiss_kappa": fleiss,
        "percent_full_agreement": pct_full,
        "pairwise": pairwise,
    }

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
    return {"annotators": _list_pilot_annotators()}


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

    per_annotator_counts: dict[str, dict[str, int]] = {
        a: {lab: 0 for lab in VALID_LABELS} for a in names
    }
    for _task, labs in by_task.items():
        for a, lab in labs.items():
            if a in per_annotator_counts:
                per_annotator_counts[a][lab] += 1

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
