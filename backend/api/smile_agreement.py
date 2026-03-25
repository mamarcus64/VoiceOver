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
ANNOTATIONS_DIR = DATA_DIR / "smile_annotations"

router = APIRouter()

VALID_LABELS = ("felt", "false", "miserable", "not_a_smile")
LABEL_INDEX = {lab: i for i, lab in enumerate(VALID_LABELS)}

# Coarse grouping: felt+false → "positive"; miserable and not_a_smile unchanged
COARSE_LABELS = ("positive", "miserable", "not_a_smile")
_COARSE_MAP = (0, 0, 1, 2)  # maps fine label index → coarse label index

# ---------------------------------------------------------------------------
# Mode definitions for the 2×2 toggle grid + binary
#   map[i] converts VALID_LABELS[i] → mode label index (-1 = exclude)
#   filter_nas: when True, exclude tasks where any relevant annotator said NaS
# ---------------------------------------------------------------------------
MODES: dict[str, dict[str, Any]] = {
    "fine": {
        "labels": ["felt", "false", "miserable", "not_a_smile"],
        "map": [0, 1, 2, 3],
        "filter_nas": False,
    },
    "coarse": {
        "labels": ["positive", "miserable", "not_a_smile"],
        "map": [0, 0, 1, 2],
        "filter_nas": False,
    },
    "smile_fine": {
        "labels": ["felt", "false", "miserable"],
        "map": [0, 1, 2, -1],
        "filter_nas": True,
    },
    "smile_coarse": {
        "labels": ["positive", "miserable"],
        "map": [0, 0, 1, -1],
        "filter_nas": True,
    },
    "binary": {
        "labels": ["smile", "not_a_smile"],
        "map": [0, 0, 0, 1],
        "filter_nas": False,
    },
}


def _list_annotator_names() -> list[str]:
    if not ANNOTATIONS_DIR.is_dir():
        return []
    return sorted(p.stem for p in ANNOTATIONS_DIR.glob("*.json"))


def _load_annotations(annotator: str) -> dict[str, Any]:
    path = ANNOTATIONS_DIR / f"{annotator}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No annotations file for '{annotator}'")
    with open(path) as f:
        return json.load(f)


def _effective_label(entry: dict[str, Any]) -> str | None:
    """Derive the agreement-level label from an annotation entry.

    New format: not_a_smile is a boolean flag; label is always an emotion.
    Old format: label may be "not_a_smile" directly.
    """
    label = entry.get("label")
    if label == "not_a_smile":
        return "not_a_smile"
    if entry.get("not_a_smile"):
        return "not_a_smile"
    if label in VALID_LABELS:
        return label
    return None


def _labels_for_tasks(annotators: list[str]) -> dict[str, dict[str, str]]:
    """task_number -> annotator -> effective label (only valid labels)."""
    out: dict[str, dict[str, str]] = {}
    for name in annotators:
        data = _load_annotations(name)
        for task_key, entry in data.get("annotations", {}).items():
            eff = _effective_label(entry)
            if eff is None:
                continue
            out.setdefault(task_key, {})[name] = eff
    return out


def _fleiss_kappa(counts_per_subject: list[list[int]]) -> float | None:
    """counts_per_subject[i][j] = raters assigning category j to subject i. Fixed n raters."""
    if not counts_per_subject:
        return None
    n = sum(counts_per_subject[0])
    if n < 2:
        return None
    k = len(counts_per_subject[0])
    N = len(counts_per_subject)
    for row in counts_per_subject:
        if sum(row) != n:
            return None

    P_parts: list[float] = []
    for row in counts_per_subject:
        acc = sum(c * (c - 1) for c in row)
        P_parts.append(acc / (n * (n - 1)))
    P_bar = sum(P_parts) / N

    col_totals = [sum(counts_per_subject[i][j] for i in range(N)) for j in range(k)]
    p_j = [ct / (N * n) for ct in col_totals]
    P_e = sum(p * p for p in p_j)
    if P_e >= 1.0 - 1e-12:
        return None
    kappa = (P_bar - P_e) / (1.0 - P_e)
    return max(min(kappa, 1.0), -1.0)


def _cohen_kappa(confusion: list[list[int]]) -> float | None:
    """Square confusion matrix [i][j] = count (rater A=i, rater B=j). Same category order."""
    k = len(confusion)
    total = sum(sum(row) for row in confusion)
    if total == 0:
        return None
    po = sum(confusion[i][i] for i in range(k)) / total
    row_m = [sum(confusion[i][j] for j in range(k)) for i in range(k)]
    col_m = [sum(confusion[i][j] for i in range(k)) for j in range(k)]
    pe = sum(row_m[i] * col_m[i] for i in range(k)) / (total * total)
    if pe >= 1.0 - 1e-12:
        return None
    return (po - pe) / (1.0 - pe)


def _empty_confusion(k: int = len(VALID_LABELS)) -> list[list[int]]:
    return [[0] * k for _ in range(k)]


def _fine_to_coarse_confusion(fine: list[list[int]]) -> list[list[int]]:
    k = len(COARSE_LABELS)
    coarse = [[0] * k for _ in range(k)]
    for ri in range(len(VALID_LABELS)):
        for ci in range(len(VALID_LABELS)):
            coarse[_COARSE_MAP[ri]][_COARSE_MAP[ci]] += fine[ri][ci]
    return coarse


def _compute_mode(
    mode_key: str,
    by_task: dict[str, dict[str, str]],
    annotators: list[str],
) -> dict[str, Any]:
    """Compute agreement stats for a particular label mode."""
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

    # --- Fleiss κ (requires ALL annotators to have a valid mapped label) ---
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

    # --- Pairwise stats ---
    pairwise: list[dict[str, Any]] = []
    for i, a in enumerate(annotators):
        for b in annotators[i + 1 :]:
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


@router.get("/smile-agreement/annotators")
async def agreement_annotators():
    return {"annotators": _list_annotator_names()}


@router.get("/smile-agreement/stats")
async def agreement_stats(annotators: str = Query(..., description="Comma-separated annotator names")):
    names = [s.strip() for s in annotators.split(",") if s.strip()]
    known = set(_list_annotator_names())
    for n in names:
        if n not in known:
            raise HTTPException(status_code=400, detail=f"Unknown annotator '{n}'")
    if len(names) < 1:
        raise HTTPException(status_code=400, detail="Select at least one annotator")

    by_task = _labels_for_tasks(names)

    # Per-annotator label counts (all tasks they labeled)
    per_annotator_counts: dict[str, dict[str, int]] = {a: {lab: 0 for lab in VALID_LABELS} for a in names}
    for _task, labs in by_task.items():
        for a, lab in labs.items():
            if a in per_annotator_counts:
                per_annotator_counts[a][lab] += 1

    fully_labeled_tasks: list[str] = []
    for task_key, labs in by_task.items():
        if all(a in labs for a in names):
            fully_labeled_tasks.append(task_key)

    fleiss: float | None = None
    coarse_fleiss: float | None = None
    percent_full_agreement: float | None = None
    coarse_percent_full_agreement: float | None = None
    if len(names) >= 2 and fully_labeled_tasks:
        fine_count_matrix: list[list[int]] = []
        coarse_count_matrix: list[list[int]] = []
        agree = 0
        coarse_agree = 0
        for task_key in fully_labeled_tasks:
            labs = by_task[task_key]
            row_fine = [0] * len(VALID_LABELS)
            row_coarse = [0] * len(COARSE_LABELS)
            for a in names:
                fi = LABEL_INDEX[labs[a]]
                row_fine[fi] += 1
                row_coarse[_COARSE_MAP[fi]] += 1
            fine_count_matrix.append(row_fine)
            coarse_count_matrix.append(row_coarse)
            if len(set(labs[a] for a in names)) == 1:
                agree += 1
            if len(set(_COARSE_MAP[LABEL_INDEX[labs[a]]] for a in names)) == 1:
                coarse_agree += 1
        n_full = len(fully_labeled_tasks)
        percent_full_agreement = 100.0 * agree / n_full
        coarse_percent_full_agreement = 100.0 * coarse_agree / n_full
        fleiss = _fleiss_kappa(fine_count_matrix)
        coarse_fleiss = _fleiss_kappa(coarse_count_matrix)

    pairwise: list[dict[str, Any]] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            conf = _empty_confusion()
            n_both = 0
            for task_key, labs in by_task.items():
                if a not in labs or b not in labs:
                    continue
                n_both += 1
                ia = LABEL_INDEX[labs[a]]
                ib = LABEL_INDEX[labs[b]]
                conf[ia][ib] += 1
            coarse_conf = _fine_to_coarse_confusion(conf)
            if n_both == 0:
                pairwise.append(
                    {
                        "annotator_a": a,
                        "annotator_b": b,
                        "n_tasks": 0,
                        "cohen_kappa": None,
                        "coarse_cohen_kappa": None,
                        "percent_agreement": None,
                        "coarse_percent_agreement": None,
                        "confusion": conf,
                        "coarse_confusion": coarse_conf,
                    }
                )
                continue
            agree_pair = sum(conf[j][j] for j in range(len(VALID_LABELS)))
            coarse_agree_pair = sum(coarse_conf[j][j] for j in range(len(COARSE_LABELS)))
            pct = 100.0 * agree_pair / n_both
            coarse_pct = 100.0 * coarse_agree_pair / n_both
            kap = _cohen_kappa(conf)
            coarse_kap = _cohen_kappa(coarse_conf)
            pairwise.append(
                {
                    "annotator_a": a,
                    "annotator_b": b,
                    "n_tasks": n_both,
                    "cohen_kappa": kap,
                    "coarse_cohen_kappa": coarse_kap,
                    "percent_agreement": pct,
                    "coarse_percent_agreement": coarse_pct,
                    "confusion": conf,
                    "coarse_confusion": coarse_conf,
                }
            )

    # Compute all modes
    modes = {key: _compute_mode(key, by_task, names) for key in MODES}

    return {
        "annotators": names,
        "valid_labels": list(VALID_LABELS),
        "coarse_labels": list(COARSE_LABELS),
        "per_annotator_counts": per_annotator_counts,
        "tasks_with_any_label": len(by_task),
        "tasks_fully_labeled": len(fully_labeled_tasks),
        "percent_full_agreement": percent_full_agreement,
        "coarse_percent_full_agreement": coarse_percent_full_agreement,
        "fleiss_kappa": fleiss,
        "coarse_fleiss_kappa": coarse_fleiss,
        "pairwise": pairwise,
        "modes": modes,
    }


@router.get("/smile-agreement/au12-scatter")
async def au12_scatter(annotators: str = Query(..., description="Comma-separated annotator names")):
    names = [s.strip() for s in annotators.split(",") if s.strip()]
    known = set(_list_annotator_names())
    for n in names:
        if n not in known:
            raise HTTPException(status_code=400, detail=f"Unknown annotator '{n}'")

    manifest_path = DATA_DIR / "smile_task_manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(status_code=500, detail="Task manifest not found")
    with open(manifest_path) as f:
        manifest = json.load(f)
    task_lookup: dict[str, dict[str, Any]] = {
        str(t["task_number"]): t for t in manifest.get("tasks", [])
    }

    points: list[dict[str, Any]] = []
    for name in names:
        data = _load_annotations(name)
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
