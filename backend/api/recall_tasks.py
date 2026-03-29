from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

DATA_DIR = Path(
    os.environ.get(
        "VOICEOVER_DATA_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "data"),
    )
)

MANIFEST_PATH = DATA_DIR / "recall_task_manifest.json"
ANNOTATIONS_DIR = DATA_DIR / "recall_annotations"
VIDEO_DIR = DATA_DIR / "videos"
FALLBACK_VIDEO_DIR = Path(os.environ.get("VOICEOVER_VIDEO_FALLBACK", "/home/mjma/voices/test_data/videos"))

router = APIRouter()

VALID_LABELS = {"smile", "not_a_smile"}


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail="Recall task manifest not found. Run scripts/generate_recall_manifest.py first.",
        )
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
    path = _annotations_path(annotator)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


@router.get("/recall-tasks/count")
async def task_count():
    manifest = _load_manifest()
    return {
        "total_tasks": manifest["total_tasks"],
        "available_tasks": _count_available_tasks(manifest),
    }


@router.get("/recall-tasks/next-incomplete")
async def next_incomplete(annotator: str = Query(...)):
    manifest = _load_manifest()
    ann_data = _load_annotations(annotator)
    completed = ann_data.get("annotations", {})
    for task in manifest["tasks"]:
        if str(task["task_number"]) not in completed:
            return {"task_number": task["task_number"]}
    return {"task_number": None}


@router.get("/recall-tasks/results")
async def get_results():
    manifest = _load_manifest()
    annotators = _list_annotators()
    all_annotations: dict[str, dict[str, str]] = {}
    for name in annotators:
        data = _load_annotations(name)
        all_annotations[name] = {
            k: v["label"]
            for k, v in data.get("annotations", {}).items()
            if v.get("label") in VALID_LABELS
        }
    return _compute_results(manifest, all_annotations)


@router.get("/recall-tasks/{task_number}")
async def get_task(task_number: int):
    manifest = _load_manifest()
    task_map = {t["task_number"]: t for t in manifest["tasks"]}
    if task_number not in task_map:
        raise HTTPException(status_code=404, detail=f"Task {task_number} not found")
    task = task_map[task_number]
    # Blinded: strip logistic_score and bin from API response
    return {
        "task_number": task["task_number"],
        "video_id": task["video_id"],
        "segment_start": task["segment_start"],
        "segment_end": task["segment_end"],
        "total_tasks": manifest["total_tasks"],
        "available_tasks": _count_available_tasks(manifest),
        "video_downloaded": _video_is_downloaded(task["video_id"]),
    }


@router.get("/recall-annotations/{annotator}")
async def get_annotations(annotator: str):
    return _load_annotations(annotator)


class RecallAnnotateBody(BaseModel):
    annotator: str
    task_number: int
    label: str


def _list_annotators() -> list[str]:
    if not ANNOTATIONS_DIR.is_dir():
        return []
    return sorted(p.stem for p in ANNOTATIONS_DIR.glob("*.json"))


def _majority_label(labels: list[str]) -> str:
    """Return majority label; ties go to 'smile'."""
    smiles = labels.count("smile")
    return "smile" if smiles >= len(labels) / 2 else "not_a_smile"


OPERATING_THRESHOLD = 0.636
BOOTSTRAP_REPS = 10_000
MIN_LABELED_BINS = 5
MIN_LABELED_PER_BIN = 5


def _compute_results(manifest: dict[str, Any], all_annotations: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Compute HT recall estimate and per-bin stats from all annotation files."""
    bins_meta: list[dict[str, Any]] = manifest["bins"]
    tasks: list[dict[str, Any]] = manifest["tasks"]

    # Build per-task consensus label from all annotators
    # all_annotations: {annotator -> {str(task_number) -> label}}
    task_labels: dict[str, list[str]] = {}
    for ann_labels in all_annotations.values():
        for task_key, label in ann_labels.items():
            if label in VALID_LABELS:
                task_labels.setdefault(task_key, []).append(label)

    # Index tasks by task_number (int) -> task dict
    task_by_num: dict[int, dict[str, Any]] = {t["task_number"]: t for t in tasks}

    # Per-bin accumulation
    bin_smile: list[list[int]] = [[] for _ in bins_meta]   # list of 0/1 per labeled task
    bin_labeled: list[int] = [0] * len(bins_meta)

    for task_num_str, labels in task_labels.items():
        task = task_by_num.get(int(task_num_str))
        if task is None:
            continue
        k = task["bin"]
        consensus = _majority_label(labels)
        y = 1 if consensus == "smile" else 0
        bin_smile[k].append(y)
        bin_labeled[k] += 1

    # Per-bin stats
    bin_stats: list[dict[str, Any]] = []
    for k, meta in enumerate(bins_meta):
        ys = bin_smile[k]
        n_labeled = len(ys)
        n_smile = sum(ys)
        smile_rate = n_smile / n_labeled if n_labeled > 0 else None
        passes = meta["score_max"] > OPERATING_THRESHOLD or (
            meta["score_min"] < OPERATING_THRESHOLD < meta["score_max"]
        )
        bin_stats.append({
            "bin": k,
            "score_min": round(meta["score_min"], 4),
            "score_max": round(meta["score_max"], 4),
            "population": meta["population"],
            "sampled": meta["sampled"],
            "labeled": n_labeled,
            "smile_count": n_smile,
            "not_smile_count": n_labeled - n_smile,
            "smile_rate": round(smile_rate, 4) if smile_rate is not None else None,
            "passes_threshold": passes,
        })

    # Horvitz-Thompson recall estimate
    # Recall = sum_k (N_k/n_k * sum_{i in S_k} 1[score_i >= theta] * y_i)
    #        / sum_k (N_k/n_k * sum_{i in S_k} y_i)
    # Since all N_k are equal (decile bins), weights cancel and it simplifies to
    # smile_rate in bins >= theta / overall smile_rate, weighted by population.
    # We implement the general form for correctness.
    bins_ok = sum(1 for b in bin_stats if b["labeled"] >= MIN_LABELED_PER_BIN)
    recall_est: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    ci_method: str | None = None

    if bins_ok >= MIN_LABELED_BINS:
        # Point estimate
        num = 0.0
        denom = 0.0
        for k, meta in enumerate(bins_meta):
            ys = bin_smile[k]
            if not ys:
                continue
            N_k = meta["population"]
            n_k = len(ys)
            w = N_k / n_k
            # passes_threshold: score_min >= OPERATING_THRESHOLD counts toward numerator
            # For the boundary bin (spans theta), we weight by the fraction above theta,
            # but since we can't know per-task scores (blinded), we treat entire bin.
            # Conservative: bin "passes" if its score_min >= threshold.
            if meta["score_min"] >= OPERATING_THRESHOLD:
                num += w * sum(ys)
            denom += w * sum(ys)

        recall_est = num / denom if denom > 0 else None

        # Stratified bootstrap CI
        if recall_est is not None:
            rng = np.random.default_rng(0)
            boot_recalls: list[float] = []
            for _ in range(BOOTSTRAP_REPS):
                b_num = 0.0
                b_den = 0.0
                for k, meta in enumerate(bins_meta):
                    ys = bin_smile[k]
                    if not ys:
                        continue
                    N_k = meta["population"]
                    n_k = len(ys)
                    w = N_k / n_k
                    sample = rng.choice(ys, size=len(ys), replace=True)
                    s = int(sample.sum())
                    if meta["score_min"] >= OPERATING_THRESHOLD:
                        b_num += w * s
                    b_den += w * s
                if b_den > 0:
                    boot_recalls.append(b_num / b_den)
            if len(boot_recalls) > 100:
                arr = np.array(boot_recalls)
                ci_low = float(np.percentile(arr, 2.5))
                ci_high = float(np.percentile(arr, 97.5))
                ci_method = f"stratified_bootstrap_{BOOTSTRAP_REPS}"

    # Per-annotator counts
    per_annotator: dict[str, dict[str, int]] = {}
    for name, ann_labels in all_annotations.items():
        counts: dict[str, int] = {"smile": 0, "not_a_smile": 0, "total": 0}
        for label in ann_labels.values():
            if label in VALID_LABELS:
                counts[label] += 1
                counts["total"] += 1
        per_annotator[name] = counts

    return {
        "operating_threshold": OPERATING_THRESHOLD,
        "annotators": sorted(all_annotations.keys()),
        "total_tasks": manifest["total_tasks"],
        "population_size": manifest["population_size"],
        "completed_tasks": sum(b["labeled"] for b in bin_stats),
        "bins": bin_stats,
        "recall_estimate": round(recall_est, 4) if recall_est is not None else None,
        "recall_ci_low": round(ci_low, 4) if ci_low is not None else None,
        "recall_ci_high": round(ci_high, 4) if ci_high is not None else None,
        "ci_method": ci_method,
        "bins_with_data": bins_ok,
        "min_bins_for_estimate": MIN_LABELED_BINS,
        "per_annotator_counts": per_annotator,
    }


@router.post("/recall-annotations")
async def save_annotation(body: RecallAnnotateBody):
    if body.label not in VALID_LABELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid label '{body.label}'. Must be one of: {', '.join(sorted(VALID_LABELS))}",
        )
    manifest = _load_manifest()
    task_numbers = {t["task_number"] for t in manifest["tasks"]}
    if body.task_number not in task_numbers:
        raise HTTPException(status_code=404, detail=f"Task {body.task_number} not found")

    data = _load_annotations(body.annotator)
    data["annotations"][str(body.task_number)] = {
        "label": body.label,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _save_annotations(body.annotator, data)
    return {"ok": True, "task_number": body.task_number, "label": body.label}
