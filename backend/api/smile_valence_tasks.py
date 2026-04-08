from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

DATA_DIR = Path(
    os.environ.get(
        "VOICEOVER_DATA_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "data"),
    )
)

MANIFEST_PATH = DATA_DIR / "smile_valence_task_manifest.json"
ANNOTATIONS_DIR = DATA_DIR / "smile_valence_annotations"
VIDEO_DIR = DATA_DIR / "videos"
FALLBACK_VIDEO_DIR = Path(
    os.environ.get("VOICEOVER_VIDEO_FALLBACK", "/home/mjma/voices/test_data/videos")
)

router = APIRouter()

Valence = Literal["negative", "neutral", "positive"]


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail="Task manifest not generated yet. Run stratification/generate_smile_valence_manifest.py first.",
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
    with open(_annotations_path(annotator), "w") as f:
        json.dump(data, f, indent=2)


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get("/smile-valence-tasks/count")
async def count_tasks():
    manifest = _load_manifest()
    return {
        "total_tasks": manifest["total_tasks"],
        "available_tasks": _count_available_tasks(manifest),
    }


@router.get("/smile-valence-tasks/next-incomplete")
async def next_incomplete(annotator: str = Query(...)):
    manifest = _load_manifest()
    ann_data = _load_annotations(annotator)
    completed = ann_data.get("annotations", {})
    for task in manifest["tasks"]:
        if task["task_number"] <= 1:
            continue
        if str(task["task_number"]) not in completed:
            return {"task_number": task["task_number"]}
    return {"task_number": None}


@router.get("/smile-valence-tasks/{task_number}")
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
        "peak_r": task.get("peak_r"),
        "total_tasks": manifest["total_tasks"],
        "available_tasks": _count_available_tasks(manifest),
        "video_downloaded": _video_is_downloaded(task["video_id"]),
    }


@router.get("/smile-valence-annotations/{annotator}")
async def get_annotations(annotator: str):
    return _load_annotations(annotator)


class SmileValenceAnnotateBody(BaseModel):
    annotator: str
    task_number: int
    narrative_valence: Valence | None = None
    speaker_valence: Valence | None = None
    not_a_smile: bool = False


@router.post("/smile-valence-annotations")
async def save_annotation(body: SmileValenceAnnotateBody):
    if not body.not_a_smile and (body.narrative_valence is None or body.speaker_valence is None):
        raise HTTPException(
            status_code=400,
            detail="narrative_valence and speaker_valence are required unless not_a_smile is True",
        )

    manifest = _load_manifest()
    task_numbers = {t["task_number"] for t in manifest["tasks"]}
    if body.task_number not in task_numbers:
        raise HTTPException(status_code=404, detail=f"Task {body.task_number} not found")

    data = _load_annotations(body.annotator)
    entry: dict = {"timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if body.not_a_smile:
        entry["not_a_smile"] = True
    else:
        entry["narrative_valence"] = body.narrative_valence
        entry["speaker_valence"] = body.speaker_valence
    data["annotations"][str(body.task_number)] = entry
    _save_annotations(body.annotator, data)
    return {"ok": True, "task_number": body.task_number}


# ── Results / stats ───────────────────────────────────────────────────────────

LLM_EYEGAZE_DIR = DATA_DIR / "llm_annotated_eyegaze"

VALENCE_LABELS = ["negative", "neutral", "positive"]


def _best_sentence_for_smile(
    sentences: list[dict], smile_start_ms: float, smile_end_ms: float
) -> dict | None:
    """Return the sentence that best covers the smile window.

    Scoring: temporal overlap duration (with a 1ms floor for point-sentences
    that fall inside the window).  Ties broken by proximity of sentence
    midpoint to smile midpoint.
    """
    smile_mid = (smile_start_ms + smile_end_ms) / 2
    best: dict | None = None
    best_score: tuple[float, float] = (-1, float("inf"))

    for s in sentences:
        fw = s.get("first_word_ms")
        lw = s.get("last_word_ms")
        if fw is None or lw is None:
            continue
        fw, lw = float(fw), float(lw)
        # Overlap duration
        overlap = min(smile_end_ms, lw) - max(smile_start_ms, fw)
        if overlap < 0:
            continue
        # Point-sentences (fw == lw) that fall inside the window get 1ms overlap
        if overlap == 0 and fw >= smile_start_ms and lw <= smile_end_ms:
            overlap = 1.0
        if overlap <= 0:
            continue
        mid_dist = abs((fw + lw) / 2 - smile_mid)
        score: tuple[float, float] = (overlap, -mid_dist)
        if score > best_score:
            best_score = score
            best = s

    return best


def _build_llm_by_task(tasks: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Return dict of task_number → LLM annotation entry.

    Prefers labels embedded directly in the manifest task (llm_narrative_valence /
    llm_speaker_valence).  Falls back to loading the eyegaze file for any task
    that lacks embedded labels.
    """
    result: dict[int, dict[str, Any]] = {}
    fallback_tasks: list[dict[str, Any]] = []

    for t in tasks:
        nv = t.get("llm_narrative_valence")
        sv = t.get("llm_speaker_valence")
        if nv in VALENCE_LABELS and sv in VALENCE_LABELS:
            result[t["task_number"]] = {
                "narrative_valence": nv,
                "speaker_valence": sv,
            }
        else:
            fallback_tasks.append(t)

    if fallback_tasks:
        # Group by video_id and load eyegaze files for remaining tasks
        by_video: dict[str, list[dict]] = {}
        for t in fallback_tasks:
            by_video.setdefault(t["video_id"], []).append(t)
        for vid, vid_tasks in by_video.items():
            fp = LLM_EYEGAZE_DIR / f"{vid}.json"
            if not fp.is_file():
                continue
            try:
                with open(fp) as f:
                    data = json.load(f)
            except Exception:
                continue
            sentences = data.get("sentences", [])
            for t in vid_tasks:
                start_ms = t["smile_start"] * 1000
                end_ms = t["smile_end"] * 1000
                sent = _best_sentence_for_smile(sentences, start_ms, end_ms)
                if sent is None:
                    continue
                nv = sent.get("narrative_valence")
                sv = sent.get("present_day_valence")
                if nv not in VALENCE_LABELS or sv not in VALENCE_LABELS:
                    continue
                result[t["task_number"]] = {
                    "narrative_valence": nv,
                    "speaker_valence": sv,
                    "content_domain": sent.get("content_domain"),
                    "narrative_structure": sent.get("narrative_structure"),
                    "sentence_text": sent.get("text", ""),
                }

    return result


def _majority(labels: list[str]) -> str | None:
    if not labels:
        return None
    cnt = Counter(labels)
    top = cnt.most_common(1)[0]
    return top[0]


def _cohen_kappa(y1: list[str], y2: list[str], labels: list[str]) -> float | None:
    """Simple Cohen's kappa implementation (unweighted)."""
    if len(y1) < 2:
        return None
    try:
        from sklearn.metrics import cohen_kappa_score  # type: ignore
        return round(float(cohen_kappa_score(y1, y2, labels=labels)), 4)
    except Exception:
        return None


def _compute_iaa(
    annotators: list[str],
    task_annotations: dict[int, dict[str, dict]],  # task_num → {annotator → entry}
    field: str,
) -> dict[str, Any]:
    """Compute IAA stats for a given field across all annotator pairs."""
    pairs = []
    for i in range(len(annotators)):
        for j in range(i + 1, len(annotators)):
            a1, a2 = annotators[i], annotators[j]
            y1, y2 = [], []
            for task_anns in task_annotations.values():
                e1 = task_anns.get(a1)
                e2 = task_anns.get(a2)
                if e1 is None or e2 is None:
                    continue
                v1 = e1.get(field)
                v2 = e2.get(field)
                if v1 not in VALENCE_LABELS or v2 not in VALENCE_LABELS:
                    continue
                y1.append(v1)
                y2.append(v2)
            n = len(y1)
            if n == 0:
                continue
            agree = sum(a == b for a, b in zip(y1, y2))
            pct = round(agree / n, 4)
            kappa = _cohen_kappa(y1, y2, VALENCE_LABELS)
            pairs.append({"annotator_1": a1, "annotator_2": a2, "n": n, "pct_agree": pct, "kappa": kappa})

    # Overall: pool all pairwise-matchable labels
    all_y1, all_y2 = [], []
    for p in pairs:
        pass  # already computed per pair above; compute pooled separately

    # Simpler pooled: for each task, take all pairs of valid annotations
    pooled_y1, pooled_y2 = [], []
    for task_anns in task_annotations.values():
        valid = [
            (ann, ent[field])
            for ann, ent in task_anns.items()
            if ent.get(field) in VALENCE_LABELS
        ]
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                pooled_y1.append(valid[i][1])
                pooled_y2.append(valid[j][1])

    n_pooled = len(pooled_y1)
    overall: dict[str, Any] = {"n_pairs": n_pooled, "pct_agree": None, "kappa": None}
    if n_pooled > 0:
        overall["pct_agree"] = round(sum(a == b for a, b in zip(pooled_y1, pooled_y2)) / n_pooled, 4)
        overall["kappa"] = _cohen_kappa(pooled_y1, pooled_y2, VALENCE_LABELS)

    return {"overall": overall, "pairs": pairs}


@router.get("/smile-valence-results")
async def get_results():
    manifest = _load_manifest()
    task_map = {t["task_number"]: t for t in manifest["tasks"]}
    llm_by_task = _build_llm_by_task(manifest["tasks"])

    # Load all human annotations
    annotators: list[str] = []
    ann_by_annotator: dict[str, dict[str, Any]] = {}
    if ANNOTATIONS_DIR.is_dir():
        for fp in sorted(ANNOTATIONS_DIR.glob("*.json")):
            try:
                with open(fp) as f:
                    ann_data = json.load(f)
            except Exception:
                continue
            name = ann_data.get("annotator", fp.stem)
            annotators.append(name)
            ann_by_annotator[name] = ann_data.get("annotations", {})

    # Build per-task annotation index: task_num → {annotator → entry}
    task_annotations: dict[int, dict[str, dict]] = {t: {} for t in task_map}
    total_ann_count = 0
    for ann_name, anns in ann_by_annotator.items():
        for task_key, entry in anns.items():
            tn = int(task_key)
            if tn in task_annotations:
                task_annotations[tn][ann_name] = entry
                total_ann_count += 1

    # IAA
    iaa = {
        "narrative_valence": _compute_iaa(annotators, task_annotations, "narrative_valence"),
        "speaker_valence": _compute_iaa(annotators, task_annotations, "speaker_valence"),
    }

    # LLM alignment: compare LLM labels to majority-vote human label
    llm_alignment: dict[str, Any] = {"narrative_valence": {}, "speaker_valence": {}}
    for field, llm_field in [("narrative_valence", "narrative_valence"), ("speaker_valence", "speaker_valence")]:
        correct, total = 0, 0
        conf_matrix: dict[str, dict[str, int]] = {l: {l2: 0 for l2 in VALENCE_LABELS} for l in VALENCE_LABELS}
        for tn, llm_ann in llm_by_task.items():
            llm_val = llm_ann.get(llm_field)
            if llm_val not in VALENCE_LABELS:
                continue
            human_vals = [
                e[field] for e in task_annotations[tn].values()
                if e.get(field) in VALENCE_LABELS
            ]
            if not human_vals:
                continue
            majority = _majority(human_vals)
            if majority:
                total += 1
                if llm_val == majority:
                    correct += 1
                conf_matrix[majority][llm_val] += 1
        llm_alignment[field] = {
            "n": total,
            "accuracy": round(correct / total, 4) if total > 0 else None,
            "confusion_matrix": conf_matrix,
        }

    # Build flat results list for the table
    results: list[dict[str, Any]] = []
    for tn, task in task_map.items():
        task_anns = task_annotations[tn]
        if not task_anns and tn not in llm_by_task:
            continue
        results.append({
            "task_number": tn,
            "video_id": task["video_id"],
            "smile_start": task["smile_start"],
            "smile_end": task["smile_end"],
            "score": task.get("score"),
            "human_annotations": [
                {"annotator": ann, **entry}
                for ann, entry in sorted(task_anns.items())
            ],
            "llm": llm_by_task.get(tn),
        })
    results.sort(key=lambda r: r["task_number"])

    return {
        "total_tasks": manifest["total_tasks"],
        "total_annotations": total_ann_count,
        "annotators": annotators,
        "iaa": iaa,
        "llm_alignment": llm_alignment,
        "llm_tasks_count": len(llm_by_task),
        "results": results,
    }
