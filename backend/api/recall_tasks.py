from __future__ import annotations

import hashlib
import json
import math
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

# ── Model definitions ─────────────────────────────────────────────────────────

MODELS: dict[str, dict[str, Any]] = {
    "logistic": {
        "score_key": "logistic_score",
        "operating_threshold": 0.636,
        "display_name": "17-AU Logistic",
    },
    "au12": {
        "score_key": "mean_r",
        "operating_threshold": 1.5,
        "display_name": "AU12 Threshold",
    },
}

# ── In-process manifest score cache (loaded once per server process) ──────────

_manifest_scores_mem: dict[str, dict[str, dict]] = {}


def _load_manifest_scores(manifest_path: Path) -> dict[str, dict]:
    """Load {str(task_number): {mean_r, logistic_score?}} from a manifest.
    Results are cached in memory so large manifests are only parsed once.
    """
    key = str(manifest_path)
    if key not in _manifest_scores_mem:
        with open(manifest_path) as f:
            m = json.load(f)
        lookup: dict[str, dict] = {}
        for t in m.get("tasks", []):
            entry: dict[str, Any] = {"mean_r": t.get("mean_r")}
            if "logistic_score" in t:
                entry["logistic_score"] = t["logistic_score"]
            lookup[str(t["task_number"])] = entry
        _manifest_scores_mem[key] = lookup
    return _manifest_scores_mem[key]


# ── Disk cache helpers ────────────────────────────────────────────────────────

def _cache_path(model: str) -> Path:
    return DATA_DIR / f"recall_pr_cache_{model}.json"


def _compute_cache_key() -> str:
    """MD5 of annotation file names+mtimes across all three sources.
    Any new file or save triggers recomputation on next request.
    """
    parts: list[str] = []
    for d in (
        ANNOTATIONS_DIR,
        DATA_DIR / "pilot_smile_annotations",
        DATA_DIR / "smile_annotations",
    ):
        if d.is_dir():
            for f in sorted(d.glob("*.json")):
                parts.append(f"{d.name}/{f.name}:{int(f.stat().st_mtime)}")
    return hashlib.md5(":".join(parts).encode()).hexdigest()[:12]


def _load_cache(model: str, cache_key: str) -> dict | None:
    path = _cache_path(model)
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if data.get("cache_key") == cache_key:
            return data
    except Exception:
        pass
    return None


def _save_cache(model: str, data: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(model), "w") as f:
        json.dump(data, f)


# ── Binary label converters ───────────────────────────────────────────────────

def _binary_label_pilot(entry: dict) -> str | None:
    label = entry.get("label")
    if label in ("genuine", "polite", "masking"):
        return "smile"
    if label == "not_a_smile":
        return "not_a_smile"
    return None


def _binary_label_main(entry: dict) -> str | None:
    if entry.get("not_a_smile") or entry.get("label") == "not_a_smile":
        return "not_a_smile"
    if entry.get("label") in ("felt", "false", "miserable"):
        return "smile"
    return None


def _binary_label_recall(entry: dict) -> str | None:
    label = entry.get("label")
    return label if label in VALID_LABELS else None


def _majority_label(labels: list[str]) -> str:
    """Majority vote; ties favour 'smile'."""
    smiles = labels.count("smile")
    return "smile" if smiles * 2 >= len(labels) else "not_a_smile"


# ── Data loading ──────────────────────────────────────────────────────────────

def _collect_annotations(ann_dir: Path, label_fn) -> dict[str, list[str]]:
    """Accumulate binary labels per task_number from all annotators in a dir."""
    result: dict[str, list[str]] = {}
    if not ann_dir.is_dir():
        return result
    for f in ann_dir.glob("*.json"):
        with open(f) as fp:
            data = json.load(fp)
        for tk, entry in data.get("annotations", {}).items():
            label = label_fn(entry)
            if label:
                result.setdefault(tk, []).append(label)
    return result


def _load_recall_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail="Recall task manifest not found. Run scripts/generate_recall_manifest.py first.",
        )
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def _load_all_labeled_points(model: str) -> dict[str, list[dict]]:
    """Return labeled data points from all three annotation sources.

    Each point: {score: float, label: str, bin: int|-1}
    Bin is set for recall-manifest points; -1 for supplementary data.
    """
    score_key = MODELS[model]["score_key"]

    # ── Recall manifest (stratified sample, full score range) ──
    recall_manifest = _load_recall_manifest()
    task_by_num: dict[str, dict] = {str(t["task_number"]): t for t in recall_manifest["tasks"]}

    recall_ann = _collect_annotations(ANNOTATIONS_DIR, _binary_label_recall)
    recall_points: list[dict] = []
    for tk, labels in recall_ann.items():
        task = task_by_num.get(tk)
        if not task:
            continue
        score = task.get(score_key)
        if score is None:
            continue
        recall_points.append({
            "score": float(score),
            "label": _majority_label(labels),
            "bin": task.get("bin", -1),
        })

    # ── Main study (high-score bias, logistic_score available) ──
    main_manifest = DATA_DIR / "smile_task_manifest.json"
    main_scores = _load_manifest_scores(main_manifest) if main_manifest.is_file() else {}
    main_ann = _collect_annotations(DATA_DIR / "smile_annotations", _binary_label_main)
    main_points: list[dict] = []
    for tk, labels in main_ann.items():
        task_scores = main_scores.get(tk)
        if not task_scores:
            continue
        score = task_scores.get(score_key)
        if score is None:
            continue
        main_points.append({
            "score": float(score),
            "label": _majority_label(labels),
            "bin": -1,
        })

    # ── Pilot study (high-score bias, AU12 only — no logistic scores stored) ──
    pilot_points: list[dict] = []
    if score_key == "mean_r":
        pilot_manifest = DATA_DIR / "pilot_smile_task_manifest.json"
        pilot_scores = _load_manifest_scores(pilot_manifest) if pilot_manifest.is_file() else {}
        pilot_ann = _collect_annotations(DATA_DIR / "pilot_smile_annotations", _binary_label_pilot)
        for tk, labels in pilot_ann.items():
            task_scores = pilot_scores.get(tk)
            if not task_scores:
                continue
            score = task_scores.get(score_key)
            if score is None:
                continue
            pilot_points.append({
                "score": float(score),
                "label": _majority_label(labels),
                "bin": -1,
            })

    return {"recall": recall_points, "main": main_points, "pilot": pilot_points}


# ── Statistics helpers ────────────────────────────────────────────────────────

def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson confidence interval for k successes in n trials."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def _r(v: float | None, d: int = 4) -> float | None:
    return round(v, d) if v is not None else None


def _compute_pr_curve(
    all_points: list[dict], thresholds: list[float]
) -> list[dict]:
    """Precision/recall with Wilson CIs at each threshold (all data, unweighted)."""
    scores = np.array([p["score"] for p in all_points])
    ys = np.array([1 if p["label"] == "smile" else 0 for p in all_points])
    thr_arr = np.array(thresholds)

    # passes[i, j] = 1 iff points[i] passes threshold[j]
    passes = (scores[:, None] >= thr_arr[None, :])  # (N, T)

    tp_arr = (passes * ys[:, None]).sum(axis=0)       # (T,)
    fp_arr = (passes * (1 - ys)[:, None]).sum(axis=0) # (T,)
    fn_arr = (~passes * ys[:, None]).sum(axis=0)      # (T,)
    tn_arr = (~passes * (1 - ys)[:, None]).sum(axis=0)

    curve: list[dict] = []
    for j, thr in enumerate(thresholds):
        tp, fp, fn, tn = int(tp_arr[j]), int(fp_arr[j]), int(fn_arr[j]), int(tn_arr[j])
        rec = tp / (tp + fn) if (tp + fn) > 0 else None
        prec = tp / (tp + fp) if (tp + fp) > 0 else None
        r_lo, r_hi = _wilson_ci(tp, tp + fn) if (tp + fn) > 0 else (None, None)
        p_lo, p_hi = _wilson_ci(tp, tp + fp) if (tp + fp) > 0 else (None, None)
        curve.append({
            "threshold": round(float(thr), 5),
            "recall": _r(rec),
            "precision": _r(prec),
            "recall_ci_low": _r(r_lo),
            "recall_ci_high": _r(r_hi),
            "precision_ci_low": _r(p_lo),
            "precision_ci_high": _r(p_hi),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return curve


def _compute_bin_stats(
    recall_points: list[dict],
    bins_meta: list[dict],
    operating_threshold: float,
) -> list[dict]:
    bin_data: dict[int, list[int]] = {}
    for p in recall_points:
        k = p["bin"]
        if k >= 0:
            y = 1 if p["label"] == "smile" else 0
            bin_data.setdefault(k, []).append(y)

    stats = []
    for bm in bins_meta:
        k = bm["bin"]
        ys = bin_data.get(k, [])
        n = len(ys)
        n_smile = sum(ys)
        stats.append({
            "bin": k,
            "score_min": round(float(bm["score_min"]), 4),
            "score_max": round(float(bm["score_max"]), 4),
            "population": bm["population"],
            "sampled": bm["sampled"],
            "labeled": n,
            "smile_count": n_smile,
            "not_smile_count": n - n_smile,
            "smile_rate": _r(n_smile / n) if n > 0 else None,
            "passes_threshold": bm["score_min"] >= operating_threshold,
        })
    return stats


# ── Main computation ──────────────────────────────────────────────────────────

N_THRESHOLDS = 200


def _compute_and_cache(model: str, cache_key: str) -> dict:
    """Full computation for one model. Loads data, computes PR curves + HT curve,
    saves to disk cache, returns result dict."""
    model_cfg = MODELS[model]
    score_key = model_cfg["score_key"]
    operating_threshold = model_cfg["operating_threshold"]

    sources = _load_all_labeled_points(model)
    recall_pts = sources["recall"]
    main_pts = sources["main"]
    pilot_pts = sources["pilot"]
    all_pts = recall_pts + main_pts + pilot_pts

    recall_manifest = _load_recall_manifest()
    bins_meta: list[dict] = recall_manifest["bins"]

    bin_stats = _compute_bin_stats(recall_pts, bins_meta, operating_threshold)

    per_annotator: dict[str, dict] = {}
    for f in ANNOTATIONS_DIR.glob("*.json"):
        with open(f) as fp:
            data = json.load(fp)
        counts: dict[str, int] = {"smile": 0, "not_a_smile": 0, "total": 0}
        for entry in data.get("annotations", {}).values():
            lbl = _binary_label_recall(entry)
            if lbl:
                counts[lbl] += 1
                counts["total"] += 1
        per_annotator[f.stem] = counts

    if not all_pts:
        result: dict[str, Any] = {
            "model": model,
            "cache_key": cache_key,
            "computed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "score_key": score_key,
            "operating_threshold": operating_threshold,
            "score_range": [0.0, 1.0],
            "sources": {"recall_manifest": 0, "main_study": 0, "pilot_study": 0},
            "total_labeled": 0,
            "pr_curve": [],
            "bins": bin_stats,
            "annotators": _list_annotators(),
            "per_annotator_counts": per_annotator,
        }
        _save_cache(model, result)
        return result

    # Threshold range: cover full observed score range
    all_scores = [p["score"] for p in all_pts]
    s_min, s_max = min(all_scores), max(all_scores)
    # Always include operating threshold in the sweep
    thresholds_set = set(np.linspace(s_min, s_max, N_THRESHOLDS).tolist())
    thresholds_set.add(float(operating_threshold))
    thresholds = sorted(thresholds_set)

    pr_curve = _compute_pr_curve(all_pts, thresholds)

    result = {
        "model": model,
        "cache_key": cache_key,
        "computed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "score_key": score_key,
        "operating_threshold": operating_threshold,
        "score_range": [round(s_min, 4), round(s_max, 4)],
        "sources": {
            "recall_manifest": len(recall_pts),
            "main_study": len(main_pts),
            "pilot_study": len(pilot_pts),
        },
        "total_labeled": len(all_pts),
        "total_recall_tasks": recall_manifest["total_tasks"],
        "population_size": recall_manifest["population_size"],
        "pr_curve": pr_curve,
        "bins": bin_stats,
        "annotators": _list_annotators(),
        "per_annotator_counts": per_annotator,
    }
    _save_cache(model, result)
    return result


# ── Route helpers ─────────────────────────────────────────────────────────────

def _load_manifest() -> dict[str, Any]:
    return _load_recall_manifest()


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


def _list_annotators() -> list[str]:
    if not ANNOTATIONS_DIR.is_dir():
        return []
    return sorted(p.stem for p in ANNOTATIONS_DIR.glob("*.json"))


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/recall-tasks/count")
async def task_count():
    manifest = _load_manifest()
    return {
        "total_tasks": manifest["total_tasks"],
        "available_tasks": _count_available_tasks(manifest),
    }


@router.get("/recall-tasks/results")
async def get_results(model: str = Query("logistic")):
    if model not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'. Choose: {list(MODELS)}")
    cache_key = _compute_cache_key()
    cached = _load_cache(model, cache_key)
    if cached is not None:
        cached["is_cached"] = True
        return cached
    result = _compute_and_cache(model, cache_key)
    result["is_cached"] = False
    return result


@router.get("/recall-tasks/next-incomplete")
async def next_incomplete(annotator: str = Query(...)):
    manifest = _load_manifest()
    ann_data = _load_annotations(annotator)
    completed = ann_data.get("annotations", {})
    for task in manifest["tasks"]:
        if str(task["task_number"]) not in completed:
            return {"task_number": task["task_number"]}
    return {"task_number": None}


@router.get("/recall-tasks/{task_number}")
async def get_task(task_number: int):
    manifest = _load_manifest()
    task_map = {t["task_number"]: t for t in manifest["tasks"]}
    if task_number not in task_map:
        raise HTTPException(status_code=404, detail=f"Task {task_number} not found")
    task = task_map[task_number]
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

    # Invalidate disk cache so next results request recomputes
    for m in MODELS:
        _cache_path(m).unlink(missing_ok=True)

    return {"ok": True, "task_number": body.task_number, "label": body.label}
