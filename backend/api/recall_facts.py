from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

DATA_DIR = Path(
    os.environ.get(
        "VOICEOVER_DATA_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "data"),
    )
)

TASKS_FILE = DATA_DIR / "recall_facts_annotations" / "tasks.json"
ANNOTATIONS_DIR = DATA_DIR / "recall_facts_annotations"

router = APIRouter()


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_tasks() -> dict[str, Any]:
    if not TASKS_FILE.is_file():
        raise HTTPException(
            status_code=404,
            detail="tasks.json not found. Run scripts/generate_recall_facts_tasks.py first.",
        )
    return json.loads(TASKS_FILE.read_text())


def _annotations_path(annotator: str) -> Path:
    return ANNOTATIONS_DIR / f"{annotator}.json"


def _load_annotations(annotator: str) -> dict[str, Any]:
    path = _annotations_path(annotator)
    if path.is_file():
        return json.loads(path.read_text())
    return {"annotator": annotator, "annotations": {}}


def _save_annotations(annotator: str, data: dict[str, Any]) -> None:
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = _annotations_path(annotator)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _list_annotators() -> list[str]:
    if not ANNOTATIONS_DIR.is_dir():
        return []
    return sorted(
        p.stem
        for p in ANNOTATIONS_DIR.glob("*.json")
        if p.name != "tasks.json"
    )


# ── request models ────────────────────────────────────────────────────────────

class AnnotationIn(BaseModel):
    annotator: str
    task_id: int
    # topic tasks: "yes" | "no" | "unsure"
    # memory_type tasks: "internal" | "external" | "unsure"
    answer: str


# ── routes ────────────────────────────────────────────────────────────────────

@router.get("/recall-facts/tasks")
async def get_tasks():
    return _load_tasks()


@router.get("/recall-facts/my-annotations")
async def get_my_annotations(annotator: str):
    if not annotator.strip():
        raise HTTPException(status_code=400, detail="annotator required")
    return _load_annotations(annotator.strip())


@router.post("/recall-facts/annotation")
async def save_annotation(body: AnnotationIn):
    annotator = body.annotator.strip()
    if not annotator:
        raise HTTPException(status_code=400, detail="annotator required")
    valid = {"yes", "no", "unsure", "internal", "external"}
    if body.answer not in valid:
        raise HTTPException(status_code=400, detail=f"answer must be one of {valid}")

    data = _load_annotations(annotator)
    data["annotations"][str(body.task_id)] = {
        "answer": body.answer,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _save_annotations(annotator, data)
    return {"ok": True, "saved": len(data["annotations"])}


@router.get("/recall-facts/annotators")
async def list_annotators():
    return {"annotators": _list_annotators()}


@router.get("/recall-facts/agreement")
async def get_agreement():
    """
    Per-annotator agreement stats vs the LLM baseline.

    Topic tasks:       human 'yes' = LLM was correct to assign the topic
                       human 'no'  = LLM false-positive
    Memory-type tasks: human 'internal'/'external' vs LLM's memory_type value
    """
    task_data = _load_tasks()
    tasks_by_id: dict[int, dict] = {t["id"]: t for t in task_data["tasks"]}
    annotators = _list_annotators()

    rows = []
    for name in annotators:
        ann = _load_annotations(name).get("annotations", {})
        completed = len(ann)

        topic_agree = topic_disagree = topic_unsure = 0
        mt_agree = mt_disagree = mt_unsure = 0

        for task_id_str, record in ann.items():
            task = tasks_by_id.get(int(task_id_str))
            if task is None:
                continue
            answer = record.get("answer")
            if task["type"] == "topic":
                # LLM always "yes" (it assigned this topic)
                if answer == "unsure":
                    topic_unsure += 1
                elif answer == "yes":
                    topic_agree += 1
                else:
                    topic_disagree += 1
            elif task["type"] == "memory_type":
                llm = task["llm_memory_type"]
                if answer == "unsure":
                    mt_unsure += 1
                elif answer == llm:
                    mt_agree += 1
                else:
                    mt_disagree += 1

        topic_done = topic_agree + topic_disagree + topic_unsure
        mt_done = mt_agree + mt_disagree + mt_unsure

        rows.append({
            "annotator": name,
            "completed": completed,
            "topic_done": topic_done,
            "topic_agree": topic_agree,
            "topic_disagree": topic_disagree,
            "topic_unsure": topic_unsure,
            "topic_pct_agree": round(topic_agree / topic_done * 100, 1) if topic_done > 0 else None,
            "mt_done": mt_done,
            "mt_agree": mt_agree,
            "mt_disagree": mt_disagree,
            "mt_unsure": mt_unsure,
            "mt_pct_agree": round(mt_agree / mt_done * 100, 1) if mt_done > 0 else None,
        })

    # Inter-annotator agreement (memory_type tasks only — topic tasks are confirmations, not labels)
    inter_agree_pct = None
    if len(annotators) >= 2:
        all_anns = {
            name: _load_annotations(name).get("annotations", {})
            for name in annotators
        }
        agree_count = total_count = 0
        for task_id, task in tasks_by_id.items():
            if task["type"] != "memory_type":
                continue
            answers = [
                all_anns[name].get(str(task_id), {}).get("answer")
                for name in annotators
            ]
            definite = [a for a in answers if a in ("internal", "external")]
            if len(definite) >= 2:
                majority = max(set(definite), key=definite.count)
                agree_count += sum(1 for a in definite if a == majority)
                total_count += len(definite)
        if total_count > 0:
            inter_agree_pct = round(agree_count / total_count * 100, 1)

    return {
        "annotators": rows,
        "inter_annotator_agreement_pct": inter_agree_pct,
        "total_tasks": task_data["total_tasks"],
        "n_topic_tasks": task_data["n_topic_tasks"],
        "n_memory_type_tasks": task_data["n_memory_type_tasks"],
    }


@router.get("/recall-facts/task-detail")
async def get_task_detail():
    """Per-task breakdown: sentence, LLM answer, each annotator's answer."""
    task_data = _load_tasks()
    annotators = _list_annotators()
    all_anns = {
        name: _load_annotations(name).get("annotations", {})
        for name in annotators
    }

    detail = []
    for task in task_data["tasks"]:
        tid = task["id"]
        responses = {
            name: all_anns[name].get(str(tid), {}).get("answer")
            for name in annotators
        }
        if task["type"] == "topic":
            llm_answer = "yes"  # LLM assigned this topic
        else:
            llm_answer = task["llm_memory_type"]
        detail.append({
            "id": tid,
            "type": task["type"],
            "sentence": task["sentence_text"][:120],
            "transcript_id": task["transcript_id"],
            "topic": task.get("topic_to_validate"),
            "llm_answer": llm_answer,
            "responses": responses,
        })
    return {"tasks": detail, "annotators": annotators}
