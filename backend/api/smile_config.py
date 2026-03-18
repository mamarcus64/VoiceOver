from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from generate_task_manifest import (
    DEFAULT_PARAMS,
    build_tasks,
    generate_and_write,
    preview_stats,
)

DATA_DIR = Path(
    os.environ.get(
        "VOICEOVER_DATA_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "data"),
    )
)

CONFIG_PATH = DATA_DIR / "smile_config.json"

DEFAULTS: dict[str, Any] = {
    **DEFAULT_PARAMS,
    "contextBefore": 10.0,
    "contextAfter": 5.0,
}

router = APIRouter()


class SmileConfigBody(BaseModel):
    intensityThreshold: float = DEFAULTS["intensityThreshold"]
    mergeDistance: float = DEFAULTS["mergeDistance"]
    minDuration: float = DEFAULTS["minDuration"]
    maxPerVideo: int = DEFAULTS["maxPerVideo"]
    contextBefore: float = DEFAULTS["contextBefore"]
    contextAfter: float = DEFAULTS["contextAfter"]


def _load_config() -> dict[str, Any]:
    if CONFIG_PATH.is_file():
        with open(CONFIG_PATH) as f:
            return {**DEFAULTS, **json.load(f)}
    return dict(DEFAULTS)


def _save_config(cfg: dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def _task_params(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: cfg[k] for k in ("intensityThreshold", "mergeDistance", "minDuration", "maxPerVideo")}


@router.get("/smile-config")
async def get_config():
    return _load_config()


@router.put("/smile-config")
async def put_config(body: SmileConfigBody):
    cfg = body.model_dump()
    _save_config(cfg)
    return cfg


@router.post("/smile-config/preview")
async def preview(body: SmileConfigBody):
    return preview_stats(_task_params(body.model_dump()), DATA_DIR)


@router.post("/smile-config/generate")
async def generate(body: SmileConfigBody):
    cfg = body.model_dump()
    _save_config(cfg)
    return generate_and_write(_task_params(cfg), DATA_DIR)
