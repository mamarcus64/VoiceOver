from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, field_validator

from services.annotation_store import (
    LABEL_BY_KEY,
    VALID_LABELS,
    save,
    load,
    load_all,
    list_annotators,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AnnotationEntry(BaseModel):
    start_sec: float
    end_sec: float
    label: str
    key: int

    @field_validator("label")
    @classmethod
    def _validate_label(cls, v: str) -> str:
        if v not in VALID_LABELS:
            raise ValueError(
                f"Invalid label '{v}'. Must be one of {sorted(VALID_LABELS)}"
            )
        return v

    @field_validator("key")
    @classmethod
    def _validate_key(cls, v: int) -> int:
        if v not in LABEL_BY_KEY:
            raise ValueError(
                f"Invalid key {v}. Must be one of {sorted(LABEL_BY_KEY)}"
            )
        return v


class SaveAnnotationsRequest(BaseModel):
    video_id: str
    annotator: str
    annotations: list[AnnotationEntry]


class SaveAnnotationsResponse(BaseModel):
    video_id: str
    annotator: str
    created_at: str
    annotations: list[AnnotationEntry]


class AnnotatorsResponse(BaseModel):
    video_id: str
    annotators: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/annotations", response_model=SaveAnnotationsResponse)
async def save_annotations(body: SaveAnnotationsRequest):
    result = save(
        video_id=body.video_id,
        annotator=body.annotator,
        annotations_list=[entry.model_dump() for entry in body.annotations],
    )
    return result


@router.get("/annotations")
async def get_annotations(
    video_id: str = Query(..., description="Video ID to look up"),
    annotator: str | None = Query(None, description="Optional annotator name"),
):
    if annotator is not None:
        data = load(video_id, annotator)
        if data is None:
            raise HTTPException(
                status_code=404,
                detail=f"No annotations found for annotator '{annotator}' on video '{video_id}'",
            )
        return data

    return load_all(video_id)


@router.get("/annotations/annotators", response_model=AnnotatorsResponse)
async def get_annotators(
    video_id: str = Query(..., description="Video ID to look up"),
):
    return AnnotatorsResponse(
        video_id=video_id,
        annotators=list_annotators(video_id),
    )
