import json
import os
import stat
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from starlette.responses import StreamingResponse, Response

from services.video_manager import video_manager

router = APIRouter()

CHUNK_SIZE = 1024 * 1024  # 1 MiB


@router.get("/videos")
async def list_videos(
    search: Optional[str] = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    total, videos = video_manager.list_videos(search=search, offset=offset, limit=limit)
    return {"total": total, "videos": videos}


def _open_range_response(path: str, file_size: int, request: Request) -> Response:
    """Build a 200 or 206 response with optional Range header support."""
    range_header = request.headers.get("range")

    if range_header is None:
        def _iter_full():
            with open(path, "rb") as f:
                while chunk := f.read(CHUNK_SIZE):
                    yield chunk

        return StreamingResponse(
            _iter_full(),
            status_code=200,
            media_type="video/mp4",
            headers={
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
            },
        )

    try:
        units, rng = range_header.split("=", 1)
        assert units.strip().lower() == "bytes"
        start_str, end_str = rng.split("-", 1)
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
    except Exception:
        raise HTTPException(status_code=416, detail="Invalid Range header")

    if start >= file_size or end >= file_size or start > end:
        raise HTTPException(
            status_code=416,
            detail="Range not satisfiable",
            headers={"Content-Range": f"bytes */{file_size}"},
        )

    content_length = end - start + 1

    def _iter_range():
        with open(path, "rb") as f:
            f.seek(start)
            remaining = content_length
            while remaining > 0:
                to_read = min(CHUNK_SIZE, remaining)
                data = f.read(to_read)
                if not data:
                    break
                remaining -= len(data)
                yield data

    return StreamingResponse(
        _iter_range(),
        status_code=206,
        media_type="video/mp4",
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(content_length),
            "Accept-Ranges": "bytes",
        },
    )


@router.get("/videos/{video_id:path}/stream")
async def stream_video(video_id: str, request: Request):
    video_path = video_manager.get_video_path(video_id)
    if video_path is None:
        raise HTTPException(status_code=404, detail="Video file not found")

    file_size = os.stat(video_path)[stat.ST_SIZE]
    return _open_range_response(str(video_path), file_size, request)


@router.post("/videos/{video_id:path}/download")
async def download_video(video_id: str):
    entry = video_manager.get_video(video_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Video not found in manifest")

    try:
        status = video_manager.start_download(video_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return Response(
        content=json.dumps(status),
        status_code=202,
        media_type="application/json",
    )


@router.get("/videos/{video_id:path}/status")
async def video_status(video_id: str):
    status = video_manager.get_download_status(video_id)
    if status is None:
        entry = video_manager.get_video(video_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Video not found in manifest")
        file_exists = video_manager.get_video_path(video_id) is not None
        return {
            "video_id": video_id,
            "status": "complete" if file_exists else "none",
            "error": None,
        }
    return status
