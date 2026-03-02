import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.videos import router as videos_router
from api.transcripts import router as transcripts_router
from api.emotions import router as emotions_router
from api.annotations import router as annotations_router

DATA_DIR = Path(os.environ.get("VOICEOVER_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"

app = FastAPI(title="VoiceOver", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(videos_router, prefix="/api")
app.include_router(transcripts_router, prefix="/api")
app.include_router(emotions_router, prefix="/api")
app.include_router(annotations_router, prefix="/api")

if FRONTEND_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = FRONTEND_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIR / "index.html")
