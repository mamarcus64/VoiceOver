import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Optional


DATA_DIR = Path(os.environ.get(
    "VOICEOVER_DATA_DIR",
    Path(__file__).resolve().parent.parent.parent / "data",
))

FALLBACK_VIDEO_DIR = Path("/home/mjma/voices/test_data/videos")


class DownloadStatus:
    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETE = "complete"
    FAILED = "failed"


class VideoManager:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.manifest_path = data_dir / "manifest.json"
        self.videos_dir = data_dir / "videos"
        self.videos_dir.mkdir(parents=True, exist_ok=True)

        self._manifest: list[dict] = []
        self._manifest_by_id: dict[str, dict] = {}
        self._download_statuses: dict[str, dict] = {}
        self._lock = threading.Lock()

        self._load_manifest()

    def _load_manifest(self) -> None:
        with open(self.manifest_path, "r") as f:
            self._manifest = json.load(f)
        self._manifest_by_id = {v["id"]: v for v in self._manifest}

    def _save_manifest(self) -> None:
        with open(self.manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)
            f.write("\n")

    def list_videos(
        self,
        search: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[int, list[dict]]:
        if search:
            filtered = [v for v in self._manifest if search in v["id"]]
        else:
            filtered = self._manifest

        total = len(filtered)
        page = filtered[offset : offset + limit]
        return total, page

    def get_video(self, video_id: str) -> Optional[dict]:
        return self._manifest_by_id.get(video_id)

    def get_video_path(self, video_id: str) -> Optional[Path]:
        """Return the path to the video file, checking the primary directory
        first, then the fallback (test_data) directory."""
        primary = self.videos_dir / f"{video_id}.mp4"
        if primary.is_file():
            return primary

        fallback = FALLBACK_VIDEO_DIR / f"{video_id}.mp4"
        if fallback.is_file():
            return fallback

        return None

    def get_download_status(self, video_id: str) -> Optional[dict]:
        with self._lock:
            return self._download_statuses.get(video_id)

    def start_download(self, video_id: str) -> dict:
        entry = self._manifest_by_id.get(video_id)
        if entry is None:
            raise ValueError(f"Video {video_id} not found in manifest")

        with self._lock:
            existing = self._download_statuses.get(video_id)
            if existing and existing["status"] in (
                DownloadStatus.PENDING,
                DownloadStatus.DOWNLOADING,
            ):
                return existing

            status = {
                "video_id": video_id,
                "status": DownloadStatus.PENDING,
                "error": None,
            }
            self._download_statuses[video_id] = status

        thread = threading.Thread(
            target=self._download_worker,
            args=(video_id, entry["youtube_url"]),
            daemon=True,
        )
        thread.start()
        return status

    def _download_worker(self, video_id: str, youtube_url: str) -> None:
        output_path = self.videos_dir / f"{video_id}.mp4"

        with self._lock:
            self._download_statuses[video_id]["status"] = DownloadStatus.DOWNLOADING

        try:
            proc = subprocess.Popen(
                [
                    "yt-dlp",
                    "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                    "--merge-output-format", "mp4",
                    "-o", str(output_path),
                    "--no-playlist",
                    youtube_url,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _, stderr = proc.communicate()

            if proc.returncode != 0:
                raise RuntimeError(stderr.decode(errors="replace").strip())

            with self._lock:
                self._download_statuses[video_id]["status"] = DownloadStatus.COMPLETE

            entry = self._manifest_by_id.get(video_id)
            if entry:
                entry["downloaded"] = True
                self._save_manifest()

        except Exception as exc:
            with self._lock:
                st = self._download_statuses[video_id]
                st["status"] = DownloadStatus.FAILED
                st["error"] = str(exc)


video_manager = VideoManager()
