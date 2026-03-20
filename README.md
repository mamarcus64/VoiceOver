# VoiceOver

A media playback and annotation tool for the VOICES dataset. Browse interview videos, view time-aligned transcripts and emotion data (audio-based and eyegaze-based VAD), and annotate emotions via keyboard.

## Quick Start

```bash
bash setup.sh
bash start.sh
```

Then open **http://localhost:1945** in your browser.

---

<details>
<summary><strong>Full Setup &amp; Troubleshooting</strong></summary>

### Prerequisites

- **Git LFS** (data files use Git Large File Storage; **they are not auto-downloaded** on `git pull` in a normal setup—see below)
- **Python 3.10+**
- **Node.js 18+** and npm
- **ffmpeg** (for video processing)
- **yt-dlp** (for downloading videos from YouTube)

On Ubuntu/Debian:
```bash
sudo apt install git-lfs python3 python3-venv nodejs npm ffmpeg
pip install yt-dlp
```

On macOS:
```bash
brew install git-lfs python node ffmpeg yt-dlp
```

### Step-by-step

```bash
# 0. Clone without downloading LFS blobs (recommended; saves many GB)
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/mamarcus64/VoiceOver.git
cd VoiceOver
# setup.sh configures this repo for lazy LFS; or run manually:
#   bash scripts/git-lfs-lazy.sh

# Pull only the LFS paths you need, e.g. manifest + one transcript type:
# git lfs pull --include='data/manifest.json'
# git lfs pull --include='data/transcripts_llm/*.json'
# Everything (very large):
# git lfs pull --include='*'

# 1. Install dependencies and build frontend
bash setup.sh

# 2. Run the server
bash start.sh

# 3. Open in browser
# http://localhost:1945
```

### Development Mode

For hot-reloading during development, run frontend and backend separately:

```bash
# Terminal 1: Backend with auto-reload
source venv/bin/activate
cd backend
uvicorn main:app --reload --port 1945

# Terminal 2: Frontend dev server (proxies API to backend)
cd frontend
npm run dev
# Open http://localhost:5173
```

</details>

---

## Project Structure

```
VoiceOver/
├── backend/          # FastAPI server
│   ├── api/          # REST endpoints (videos, transcripts, emotions, annotations)
│   ├── services/     # Business logic (video manager, annotation store)
│   └── main.py       # App entry point
├── frontend/         # React + Vite + TypeScript
│   └── src/
│       ├── components/   # VideoBrowser, PlayerPage, TranscriptTrack, etc.
│       └── hooks/        # usePlayback, useAnnotation
├── data/             # All data files (tracked with Git LFS)
│   ├── manifest.json     # Video catalog (5,151 entries)
│   ├── transcripts/      # Standardized JSON transcripts (per video)
│   ├── transcripts_llm/  # LLM-corrected transcripts
│   ├── audio_vad/        # Audio emotion data (per video)
│   ├── eyegaze_vad/      # Eyegaze emotion data (per video, from GLASS)
│   ├── eyegaze_vectors/  # Binocular gaze CSVs (local only — see data/eyegaze_vectors/README.md)
│   ├── smiling_segments/ # Pre-extracted AU12 smile segments (per video)
│   ├── annotations/      # User annotations (gitignored, per annotator)
│   └── videos/           # Downloaded .mp4 files (gitignored)
└── scripts/          # Data preparation and processing scripts
```

## Data

All data files under `data/` are tracked with **Git LFS** (~2.6 GB total). They download automatically on clone if Git LFS is installed.

| Directory | Files | Size | Description |
|---|---|---|---|
| `manifest.json` | 1 | 726 KB | Video catalog (5,151 entries with YouTube links) |
| `transcripts/` | 5,129 | 1.2 GB | Standardized JSON transcripts (word-level timestamps) |
| `transcripts_llm/` | 4,739 | 1.1 GB | LLM-corrected transcripts |
| `audio_vad/` | 5,100 | 195 MB | Audio-based valence/arousal/dominance |
| `eyegaze_vad/` | 3,998 | 110 MB | Eyegaze-based VAD (from GLASS) |
| `smiling_segments/` | 3,997 | 37 MB | Pre-extracted AU12 smile segments |

### Videos
Videos are **not** included (5,000+ files, too large). Use the browser UI to selectively download videos from YouTube, or use yt-dlp directly.

### Eyegaze vectors
Per-frame binocular gaze CSVs under `data/eyegaze_vectors/` are **not** in git (too large for LFS in this repo). Add them locally if you want the gaze figure in the player; see `data/eyegaze_vectors/README.md`. If your clone used to have them as tracked LFS files, **back up that folder before `git pull`** after this change, then restore it so existing copies stay on disk.

### Annotations
Annotations are user-generated and stored locally in `data/annotations/` (not tracked in git).

## Generating Eyegaze Emotion Data

The eyegaze VAD data requires running the GLASS pipeline (OpenFace + GLASS model). See the GLASS repository for installation.

```bash
# Configure GPU and paths at the top of the script, then run:
./scripts/run_glass_batch.sh        # process all videos
./scripts/run_glass_batch.sh 5      # dry-run on first 5 videos
```

## LLM Transcript Quality Pass

An optional LLM-based pass can improve speaker role assignments in transcripts:

```bash
# Requires VOICES_OPENROUTER_KEY environment variable
python scripts/llm_transcript_pass.py --limit 20 --diff-report --provider Cerebras

# After reviewing the diff report, run on all files:
python scripts/llm_transcript_pass.py --provider Cerebras
```

## Smiling Moments

The player includes a **Smiling Moments** mode for browsing smile events detected via OpenFace AU12 (lip corner puller). Click the "Smiling Moments" toggle in the player header to switch modes.

Five parameters can be adjusted in real-time (persisted across sessions):
- **Intensity threshold** (default 1.8): minimum mean AU12\_r
- **Merge distance** (default 0.5s): combine nearby segments
- **Min duration** (default 0.5s): discard short segments after merging
- **Context before** (default 3s): playback starts this many seconds before the smile
- **Context after** (default 2s): playback extends this many seconds after

To regenerate smiling segments from raw OpenFace data:
```bash
python scripts/extract_smiling_segments.py --workers 64
```

## Annotation

1. Open a video in the player
2. Enter your name in the annotator field
3. Hold number keys (1-5) while the video plays to annotate emotions:
   - **1**: Very Happy
   - **2**: Happy
   - **3**: Neutral
   - **4**: Sad
   - **5**: Very Sad
4. Click **Save** to persist annotations
5. Click **Load** to retrieve previous annotations

Annotations are saved as JSON files in `data/annotations/{video_id}/`.

## Validation

```bash
python scripts/validate_data.py
```

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/videos` | GET | List videos (search, pagination) |
| `/api/videos/{id}/stream` | GET | Stream video (Range support) |
| `/api/videos/{id}/download` | POST | Download video from YouTube |
| `/api/videos/{id}/status` | GET | Download status |
| `/api/videos/{id}/transcript` | GET | Standardized transcript |
| `/api/videos/{id}/audio-emotion` | GET | Audio VAD data |
| `/api/videos/{id}/eyegaze-emotion` | GET | Eyegaze VAD data |
| `/api/videos/{id}/eyegaze-vectors` | GET | Binocular gaze vectors (`gaze_0_*`, `gaze_1_*`) time-aligned to video |
| `/api/videos/{id}/smiling-segments` | GET | Pre-extracted AU12 smile segments |
| `/api/annotations` | GET | Load annotations |
| `/api/annotations` | POST | Save annotations |
| `/api/annotations/annotators` | GET | List annotators for a video |
