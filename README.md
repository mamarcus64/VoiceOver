# VoiceOver

A media playback and annotation tool for the VOICES dataset. Browse interview videos, view time-aligned transcripts and emotion data (audio-based and eyegaze-based VAD), and annotate emotions via keyboard.

## Prerequisites

- **Python 3.10+**
- **Node.js 18+** and npm
- **ffmpeg** (for video processing)
- **yt-dlp** (for downloading videos from YouTube)

On Ubuntu/Debian:
```bash
sudo apt install python3 python3-venv nodejs npm ffmpeg
pip install yt-dlp
```

On macOS:
```bash
brew install python node ffmpeg yt-dlp
```

## Quick Start

```bash
# 1. Install dependencies and build
./setup.sh

# 2. Run the server
source venv/bin/activate
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000

# 3. Open in browser
# http://localhost:8000
```

## Development Mode

For hot-reloading during development, run frontend and backend separately:

```bash
# Terminal 1: Backend with auto-reload
source venv/bin/activate
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2: Frontend dev server (proxies API to backend)
cd frontend
npm run dev
# Open http://localhost:5173
```

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
├── data/             # All data files
│   ├── manifest.json     # Video catalog (5,151 entries)
│   ├── transcripts/      # Standardized JSON transcripts (per video)
│   ├── audio_vad/        # Audio emotion data (per video)
│   ├── eyegaze_vad/      # Eyegaze emotion data (per video, from GLASS)
│   ├── annotations/      # User annotations (per video, per annotator)
│   └── videos/           # Downloaded .mp4 files
└── scripts/          # Data preparation and processing scripts
```

## Data

### Included Data (< 1 GB)
- **manifest.json**: catalog of all 5,151 videos with YouTube links
- **transcripts/**: standardized JSON transcripts with word-level timestamps
- **audio_vad/**: audio-based valence/arousal/dominance per video

### Generated Data
- **eyegaze_vad/**: must be generated using GLASS (see below)

### Videos
Videos are **not** included (5,000+ files). Use the browser UI to selectively download videos from YouTube, or use yt-dlp directly.

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
| `/api/annotations` | GET | Load annotations |
| `/api/annotations` | POST | Save annotations |
| `/api/annotations/annotators` | GET | List annotators for a video |
