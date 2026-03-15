#!/usr/bin/env bash
# Download the first N videos from data/annotation_sample.json.
#
# Usage:
#   ./scripts/download_videos.sh --limit 200
#
# Idempotent: skips tapes whose .mp4 already exists in data/videos/.
# Resumable:  re-running with the same or larger --limit picks up where it left off.
# Order:      fixed by annotation_sample.json — increasing --limit only appends
#             new downloads at the end; existing ones are never re-ordered.

set -euo pipefail

###############################################################################
# Configuration
###############################################################################

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAMPLE_JSON="${REPO_ROOT}/data/annotation_sample.json"
VIDEO_DIR="${REPO_ROOT}/data/videos"
MANIFEST="${REPO_ROOT}/data/manifest.json"
VENV="${REPO_ROOT}/venv"

# yt-dlp download format: best mp4 up to 1080p, fallback to best available
YT_FORMAT="bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best"

###############################################################################
# Parse arguments
###############################################################################

LIMIT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --limit=*)
            LIMIT="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 --limit N" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${LIMIT}" ]]; then
    echo "Error: --limit N is required." >&2
    echo "Usage: $0 --limit 200" >&2
    exit 1
fi

if ! [[ "${LIMIT}" =~ ^[0-9]+$ ]] || [[ "${LIMIT}" -lt 1 ]]; then
    echo "Error: --limit must be a positive integer, got '${LIMIT}'" >&2
    exit 1
fi

###############################################################################
# Pre-flight checks
###############################################################################

# Activate the project venv so yt-dlp and python3 use Python 3.10+
if [[ -f "${VENV}/bin/activate" ]]; then
    source "${VENV}/bin/activate"
else
    echo "Warning: venv not found at ${VENV} — run setup.sh first to create it with Python 3.10+" >&2
fi

command -v yt-dlp >/dev/null 2>&1 || { echo "Error: yt-dlp not found. Install it with: pip install yt-dlp" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "Error: python3 not found in PATH" >&2; exit 1; }
[[ -f "${SAMPLE_JSON}" ]] || { echo "Error: ${SAMPLE_JSON} not found. Run scripts/build_annotation_sample.py first." >&2; exit 1; }

mkdir -p "${VIDEO_DIR}"

###############################################################################
# Extract first N entries from annotation_sample.json
###############################################################################

# Emit one "id|youtube_url" line per tape, limited to LIMIT entries.
mapfile -t ENTRIES < <(python3 - "${SAMPLE_JSON}" "${LIMIT}" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
limit = int(sys.argv[2])
for e in data[:limit]:
    url = e.get("youtube_url", "")
    if url and url != "NULL":
        print(f"{e['id']}|{url}")
    else:
        print(f"{e['id']}|NULL")
PYEOF
)

TOTAL=${#ENTRIES[@]}
echo "Annotation sample size: ${TOTAL} tapes (limit=${LIMIT})"

###############################################################################
# Download loop
###############################################################################

DOWNLOADED=0
SKIPPED=0
FAILED=0
FAILED_IDS=()

for entry in "${ENTRIES[@]}"; do
    TAPE_ID="${entry%%|*}"
    URL="${entry##*|}"
    OUT_FILE="${VIDEO_DIR}/${TAPE_ID}.mp4"

    # Idempotency: skip if already downloaded
    if [[ -f "${OUT_FILE}" ]]; then
        (( SKIPPED++ )) || true
        continue
    fi

    if [[ "${URL}" == "NULL" || -z "${URL}" ]]; then
        echo "  [SKIP] ${TAPE_ID} — no YouTube URL"
        (( SKIPPED++ )) || true
        continue
    fi

    echo "  [DL] ${TAPE_ID}  ($(( DOWNLOADED + SKIPPED + FAILED + 1 ))/${TOTAL})  ${URL}"

    if yt-dlp \
        --format "${YT_FORMAT}" \
        --output "${OUT_FILE}" \
        --no-playlist \
        --quiet \
        --progress \
        "${URL}"; then
        (( DOWNLOADED++ )) || true
    else
        echo "  [FAIL] ${TAPE_ID}"
        (( FAILED++ )) || true
        FAILED_IDS+=("${TAPE_ID}")
    fi
done

###############################################################################
# Update manifest downloaded flags
###############################################################################

python3 - "${MANIFEST}" "${VIDEO_DIR}" <<'PYEOF'
import json, os, sys
manifest_path, video_dir = sys.argv[1], sys.argv[2]
video_files = {os.path.splitext(f)[0] for f in os.listdir(video_dir) if f.endswith('.mp4')}
with open(manifest_path) as f:
    manifest = json.load(f)
for entry in manifest:
    entry['downloaded'] = entry['id'] in video_files
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)
print(f"Manifest updated: {sum(1 for e in manifest if e['downloaded'])} downloaded")
PYEOF

###############################################################################
# Summary
###############################################################################

echo ""
echo "========================================"
echo "  DOWNLOAD SUMMARY  (limit=${LIMIT})"
echo "========================================"
echo "  Downloaded:        ${DOWNLOADED}"
echo "  Skipped (cached):  ${SKIPPED}"
echo "  Failed:            ${FAILED}"
if (( ${#FAILED_IDS[@]} > 0 )); then
echo "  Failed IDs:        ${FAILED_IDS[*]}"
fi
echo "========================================"
