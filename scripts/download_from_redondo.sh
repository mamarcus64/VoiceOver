#!/usr/bin/env bash
# Copy videos from redondo via scp instead of downloading from YouTube.
#
# Usage:
#   ./scripts/download_from_redondo.sh --limit 200
#   ./scripts/download_from_redondo.sh --from-manifest data/recall_task_manifest.json
#
# Idempotent: skips videos whose .mp4 already exists in data/videos/.
# Source:     redondo:/home/mjma/voices/test_data/videos/{video_id}.mp4

set -euo pipefail

###############################################################################
# Configuration
###############################################################################

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAMPLE_JSON="${REPO_ROOT}/data/annotation_sample.json"
VIDEO_DIR="${REPO_ROOT}/data/videos"
MANIFEST="${REPO_ROOT}/data/manifest.json"

REMOTE_HOST="redondo"
REMOTE_DIR="/home/mjma/voices/test_data/videos"

###############################################################################
# Parse arguments
###############################################################################

LIMIT=""
FROM_MANIFEST=""

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
        --from-manifest)
            FROM_MANIFEST="$2"
            shift 2
            ;;
        --from-manifest=*)
            FROM_MANIFEST="${1#*=}"
            shift
            ;;
        --remote-host)
            REMOTE_HOST="$2"
            shift 2
            ;;
        --remote-host=*)
            REMOTE_HOST="${1#*=}"
            shift
            ;;
        --remote-dir)
            REMOTE_DIR="$2"
            shift 2
            ;;
        --remote-dir=*)
            REMOTE_DIR="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 --limit N" >&2
            echo "       $0 --from-manifest path/to/task_manifest.json" >&2
            echo "       $0 --from-manifest ... --remote-host redondo --remote-dir /path/to/videos" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${LIMIT}" && -z "${FROM_MANIFEST}" ]]; then
    echo "Error: either --limit N or --from-manifest PATH is required." >&2
    echo "Usage: $0 --limit 200" >&2
    echo "       $0 --from-manifest data/recall_task_manifest.json" >&2
    exit 1
fi

if [[ -n "${LIMIT}" ]]; then
    if ! [[ "${LIMIT}" =~ ^[0-9]+$ ]] || [[ "${LIMIT}" -lt 1 ]]; then
        echo "Error: --limit must be a positive integer, got '${LIMIT}'" >&2
        exit 1
    fi
fi

if [[ -n "${FROM_MANIFEST}" && ! -f "${FROM_MANIFEST}" ]]; then
    if [[ -f "${REPO_ROOT}/${FROM_MANIFEST}" ]]; then
        FROM_MANIFEST="${REPO_ROOT}/${FROM_MANIFEST}"
    else
        echo "Error: manifest file not found: ${FROM_MANIFEST}" >&2
        exit 1
    fi
fi

###############################################################################
# Pre-flight checks
###############################################################################

command -v python3 >/dev/null 2>&1 || { echo "Error: python3 not found in PATH" >&2; exit 1; }
command -v scp    >/dev/null 2>&1 || { echo "Error: scp not found in PATH" >&2; exit 1; }
[[ -f "${SAMPLE_JSON}" ]] || { echo "Error: ${SAMPLE_JSON} not found." >&2; exit 1; }

mkdir -p "${VIDEO_DIR}"

# Quick connectivity check
echo "Checking SSH connectivity to ${REMOTE_HOST}..."
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_HOST}" true 2>/dev/null; then
    echo "Error: cannot reach ${REMOTE_HOST} via SSH (passwordless auth required)." >&2
    exit 1
fi
echo "  Connected."

###############################################################################
# Build video ID list
###############################################################################

# Emit one video_id per line — no URLs needed (scp uses video_id directly).
VIDEO_IDS=()
if [[ -n "${FROM_MANIFEST}" ]]; then
    _TMPPY=$(mktemp /tmp/voiceover_dl_XXXXXX.py)
    _TMPOUT=$(mktemp /tmp/voiceover_dl_XXXXXX.txt)
    cat > "${_TMPPY}" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    task_manifest = json.load(f)
seen = set()
for task in task_manifest.get("tasks", []):
    vid = task.get("video_id", "")
    if vid and vid not in seen:
        seen.add(vid)
        print(vid)
PYEOF
    python3 "${_TMPPY}" "${FROM_MANIFEST}" > "${_TMPOUT}"
    rm -f "${_TMPPY}"
    while IFS= read -r line; do
        VIDEO_IDS+=("$line")
    done < "${_TMPOUT}"
    rm -f "${_TMPOUT}"
    TOTAL=${#VIDEO_IDS[@]}
    echo "Task manifest: ${FROM_MANIFEST}"
    echo "Unique videos needed: ${TOTAL}"
else
    _TMPPY=$(mktemp /tmp/voiceover_dl_XXXXXX.py)
    _TMPOUT=$(mktemp /tmp/voiceover_dl_XXXXXX.txt)
    cat > "${_TMPPY}" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
limit = int(sys.argv[2])
for e in data[:limit]:
    print(e["id"])
PYEOF
    python3 "${_TMPPY}" "${SAMPLE_JSON}" "${LIMIT}" > "${_TMPOUT}"
    rm -f "${_TMPPY}"
    while IFS= read -r line; do
        VIDEO_IDS+=("$line")
    done < "${_TMPOUT}"
    rm -f "${_TMPOUT}"
    TOTAL=${#VIDEO_IDS[@]}
    echo "Annotation sample size: ${TOTAL} videos (limit=${LIMIT})"
fi

###############################################################################
# Copy loop
###############################################################################

DOWNLOADED=0
SKIPPED=0
FAILED=0
FAILED_IDS=()

for VIDEO_ID in "${VIDEO_IDS[@]}"; do
    OUT_FILE="${VIDEO_DIR}/${VIDEO_ID}.mp4"

    if [[ -f "${OUT_FILE}" ]]; then
        (( SKIPPED++ )) || true
        continue
    fi

    REMOTE_FILE="${REMOTE_HOST}:${REMOTE_DIR}/${VIDEO_ID}.mp4"
    echo "  [CP] ${VIDEO_ID}  ($(( DOWNLOADED + SKIPPED + FAILED + 1 ))/${TOTAL})  ${REMOTE_FILE}"

    if scp -q "${REMOTE_FILE}" "${OUT_FILE}"; then
        (( DOWNLOADED++ )) || true
    else
        echo "  [FAIL] ${VIDEO_ID} — not found or scp error"
        (( FAILED++ )) || true
        FAILED_IDS+=("${VIDEO_ID}")
        rm -f "${OUT_FILE}"   # remove partial file
    fi
done

###############################################################################
# Update manifest downloaded flags
###############################################################################

python3 - "${MANIFEST}" "${VIDEO_DIR}" <<'PYEOF'
import json, os, sys
manifest_path, video_dir = sys.argv[1], sys.argv[2]
video_files = set(os.path.splitext(f)[0] for f in os.listdir(video_dir) if f.endswith('.mp4'))
with open(manifest_path) as f:
    manifest = json.load(f)
for entry in manifest:
    entry['downloaded'] = entry['id'] in video_files
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)
print("Manifest updated: " + str(sum(1 for e in manifest if e['downloaded'])) + " downloaded")
PYEOF

###############################################################################
# Summary
###############################################################################

echo ""
MODE_DESC="${FROM_MANIFEST:+manifest=$(basename "${FROM_MANIFEST}")}${LIMIT:+limit=${LIMIT}}"
echo "========================================"
echo "  COPY SUMMARY  (${MODE_DESC})"
echo "  Source: ${REMOTE_HOST}:${REMOTE_DIR}"
echo "========================================"
echo "  Copied:            ${DOWNLOADED}"
echo "  Skipped (cached):  ${SKIPPED}"
echo "  Failed/missing:    ${FAILED}"
if (( ${#FAILED_IDS[@]} > 0 )); then
echo "  Failed IDs:        ${FAILED_IDS[*]}"
fi
echo "========================================"
