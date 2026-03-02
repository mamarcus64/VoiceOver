#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# GLASS Eyegaze Emotion Pipeline — Batch Processing
#
# Two-step pipeline:
#   1. OpenFace feature extraction  (conda: openface_env)
#   2. GLASS VAD inference           (conda: glass_env)
#
# Usage:
#   ./run_glass_batch.sh          # process all videos
#   ./run_glass_batch.sh 10       # dry-run: first 10 videos only
###############################################################################

# ─── GPU / Parallelism ──────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES="0"
NUM_WORKERS=4
NUM_GPUS_PER_WORKER=1

# ─── Paths (edit as needed) ─────────────────────────────────────────────────
VIDEO_DIR="/home/mjma/voices/test_data/videos"
OUTPUT_DIR="/home/mjma/voices/VoiceOver/data/eyegaze_vad"
OPENFACE_CACHE="${OUTPUT_DIR}/_openface_cache"
OPENFACE_PRECOMPUTED="/home/mjma/voices/threadward_results"
OPENFACE_BIN="/home/mjma/voices/OpenFace/build/bin/FeatureExtraction"
GLASS_REPO="/home/mjma/voices/GLASS"
GLASS_MODEL="${GLASS_REPO}/emotion_prediction/best_model/model.ckpt"

# ─── Conda environments ────────────────────────────────────────────────────
OPENFACE_ENV="openface_env"
GLASS_ENV="glass_env"

# ─── Conda setup ────────────────────────────────────────────────────────────
# Source conda so we can use `conda run` reliably.
CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

###############################################################################
# Helpers
###############################################################################

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { log "FATAL: $*"; exit 1; }

###############################################################################
# Pre-flight checks
###############################################################################

[[ -d "${VIDEO_DIR}" ]]      || die "Video directory not found: ${VIDEO_DIR}"
[[ -x "${OPENFACE_BIN}" ]]   || die "OpenFace binary not found/executable: ${OPENFACE_BIN}"
[[ -d "${GLASS_REPO}" ]]     || die "GLASS repo not found: ${GLASS_REPO}"
[[ -f "${GLASS_MODEL}" ]]    || die "GLASS model checkpoint not found: ${GLASS_MODEL}"

mkdir -p "${OUTPUT_DIR}" "${OPENFACE_CACHE}"

export CUDA_VISIBLE_DEVICES

###############################################################################
# Collect video list from precomputed OpenFace results (threadward_results)
###############################################################################

mapfile -t ALL_VIDEO_IDS < <(find "${OPENFACE_PRECOMPUTED}" -maxdepth 2 -name 'result.csv' -type f \
    | sed 's|.*/\([^/]*\)/result.csv|\1|' | sort)
TOTAL=${#ALL_VIDEO_IDS[@]}

if [[ ${TOTAL} -eq 0 ]]; then
    die "No result.csv files found in ${OPENFACE_PRECOMPUTED}"
fi

# Dry-run: limit to first N videos if a number is passed as $1
LIMIT=${1:-${TOTAL}}
if [[ "${LIMIT}" =~ ^[0-9]+$ ]]; then
    if (( LIMIT < TOTAL )); then
        log "DRY-RUN MODE: processing first ${LIMIT} of ${TOTAL} videos"
        ALL_VIDEO_IDS=("${ALL_VIDEO_IDS[@]:0:${LIMIT}}")
        TOTAL=${#ALL_VIDEO_IDS[@]}
    fi
else
    die "Invalid argument '${LIMIT}'. Pass a number for dry-run mode or omit for all."
fi

log "Found ${TOTAL} videos with precomputed OpenFace features in ${OPENFACE_PRECOMPUTED}"
log "Skipping OpenFace extraction (Step 1) — all features already exist."

###############################################################################
# Step 2 — GLASS VAD inference
###############################################################################

log "===== GLASS VAD Inference ====="

GL_PROCESSED=0
GL_SKIPPED=0
GL_FAILED=0
GL_FAIL_LIST=()

for (( i=0; i<TOTAL; i++ )); do
    VIDEO_ID="${ALL_VIDEO_IDS[$i]}"
    RESULT_CSV="${OPENFACE_PRECOMPUTED}/${VIDEO_ID}/result.csv"
    PRED_CSV="${OUTPUT_DIR}/${VIDEO_ID}.csv"

    # Checkpointing: skip if predictions already exist
    if [[ -f "${PRED_CSV}" ]]; then
        (( GL_SKIPPED++ )) || true
        continue
    fi

    log "[$(( i+1 ))/${TOTAL}] GLASS inference: ${VIDEO_ID}"

    if conda run -n "${GLASS_ENV}" --no-capture-output \
        python "${GLASS_REPO}/scripts/inference_pipeline.py" \
            --csv "${RESULT_CSV}" \
            --model "${GLASS_MODEL}" \
            --output "${PRED_CSV}" \
        2>&1 | tail -5; then
        if [[ -f "${PRED_CSV}" ]]; then
            (( GL_PROCESSED++ )) || true
        else
            log "  WARNING: GLASS exited 0 but predictions.csv not created for ${VIDEO_ID}"
            (( GL_FAILED++ )) || true
            GL_FAIL_LIST+=("${VIDEO_ID}")
        fi
    else
        log "  ERROR: GLASS inference failed for ${VIDEO_ID}"
        (( GL_FAILED++ )) || true
        GL_FAIL_LIST+=("${VIDEO_ID}")
    fi
done

log "GLASS inference done — processed: ${GL_PROCESSED}, skipped: ${GL_SKIPPED}, failed: ${GL_FAILED}"

###############################################################################
# Summary
###############################################################################

echo ""
echo "========================================================================"
echo "  GLASS BATCH PROCESSING SUMMARY"
echo "========================================================================"
echo "  Total videos:         ${TOTAL}"
echo "  OpenFace source:      ${OPENFACE_PRECOMPUTED}"
echo ""
echo "  GLASS inference:"
echo "    Processed:          ${GL_PROCESSED}"
echo "    Skipped (cached):   ${GL_SKIPPED}"
echo "    Failed:             ${GL_FAILED}"
if (( ${#GL_FAIL_LIST[@]} > 0 )); then
echo "    Failed IDs:         ${GL_FAIL_LIST[*]}"
fi
echo ""
echo "  Output directory:     ${OUTPUT_DIR}"
echo "========================================================================"
