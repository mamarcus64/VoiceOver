#!/usr/bin/env python3
"""
Compute per-video temporal offsets between archive XML transcript timestamps
(tape-time) and YouTube video files (video-time).

Runs a lightweight ASR pass (faster-whisper) on the first N seconds of each
video, then aligns ASR words with XML words via median time-difference
estimation.  Uses concurrent workers to overlap ffmpeg extraction and GPU
inference for high throughput.

Output: VoiceOver/data/transcript_offsets.json  —  {video_id: offset_ms, ...}
"""

# ── GPU selection (must precede any CUDA imports) ──────────────────────────
CUDA_DEVICE = "2"

import os
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE

import argparse
import json
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np
from faster_whisper import WhisperModel

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
XML_DIR = Path("/home/mjma/voices/test_data/transcripts")
VIDEO_DIR = Path("/home/mjma/voices/test_data/videos")
OUTPUT_PATH = DATA_DIR / "transcript_offsets.json"
MANIFEST_PATH = DATA_DIR / "manifest.json"

SPEAKER_TAG_RE = re.compile(
    r"^\s*[A-Za-z][A-Za-z0-9 ._-]{0,50}?\s*:\s*", re.DOTALL
)
BRACKET_RE = re.compile(r"\[.*?\]")
PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def parse_args():
    p = argparse.ArgumentParser(description="Compute transcript time offsets via ASR alignment")
    p.add_argument("--limit", type=int, default=None, help="Only process first N videos")
    p.add_argument("--asr-seconds", type=int, default=300, help="Seconds of audio to transcribe (default 300)")
    p.add_argument("--model", default="base", help="Whisper model size (default: base)")
    p.add_argument("--output", default=str(OUTPUT_PATH))
    p.add_argument("--resume", action="store_true", help="Skip videos already in output file")
    p.add_argument("--workers", type=int, default=8,
                   help="Concurrent workers (overlap ffmpeg + GPU inference, default 8)")
    p.add_argument("--video-list", type=str, default=None,
                   help="Path to a text file with one video_id per line (overrides auto-detection)")
    return p.parse_args()


# ── XML parsing ────────────────────────────────────────────────────────────

def extract_xml_words(xml_path: Path) -> list[tuple[str, int]]:
    """Parse XML transcript, return [(word_text, ms), ...] in order."""
    tree = ET.parse(xml_path)
    words = []
    for span in tree.findall(".//span"):
        m = span.attrib.get("m")
        raw = (span.text or "").strip()
        if not m or not raw:
            continue
        try:
            ms = int(m)
        except ValueError:
            continue

        text = SPEAKER_TAG_RE.sub("", raw).strip()
        text = BRACKET_RE.sub("", text).strip()
        if not text:
            continue

        for token in text.split():
            clean = PUNCT_RE.sub("", token).lower().strip()
            if clean:
                words.append((clean, ms))
    return words


# ── Audio extraction ───────────────────────────────────────────────────────

def extract_audio_segment(video_path: Path, duration_s: int, tmp_dir: Path) -> Path:
    """Extract first N seconds of audio as 16kHz mono WAV."""
    out = tmp_dir / f"{video_path.stem}.wav"
    if out.exists():
        out.unlink()
    subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-i", str(video_path),
            "-t", str(duration_s),
            "-ac", "1", "-ar", "16000",
            "-f", "wav", str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


# ── ASR ────────────────────────────────────────────────────────────────────

def asr_words(model: WhisperModel, wav_path: Path) -> list[tuple[str, int]]:
    """Run ASR, return [(word_text, ms), ...]."""
    segments, _ = model.transcribe(
        str(wav_path),
        language="en",
        word_timestamps=True,
        vad_filter=True,
    )
    words = []
    for seg in segments:
        if seg.words is None:
            continue
        for w in seg.words:
            clean = PUNCT_RE.sub("", w.word).lower().strip()
            if clean:
                words.append((clean, int(w.start * 1000)))
    return words


# ── Offset estimation ─────────────────────────────────────────────────────

def compute_offset(
    xml_words: list[tuple[str, int]],
    asr_words_list: list[tuple[str, int]],
    search_window_ms: int = 60_000,
) -> dict:
    """
    Find the constant offset such that xml_ms ≈ asr_ms + offset.

    For each ASR word, scan a window of XML words for a text match and record
    the time difference.  Return the median difference as the offset estimate.
    """
    if not xml_words or not asr_words_list:
        return {"offset_ms": None, "n_matches": 0, "mad_ms": None}

    xml_idx = 0
    deltas = []

    for asr_text, asr_ms in asr_words_list:
        best_delta = None
        best_dist = float("inf")

        lo = xml_idx
        for j in range(max(0, lo - 50), len(xml_words)):
            xt, xms = xml_words[j]
            if xt != asr_text:
                continue
            delta = xms - asr_ms
            if abs(delta) > search_window_ms:
                if xms - asr_ms > search_window_ms:
                    break
                continue
            dist = abs(delta - (deltas[-1] if deltas else delta))
            if dist < best_dist:
                best_dist = dist
                best_delta = delta
                xml_idx = j

        if best_delta is not None:
            deltas.append(best_delta)

    if len(deltas) < 3:
        return {"offset_ms": None, "n_matches": len(deltas), "mad_ms": None}

    arr = np.array(deltas, dtype=np.float64)
    median = float(np.median(arr))

    inliers = arr[np.abs(arr - median) < 5000]
    if len(inliers) >= 3:
        offset = float(np.median(inliers))
        mad = float(np.median(np.abs(inliers - offset)))
    else:
        offset = median
        mad = float(np.median(np.abs(arr - median)))

    return {
        "offset_ms": round(offset),
        "n_matches": len(inliers) if len(inliers) >= 3 else len(deltas),
        "mad_ms": round(mad),
    }


# ── Per-video worker ──────────────────────────────────────────────────────

MIN_GOOD_MATCHES = 15

def process_one(vid, xml_words_cache, model, asr_seconds, tmp_dir):
    """Full pipeline for a single video. Returns (vid, result_dict, error).

    Tries with the requested duration first; if too few matches, retries with
    a longer window (up to 300s) to handle videos with long silent intros.
    """
    mp4_path = VIDEO_DIR / f"{vid}.mp4"

    xml_w = xml_words_cache.get(vid)
    if xml_w is None:
        xml_w = extract_xml_words(XML_DIR / f"{vid}.xml")

    durations = [asr_seconds]
    if asr_seconds < 300:
        durations.append(300)

    for dur in durations:
        wav = extract_audio_segment(mp4_path, dur, tmp_dir)
        asr_w = asr_words(model, wav)
        wav.unlink(missing_ok=True)

        result = compute_offset(xml_w, asr_w)
        if result["n_matches"] >= MIN_GOOD_MATCHES and (
            result["mad_ms"] is None or result["mad_ms"] <= 1000
        ):
            break

    return vid, result, None


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path("/tmp/transcript_offset_wavs")
    tmp_dir.mkdir(exist_ok=True)

    # ── Build video list ──────────────────────────────────────────────────
    if args.video_list:
        with open(args.video_list) as f:
            video_ids = [line.strip() for line in f if line.strip()]
        video_ids = [v for v in video_ids
                     if (XML_DIR / f"{v}.xml").exists() and (VIDEO_DIR / f"{v}.mp4").exists()]
    else:
        manifest_ids = None
        if MANIFEST_PATH.exists():
            with open(MANIFEST_PATH) as f:
                manifest_ids = {v["id"] for v in json.load(f)}

        xml_stems = {p.stem for p in XML_DIR.glob("*.xml")}
        mp4_stems = {p.stem for p in VIDEO_DIR.glob("*.mp4")}
        candidates = xml_stems & mp4_stems
        if manifest_ids is not None:
            candidates &= manifest_ids
        video_ids = sorted(candidates)

    if args.limit:
        video_ids = video_ids[: args.limit]

    existing = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        before = len(video_ids)
        video_ids = [v for v in video_ids if v not in existing]
        print(f"Resuming: {before - len(video_ids)} already done, {len(video_ids)} remaining")

    print(f"Videos to process: {len(video_ids)}")
    print(f"Whisper model:     {args.model}")
    print(f"ASR window:        first {args.asr_seconds}s")
    print(f"Workers:           {args.workers}")
    print(f"Output:            {output_path}")
    print(f"GPU:               CUDA_VISIBLE_DEVICES={CUDA_DEVICE}")
    print()

    if not video_ids:
        print("Nothing to do.")
        return

    # ── Pre-parse all XMLs (fast, pure CPU) ───────────────────────────────
    print("Pre-parsing XML transcripts...", end=" ", flush=True)
    t0 = time.time()
    xml_cache = {}
    with ThreadPoolExecutor(max_workers=min(16, len(video_ids))) as pool:
        futs = {pool.submit(extract_xml_words, XML_DIR / f"{v}.xml"): v for v in video_ids}
        for fut in as_completed(futs):
            xml_cache[futs[fut]] = fut.result()
    print(f"{len(xml_cache)} files in {time.time() - t0:.1f}s")

    # ── Load model ────────────────────────────────────────────────────────
    print("Loading Whisper model...", end=" ", flush=True)
    t0 = time.time()
    model = WhisperModel(args.model, device="cuda", compute_type="float16")
    print(f"ready in {time.time() - t0:.1f}s\n")

    # ── Process concurrently ──────────────────────────────────────────────
    offsets = dict(existing)
    flagged = []
    done = 0
    save_lock = Lock()
    t_start = time.time()

    def _worker(vid):
        return process_one(vid, xml_cache, model, args.asr_seconds, tmp_dir)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_worker, vid): vid for vid in video_ids}

        for fut in as_completed(futs):
            vid = futs[fut]
            done += 1

            try:
                vid, result, _ = fut.result()
                off = result["offset_ms"]
                n = result["n_matches"]
                mad = result["mad_ms"]

                if off is None:
                    status = f"FAILED ({n} matches)"
                    flagged.append(vid)
                else:
                    if mad is not None and mad > 1000:
                        status = f"WARN MAD={mad}ms"
                        flagged.append(vid)
                    elif n < 15:
                        status = f"WARN {n} matches"
                        flagged.append(vid)
                    else:
                        status = "OK"
                    with save_lock:
                        offsets[vid] = off

                elapsed = time.time() - t_start
                rate = done / elapsed
                eta = (len(video_ids) - done) / rate if rate > 0 else 0
                print(f"[{done}/{len(video_ids)}] {vid}: "
                      f"offset={off}ms  n={n}  MAD={mad}ms  {status}  "
                      f"({rate:.1f} vid/s, ETA {eta/60:.0f}m)",
                      flush=True)

            except Exception as e:
                elapsed = time.time() - t_start
                print(f"[{done}/{len(video_ids)}] {vid}: ERROR {e}", flush=True)
                flagged.append(vid)

            if done % 100 == 0:
                with save_lock:
                    with open(output_path, "w") as f:
                        json.dump(offsets, f, indent=2)

    with open(output_path, "w") as f:
        json.dump(offsets, f, indent=2)

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done in {total_time/60:.1f} minutes ({total_time/len(video_ids):.2f}s/video)")
    print(f"Offsets computed: {len(offsets)}")
    if flagged:
        print(f"Flagged ({len(flagged)}): {flagged[:20]}{'...' if len(flagged) > 20 else ''}")
    vals = [v for v in offsets.values() if v is not None]
    if vals:
        print(f"Offset range: {min(vals)}ms .. {max(vals)}ms  (median={int(np.median(vals))}ms)")


if __name__ == "__main__":
    main()
