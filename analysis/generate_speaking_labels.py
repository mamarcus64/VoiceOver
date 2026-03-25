#!/usr/bin/env python3
"""
Generate speaking/pausing label spans for each transcript.

Output: data/speaking_labels/{video_id}.csv per transcript (columns: start_ms, end_ms, label).

Labels
------
  interviewee_speaking : interviewee is actively speaking
  interviewer_speaking : interviewer is actively speaking
  pause_filler         : silence at recording start/end, or any pause > 30 s
  pause_question       : silence after interviewer (waiting for interviewee)
  pause_answered       : silence after interviewee (waiting for interviewer)
  pause_narrative      : long intra-turn pause (interviewee -> interviewee)

Algorithm
---------
All words across all segments are flattened into a single time-ordered list
(segment boundaries are ignored; only word timestamps and speaker tags matter).

For each word i with gap G to the next word:
  - G <= max_word_duration  → word speaks from word.ms to next_word.ms (no pause)
  - G >  max_word_duration  → word speaks for max_word_duration // 2 ms (buffer),
                               then [word_end, next_word.ms) is a pause classified
                               by the surrounding speaker context.

Pause classification (after buffer):
  interviewee → interviewer : pause_answered
  interviewer → *           : pause_question
  interviewee → interviewee : pause_narrative
  Any pause > long_pause_threshold (default 30 s): pause_filler

Before the first word and after the last word: pause_filler.
Consecutive spans with the same label are merged.

Resumable: output files that already exist are skipped.
"""

import argparse
import csv
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts_llm"
OUTPUT_DIR = DATA_DIR / "speaking_labels"


def classify_pause(
    prev_speaker: str,
    next_speaker: str,
    duration_ms: int,
    long_pause_threshold_ms: int,
) -> str:
    if duration_ms > long_pause_threshold_ms:
        return "pause_filler"
    if prev_speaker == "interviewee" and next_speaker == "interviewer":
        return "pause_answered"
    if prev_speaker == "interviewer":
        # interviewer -> interviewee  OR  interviewer -> interviewer
        return "pause_question"
    # interviewee -> interviewee
    return "pause_narrative"


def merge_spans(spans: list[dict]) -> list[dict]:
    """Merge consecutive spans that share the same label."""
    if not spans:
        return []
    merged = [dict(spans[0])]
    for span in spans[1:]:
        if span["label"] == merged[-1]["label"] and span["start_ms"] == merged[-1]["end_ms"]:
            merged[-1]["end_ms"] = span["end_ms"]
        else:
            merged.append(dict(span))
    return merged


def generate_labels(
    transcript: list[dict],
    max_word_duration_ms: int = 1000,
    long_pause_threshold_ms: int = 30_000,
) -> list[dict]:
    """Return a list of non-overlapping, contiguous labeled spans for one transcript."""
    if not transcript:
        return []

    # Flatten all words across segments, keeping speaker tag
    words = []
    for seg in transcript:
        speaker = seg["speaker"]
        for w in seg.get("words", []):
            words.append({"ms": int(w["ms"]), "speaker": speaker})

    if not words:
        return []

    words.sort(key=lambda w: w["ms"])

    rec_start = int(transcript[0]["start_ms"])
    rec_end = int(transcript[-1]["end_ms"])
    buffer_ms = max_word_duration_ms // 2  # applied when gap exceeds threshold

    spans = []

    # Leading filler: silence before the first word
    if words[0]["ms"] > rec_start:
        spans.append({"start_ms": rec_start, "end_ms": words[0]["ms"], "label": "pause_filler"})

    for i, word in enumerate(words):
        is_last = i == len(words) - 1
        next_word = None if is_last else words[i + 1]

        # Determine end of this word's speaking region
        if is_last:
            word_end = min(word["ms"] + max_word_duration_ms, rec_end)
        else:
            gap = next_word["ms"] - word["ms"]
            if gap <= max_word_duration_ms:
                # Short gap: word fills all the way to the next word (no pause)
                word_end = next_word["ms"]
            else:
                # Long gap: word gets a half-duration buffer, then a pause begins
                word_end = word["ms"] + buffer_ms

        spans.append({
            "start_ms": word["ms"],
            "end_ms": word_end,
            "label": f"{word['speaker']}_speaking",
        })

        # Pause region between this word's speaking end and next word's start
        if next_word is not None and word_end < next_word["ms"]:
            pause_duration = next_word["ms"] - word_end
            label = classify_pause(
                word["speaker"],
                next_word["speaker"],
                pause_duration,
                long_pause_threshold_ms,
            )
            spans.append({
                "start_ms": word_end,
                "end_ms": next_word["ms"],
                "label": label,
            })

    # Trailing filler: silence after the last word's speaking region
    last_end = spans[-1]["end_ms"] if spans else rec_start
    if last_end < rec_end:
        spans.append({"start_ms": last_end, "end_ms": rec_end, "label": "pause_filler"})

    return merge_spans(spans)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate speaking/pausing label spans from LLM-cleaned transcripts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--max-word-duration",
        type=int,
        default=1000,
        metavar="MS",
        help=(
            "Maximum speaking duration assigned to a single word (ms). "
            "Words whose gap to the next word exceeds this receive a buffer "
            "of max_word_duration/2 ms before the pause region begins."
        ),
    )
    parser.add_argument(
        "--long-pause-threshold",
        type=int,
        default=30_000,
        metavar="MS",
        help="Pauses longer than this (ms) are labelled pause_filler regardless of context.",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=TRANSCRIPTS_DIR,
        metavar="DIR",
        help="Directory containing input transcript JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        metavar="DIR",
        help="Directory to write output CSV files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N transcripts (useful for testing).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    transcript_files = sorted(args.transcripts_dir.glob("*.json"))
    if args.limit:
        transcript_files = transcript_files[: args.limit]

    total = len(transcript_files)
    processed = skipped = errors = 0

    print(f"Processing {total} transcript(s)  →  {args.output_dir}")
    print(f"  max_word_duration={args.max_word_duration} ms  "
          f"long_pause_threshold={args.long_pause_threshold} ms\n")

    for path in transcript_files:
        video_id = path.stem
        out_path = args.output_dir / f"{video_id}.csv"

        if out_path.exists():
            skipped += 1
            continue

        try:
            with open(path) as f:
                transcript = json.load(f)

            spans = generate_labels(
                transcript,
                max_word_duration_ms=args.max_word_duration,
                long_pause_threshold_ms=args.long_pause_threshold,
            )

            with open(out_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["start_ms", "end_ms", "label"])
                for s in spans:
                    w.writerow([s["start_ms"], s["end_ms"], s["label"]])

            processed += 1
            if processed % 500 == 0:
                print(f"  [{processed + skipped}/{total}]  processed={processed}  skipped={skipped}")

        except Exception as exc:
            print(f"  ERROR {video_id}: {exc}")
            errors += 1

    print(
        f"\nDone.  processed={processed}  skipped(already existed)={skipped}  "
        f"errors={errors}  total={total}"
    )


if __name__ == "__main__":
    main()
