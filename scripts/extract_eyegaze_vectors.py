#!/usr/bin/env python3
"""
Copy binocular gaze vector columns from OpenFace result.csv into VoiceOver data.

Source layout (same as extract_smiling_segments.py):
  OPENFACE_DIR/<video_id>/result.csv

Output:
  <VoiceOver>/data/eyegaze_vectors/<video_id>.csv

Columns written: frame, timestamp, gaze_0_{x,y,z}, gaze_1_{x,y,z}, and
gaze_angle_x, gaze_angle_y when present (OpenFace head-in-camera angles).

Resumable: skips videos whose output exists and is at least as new as the source
(use --force to re-extract all). Writes via a temp file then rename so partial
runs do not leave corrupt CSVs.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

OPENFACE_DIR = Path(
    os.environ.get(
        "OPENFACE_DIR",
        str(PROJECT_DIR.parent / "threadward_results"),
    )
)
OUTPUT_DIR = PROJECT_DIR / "data" / "eyegaze_vectors"

BASE_COLS = [
    "frame",
    "timestamp",
    "gaze_0_x",
    "gaze_0_y",
    "gaze_0_z",
    "gaze_1_x",
    "gaze_1_y",
    "gaze_1_z",
]
ANGLE_COLS = ["gaze_angle_x", "gaze_angle_y"]


def needs_extract(video_id: str, force: bool) -> bool:
    """True if we should (re)build eyegaze_vectors for this id."""
    if force:
        return True
    src = OPENFACE_DIR / video_id / "result.csv"
    dst = OUTPUT_DIR / f"{video_id}.csv"
    if not dst.is_file():
        return True
    try:
        if dst.stat().st_size < 64:
            return True
    except OSError:
        return True
    try:
        return src.stat().st_mtime > dst.stat().st_mtime
    except OSError:
        return True


def process_video(video_id: str) -> tuple[str, int | None, str | None, str]:
    """
    Read one result.csv, write slim eyegaze_vectors CSV.
    Returns (id, n_rows, err, status) where status is 'ok' or 'error'.
    """
    src = OPENFACE_DIR / video_id / "result.csv"
    out_path = OUTPUT_DIR / f"{video_id}.csv"
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    try:
        with open(src, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return video_id, None, "no header", "error"

            missing = [c for c in BASE_COLS if c not in reader.fieldnames]
            if missing:
                return video_id, None, f"missing columns: {missing}", "error"

            out_fields = list(BASE_COLS)
            for c in ANGLE_COLS:
                if c in reader.fieldnames:
                    out_fields.append(c)

            rows = 0
            try:
                with open(tmp_path, "w", newline="", encoding="utf-8") as out_f:
                    writer = csv.DictWriter(
                        out_f, fieldnames=out_fields, extrasaction="ignore"
                    )
                    writer.writeheader()
                    for row in reader:
                        writer.writerow({k: row.get(k, "") for k in out_fields})
                        rows += 1
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise

        if rows == 0:
            tmp_path.unlink(missing_ok=True)
            return video_id, None, "empty csv", "error"

        tmp_path.replace(out_path)
        return video_id, rows, None, "ok"

    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        return video_id, None, str(e), "error"


def _imap_with_progress(
    pool: Pool,
    func,
    tasks: list[str],
    workers: int,
    desc: str,
):
    """Iterate imap_unordered with tqdm or periodic stderr lines."""
    n = len(tasks)
    chunksize = max(1, min(32, n // max(workers * 8, 1)))
    it = pool.imap_unordered(func, tasks, chunksize=chunksize)

    if tqdm is not None:
        yield from tqdm(it, total=n, desc=desc, unit="vid", file=sys.stderr)
        return

    t0 = time.monotonic()
    last_report = t0
    interval = 2.0
    step = max(1, n // 80) if n >= 80 else 1

    for i, item in enumerate(it, 1):
        now = time.monotonic()
        if i == n or i % step == 0 or (now - last_report) >= interval:
            elapsed = now - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (n - i) / rate if rate > 0 else 0.0
            pct = 100.0 * i / n if n else 100.0
            print(
                f"  {desc}  {i}/{n}  {pct:5.1f}%  {rate:5.1f}/s  ETA {eta:5.0f}s",
                file=sys.stderr,
                flush=True,
            )
            last_report = now
        yield item


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract gaze vectors from OpenFace CSVs.")
    parser.add_argument("--workers", type=int, default=min(32, cpu_count()))
    parser.add_argument(
        "--limit", type=int, default=None, help="Consider only first N ids (sorted)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even when output is up to date",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm / progress lines (quiet worker pool only)",
    )
    args = parser.parse_args()

    if not OPENFACE_DIR.is_dir():
        print(f"ERROR: OPENFACE_DIR not found: {OPENFACE_DIR}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video_ids = sorted(
        d
        for d in os.listdir(OPENFACE_DIR)
        if (OPENFACE_DIR / d / "result.csv").is_file()
    )

    if args.limit is not None:
        video_ids = video_ids[: args.limit]

    skipped = [v for v in video_ids if not needs_extract(v, args.force)]
    todo = [v for v in video_ids if needs_extract(v, args.force)]

    print(f"Source: {OPENFACE_DIR}", file=sys.stderr)
    print(f"Output: {OUTPUT_DIR}", file=sys.stderr)
    print(
        f"Videos with result.csv: {len(video_ids)}  |  "
        f"skip (up to date): {len(skipped)}  |  to process: {len(todo)}",
        file=sys.stderr,
    )

    if not todo:
        print("Nothing to do (all outputs current). Use --force to rebuild.", file=sys.stderr)
        return

    t0 = time.time()
    errors = 0
    total_rows = 0
    first_errors: list[tuple[str, str]] = []

    with Pool(args.workers) as pool:
        if args.no_progress:
            it = pool.imap_unordered(
                process_video,
                todo,
                chunksize=max(1, min(32, len(todo) // max(args.workers * 8, 1))),
            )
            results = list(it)
        else:
            results = list(
                _imap_with_progress(
                    pool,
                    process_video,
                    todo,
                    args.workers,
                    desc="eyegaze_vectors",
                )
            )

        for vid, n_rows, err, status in results:
            if status != "ok" or n_rows is None:
                errors += 1
                if len(first_errors) < 8:
                    first_errors.append((vid, err or "?"))
            else:
                total_rows += n_rows

    elapsed = time.time() - t0
    ok = len(todo) - errors

    print(file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print("EYEGAZE VECTORS EXTRACTION SUMMARY", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Skipped (resume):  {len(skipped)}", file=sys.stderr)
    print(f"  Attempted:         {len(todo)}", file=sys.stderr)
    print(f"  Videos OK:         {ok}", file=sys.stderr)
    print(f"  Errors:            {errors}", file=sys.stderr)
    print(f"  Rows (this run):   {total_rows}", file=sys.stderr)
    print(f"  Time (this run):   {elapsed:.1f}s", file=sys.stderr)
    print(f"  Output:            {OUTPUT_DIR}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    for vid, msg in first_errors:
        print(f"  ERROR {vid}: {msg}", file=sys.stderr)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
