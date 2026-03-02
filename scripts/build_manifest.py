#!/usr/bin/env python3
"""Build a video manifest JSON from youtube_links.csv and the videos directory."""

import csv
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TSV_PATH = REPO_ROOT / "test_data" / "youtube_links.csv"
VIDEOS_DIR = REPO_ROOT / "test_data" / "videos"
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data" / "manifest.json"


def build_manifest():
    video_files = {f.stem for f in VIDEOS_DIR.glob("*.mp4")}

    manifest = []
    with open(TSV_PATH, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            int_code = int(row["IntCode"])
            tape = int(row["TapeNumber"])
            vid_id = f"{int_code}.{tape}"
            manifest.append({
                "id": vid_id,
                "int_code": int_code,
                "tape": tape,
                "youtube_url": row["YouTubeLink"].strip(),
                "downloaded": vid_id in video_files,
            })

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return manifest, video_files


def validate(manifest, video_files):
    print(f"Total manifest entries: {len(manifest)}")

    downloaded = sum(1 for e in manifest if e["downloaded"])
    not_downloaded = len(manifest) - downloaded
    print(f"  Downloaded: {downloaded}")
    print(f"  Not downloaded: {not_downloaded}")

    null_urls = [e for e in manifest if e["youtube_url"] == "NULL"]
    bad_urls = [e for e in manifest
                if e["youtube_url"] != "NULL" and not e["youtube_url"].startswith("https://")]
    if null_urls:
        print(f"  NULL YouTube URLs (no link available): {len(null_urls)}")
    assert not bad_urls, f"Non-NULL URLs not starting with https://: {bad_urls[:5]}"
    print("URL check passed — all non-NULL URLs start with https://")

    seen, dupes = set(), set()
    for e in manifest:
        (dupes if e["id"] in seen else seen).add(e["id"])
    assert not dupes, f"Duplicate IDs found: {dupes}"
    print("ID uniqueness check passed — no duplicates")

    manifest_ids = {e["id"] for e in manifest}
    orphans = sorted(video_files - manifest_ids)
    if orphans:
        print(f"Orphan .mp4 files (no manifest entry): {len(orphans)}")
        for o in orphans[:20]:
            print(f"  {o}.mp4")
    else:
        print("No orphan .mp4 files found")

    print("\nFirst 3 entries:")
    for e in manifest[:3]:
        print(f"  {e}")
    print("Last 3 entries:")
    for e in manifest[-3:]:
        print(f"  {e}")


if __name__ == "__main__":
    manifest, video_files = build_manifest()
    print(f"Manifest written to {MANIFEST_PATH}\n")
    validate(manifest, video_files)
