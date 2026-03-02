#!/usr/bin/env python3
"""Validate the integrity of all VoiceOver data files."""

import json
import os
import sys
from pathlib import Path

DATA_DIR = Path(os.environ.get(
    "VOICEOVER_DATA_DIR",
    Path(__file__).resolve().parent.parent / "data",
))

errors = []
warnings = []


def check(condition, msg):
    if not condition:
        errors.append(msg)
        print(f"  FAIL: {msg}")
    return condition


def warn(msg):
    warnings.append(msg)
    print(f"  WARN: {msg}")


def validate_manifest():
    print("Checking manifest.json...")
    path = DATA_DIR / "manifest.json"
    if not check(path.is_file(), "manifest.json not found"):
        return
    with open(path) as f:
        manifest = json.load(f)
    check(isinstance(manifest, list), "manifest should be a list")
    check(len(manifest) > 1000, f"manifest has only {len(manifest)} entries (expected 5000+)")

    ids = set()
    for entry in manifest:
        check("id" in entry, f"entry missing 'id': {entry}")
        check("youtube_url" in entry, f"entry missing 'youtube_url': {entry.get('id', '?')}")
        check("downloaded" in entry, f"entry missing 'downloaded': {entry.get('id', '?')}")
        if entry.get("id"):
            check(entry["id"] not in ids, f"duplicate id: {entry['id']}")
            ids.add(entry["id"])

    downloaded = sum(1 for e in manifest if e.get("downloaded"))
    print(f"  {len(manifest)} entries, {downloaded} downloaded")


def validate_transcripts():
    print("Checking transcripts...")
    transcript_dir = DATA_DIR / "transcripts"
    if not check(transcript_dir.is_dir(), "transcripts/ dir not found"):
        return
    files = list(transcript_dir.glob("*.json"))
    check(len(files) > 1000, f"only {len(files)} transcript files (expected 5000+)")

    sample = files[:5]
    for f in sample:
        with open(f) as fh:
            data = json.load(fh)
        check(isinstance(data, list), f"{f.name}: should be a list")
        if data:
            u = data[0]
            check("speaker" in u, f"{f.name}: missing 'speaker'")
            check("start_ms" in u, f"{f.name}: missing 'start_ms'")
            check("text" in u, f"{f.name}: missing 'text'")

    print(f"  {len(files)} transcript files, schema OK on sample")


def validate_audio_vad():
    print("Checking audio_vad/...")
    vad_dir = DATA_DIR / "audio_vad"
    if not check(vad_dir.is_dir(), "audio_vad/ dir not found"):
        return
    files = list(vad_dir.glob("*.json"))
    check(len(files) > 1000, f"only {len(files)} audio_vad files (expected 5000+)")

    sample = files[:5]
    for f in sample:
        with open(f) as fh:
            data = json.load(fh)
        check("segments" in data, f"{f.name}: missing 'segments'")
        if data.get("segments"):
            seg = data["segments"][0]
            for key in ("start", "end", "valence", "arousal", "dominance"):
                check(key in seg, f"{f.name}: segment missing '{key}'")

    print(f"  {len(files)} audio_vad files, schema OK on sample")


def validate_eyegaze_vad():
    print("Checking eyegaze_vad/...")
    vad_dir = DATA_DIR / "eyegaze_vad"
    if not vad_dir.is_dir():
        warn("eyegaze_vad/ not found (run GLASS batch script to generate)")
        return
    files = list(vad_dir.glob("*.csv"))
    if not files:
        warn("no eyegaze_vad CSV files found yet")
        return
    print(f"  {len(files)} eyegaze_vad files found")


def validate_annotations_dir():
    print("Checking annotations/...")
    ann_dir = DATA_DIR / "annotations"
    if not ann_dir.is_dir():
        warn("annotations/ dir not found (will be created on first annotation)")
        return
    print(f"  annotations/ exists")


def main():
    print(f"Data directory: {DATA_DIR}\n")

    validate_manifest()
    validate_transcripts()
    validate_audio_vad()
    validate_eyegaze_vad()
    validate_annotations_dir()

    print()
    if errors:
        print(f"RESULT: {len(errors)} errors, {len(warnings)} warnings")
        sys.exit(1)
    elif warnings:
        print(f"RESULT: OK with {len(warnings)} warnings")
    else:
        print("RESULT: All checks passed")


if __name__ == "__main__":
    main()
