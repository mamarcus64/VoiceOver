#!/usr/bin/env python3
"""Repackage the monolithic arousal_valence_dominance.json into per-video JSON files."""

import json
import os
import random
import sys
import time

INPUT_PATH = "/home/mjma/voices/test_data/vad_output/arousal_valence_dominance.json"
OUTPUT_DIR = "/home/mjma/voices/VoiceOver/data/audio_vad"


def repackage():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading {INPUT_PATH} ...")
    t0 = time.time()
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)
    elapsed = time.time() - t0
    print(f"Loaded in {elapsed:.1f}s — {len(data)} videos found.")

    written = 0
    for i, (video_id, segments_raw) in enumerate(data.items()):
        segments = []
        for row in segments_raw:
            segments.append({
                "start": row[0],
                "end": row[1],
                "valence": row[8],
                "arousal": row[9],
                "dominance": row[10],
            })

        out = {"video_id": video_id, "segments": segments}
        out_path = os.path.join(OUTPUT_DIR, f"{video_id}.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        written += 1

        if (i + 1) % 500 == 0 or (i + 1) == len(data):
            print(f"  Written {i + 1}/{len(data)} files ...")

    print(f"\nDone. {written} video files created in {OUTPUT_DIR}")
    return data, written


def validate(data, written):
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    # 1. Total files
    output_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".json")]
    print(f"\n1) Total output files: {len(output_files)}")
    print(f"   Expected (videos in source): {len(data)}")
    assert len(output_files) == len(data), "Mismatch in file count!"

    # 2. Spot-check 3 random videos
    sample_ids = random.sample(list(data.keys()), min(3, len(data)))
    print(f"\n2) Spot-checking videos: {sample_ids}")
    for vid in sample_ids:
        print(f"\n   --- Video {vid} ---")
        source_rows = data[vid][:2]
        with open(os.path.join(OUTPUT_DIR, f"{vid}.json"), "r") as f:
            output_data = json.load(f)
        output_segs = output_data["segments"][:2]

        for j, (src, out) in enumerate(zip(source_rows, output_segs)):
            print(f"   Segment {j} source: start={src[0]}, end={src[1]}, "
                  f"v={src[8]}, a={src[9]}, d={src[10]}")
            print(f"   Segment {j} output: start={out['start']}, end={out['end']}, "
                  f"v={out['valence']}, a={out['arousal']}, d={out['dominance']}")
            assert src[0] == out["start"]
            assert src[1] == out["end"]
            assert src[8] == out["valence"]
            assert src[9] == out["arousal"]
            assert src[10] == out["dominance"]
        print("   ✓ Matches.")

    # 3 & 4 & 5. Range and ordering checks
    print("\n3-5) Checking value ranges and segment ordering ...")
    bad_range_videos = []
    bad_order_videos = []
    total_segments = 0

    for video_id, segments_raw in data.items():
        for row in segments_raw:
            total_segments += 1
            start, end = row[0], row[1]
            v, a, d = row[8], row[9], row[10]

            if not (0 <= v <= 1 and 0 <= a <= 1 and 0 <= d <= 1):
                bad_range_videos.append((video_id, row))
            if start >= end:
                bad_order_videos.append((video_id, start, end))

    print(f"   Total segments checked: {total_segments}")

    if bad_range_videos:
        print(f"   ✗ {len(bad_range_videos)} segments with out-of-range VAD values:")
        for vid, row in bad_range_videos[:10]:
            print(f"     Video {vid}: v={row[8]}, a={row[9]}, d={row[10]}")
    else:
        print("   ✓ All valence/arousal/dominance values in [0, 1].")

    if bad_order_videos:
        print(f"   ✗ {len(bad_order_videos)} segments with start >= end:")
        for vid, s, e in bad_order_videos[:10]:
            print(f"     Video {vid}: start={s}, end={e}")
    else:
        print("   ✓ All segments have start < end.")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    random.seed(42)
    data, written = repackage()
    validate(data, written)
