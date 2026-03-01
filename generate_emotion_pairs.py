"""
Generate emotion comparison pairs by finding emotional outliers in VAD data.

For each subject, finds the segments with the most extreme valence and arousal
values, then pairs the highest with the lowest to create comparison stimuli.

Usage:
    python generate_emotion_pairs.py [--vad-json PATH] [--video-dir PATH] [--output PATH]
                                     [--min-segment-duration 2.0] [--clip-duration 10.0]
                                     [--min-diff-percentile 75]
"""
import json
import argparse
import os
import random
import csv
import numpy as np
from pathlib import Path


def load_vad_data(vad_json_path):
    """Load VAD JSON: {subject_id: [[start, end, ...9 values...], ...]}"""
    with open(vad_json_path) as f:
        return json.load(f)


def find_outlier_pairs(vad_data, video_dir, min_segment_duration=2.0,
                       clip_duration=10.0, min_diff_percentile=75, seed=42):
    """
    For each subject, find emotional outlier pairs.
    
    Each segment is an 11-element list:
      [0] start_time, [1] end_time, [2-7] per-frame VAD, 
      [8] avg_valence, [9] avg_arousal, [10] avg_dominance
    
    Returns list of dicts with pair information.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # First pass: collect all valence/arousal differences per subject
    subject_diffs = {}  # subject_id -> {valence_diff, arousal_diff}
    
    for subject_id, segments in vad_data.items():
        # Check video file exists
        video_file = f"{subject_id}.mp4"
        video_path = os.path.join(video_dir, video_file)
        if not os.path.exists(video_path):
            continue
        
        # Filter segments by minimum duration
        valid_segments = [
            s for s in segments
            if (s[1] - s[0]) >= min_segment_duration
        ]
        
        if len(valid_segments) < 2:
            continue
        
        valences = [s[8] for s in valid_segments]
        arousals = [s[9] for s in valid_segments]
        
        subject_diffs[subject_id] = {
            'valence_diff': max(valences) - min(valences),
            'arousal_diff': max(arousals) - min(arousals),
            'segments': valid_segments,
        }
    
    # Compute percentile thresholds for filtering
    all_valence_diffs = [d['valence_diff'] for d in subject_diffs.values()]
    all_arousal_diffs = [d['arousal_diff'] for d in subject_diffs.values()]
    
    valence_threshold = np.percentile(all_valence_diffs, min_diff_percentile)
    arousal_threshold = np.percentile(all_arousal_diffs, min_diff_percentile)
    
    print(f"Valence diff threshold (p{min_diff_percentile}): {valence_threshold:.4f}")
    print(f"Arousal diff threshold (p{min_diff_percentile}): {arousal_threshold:.4f}")
    print(f"Subjects with valid segments: {len(subject_diffs)}")
    
    # Second pass: create pairs
    pairs = []
    pair_id = 0
    
    for subject_id, info in subject_diffs.items():
        segments = info['segments']
        video_file = f"{subject_id}.mp4"
        
        # Valence comparison: max vs min valence
        if info['valence_diff'] >= valence_threshold:
            valences = [s[8] for s in segments]
            max_v_idx = int(np.argmax(valences))
            min_v_idx = int(np.argmin(valences))
            
            if max_v_idx != min_v_idx:
                seg_high = segments[max_v_idx]
                seg_low = segments[min_v_idx]
                
                # Randomize which is A and which is B
                if random.random() < 0.5:
                    seg_a, seg_b = seg_high, seg_low
                    high_label = "A"
                else:
                    seg_a, seg_b = seg_low, seg_high
                    high_label = "B"
                
                clip_a_start, clip_a_end = get_clip_times(seg_a, clip_duration)
                clip_b_start, clip_b_end = get_clip_times(seg_b, clip_duration)
                
                pairs.append({
                    'stimulus_id': f"{pair_id:05d}",
                    'subject_id': subject_id,
                    'video_file': video_file,
                    'comparison_type': 'valence',
                    'clip_a_start': round(clip_a_start, 3),
                    'clip_a_end': round(clip_a_end, 3),
                    'clip_b_start': round(clip_b_start, 3),
                    'clip_b_end': round(clip_b_end, 3),
                    'clip_a_valence': round(seg_a[8], 4),
                    'clip_a_arousal': round(seg_a[9], 4),
                    'clip_b_valence': round(seg_b[8], 4),
                    'clip_b_arousal': round(seg_b[9], 4),
                    'high_valence_clip': high_label,
                    'valence_diff': round(info['valence_diff'], 4),
                })
                pair_id += 1
        
        # Arousal comparison: max vs min arousal
        if info['arousal_diff'] >= arousal_threshold:
            arousals = [s[9] for s in segments]
            max_a_idx = int(np.argmax(arousals))
            min_a_idx = int(np.argmin(arousals))
            
            if max_a_idx != min_a_idx:
                seg_high = segments[max_a_idx]
                seg_low = segments[min_a_idx]
                
                # Randomize which is A and which is B
                if random.random() < 0.5:
                    seg_a, seg_b = seg_high, seg_low
                    high_label = "A"
                else:
                    seg_a, seg_b = seg_low, seg_high
                    high_label = "B"
                
                clip_a_start, clip_a_end = get_clip_times(seg_a, clip_duration)
                clip_b_start, clip_b_end = get_clip_times(seg_b, clip_duration)
                
                pairs.append({
                    'stimulus_id': f"{pair_id:05d}",
                    'subject_id': subject_id,
                    'video_file': video_file,
                    'comparison_type': 'arousal',
                    'clip_a_start': round(clip_a_start, 3),
                    'clip_a_end': round(clip_a_end, 3),
                    'clip_b_start': round(clip_b_start, 3),
                    'clip_b_end': round(clip_b_end, 3),
                    'clip_a_valence': round(seg_a[8], 4),
                    'clip_a_arousal': round(seg_a[9], 4),
                    'clip_b_valence': round(seg_b[8], 4),
                    'clip_b_arousal': round(seg_b[9], 4),
                    'high_arousal_clip': high_label,
                    'arousal_diff': round(info['arousal_diff'], 4),
                })
                pair_id += 1
    
    # Shuffle all pairs
    random.shuffle(pairs)
    
    # Re-assign stimulus IDs after shuffle
    for i, pair in enumerate(pairs):
        pair['stimulus_id'] = f"{i:05d}"
    
    return pairs


def get_clip_times(segment, clip_duration):
    """
    Get clip start/end times centered on the segment midpoint.
    Clips are `clip_duration` seconds long.
    """
    seg_start = segment[0]
    seg_end = segment[1]
    midpoint = (seg_start + seg_end) / 2.0
    
    clip_start = midpoint - clip_duration / 2.0
    clip_end = midpoint + clip_duration / 2.0
    
    # Clamp start to >= 0
    if clip_start < 0:
        clip_end += (-clip_start)
        clip_start = 0.0
    
    return clip_start, clip_end


def save_pairs(pairs, output_path):
    """Save pairs to CSV."""
    if not pairs:
        print("No pairs to save!")
        return
    
    fieldnames = [
        'stimulus_id', 'subject_id', 'video_file', 'comparison_type',
        'clip_a_start', 'clip_a_end', 'clip_b_start', 'clip_b_end',
        'clip_a_valence', 'clip_a_arousal', 'clip_b_valence', 'clip_b_arousal',
        'high_valence_clip', 'valence_diff', 'high_arousal_clip', 'arousal_diff',
    ]
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(pairs)
    
    print(f"Saved {len(pairs)} pairs to {output_path}")
    
    # Stats
    valence_pairs = [p for p in pairs if p['comparison_type'] == 'valence']
    arousal_pairs = [p for p in pairs if p['comparison_type'] == 'arousal']
    print(f"  Valence comparison pairs: {len(valence_pairs)}")
    print(f"  Arousal comparison pairs: {len(arousal_pairs)}")
    unique_subjects = len(set(p['subject_id'] for p in pairs))
    print(f"  Unique subjects: {unique_subjects}")


def main():
    parser = argparse.ArgumentParser(description="Generate emotion comparison pairs from VAD data")
    parser.add_argument('--vad-json', type=str,
                        default='/home/mjma/voices/test_data/vad_output/arousal_valence_dominance.json',
                        help='Path to VAD JSON file')
    parser.add_argument('--video-dir', type=str,
                        default='/data2/mjma/voices/test_data/videos/',
                        help='Path to video directory')
    parser.add_argument('--output', type=str,
                        default='/home/mjma/voices/VoiceOver/emotion_pairs.csv',
                        help='Output CSV path')
    parser.add_argument('--min-segment-duration', type=float, default=2.0,
                        help='Minimum segment duration in seconds')
    parser.add_argument('--clip-duration', type=float, default=10.0,
                        help='Duration of each clip in seconds')
    parser.add_argument('--min-diff-percentile', type=int, default=75,
                        help='Minimum percentile for V/A difference to include a pair')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    print(f"Loading VAD data from {args.vad_json}...")
    vad_data = load_vad_data(args.vad_json)
    print(f"Loaded {len(vad_data)} subjects")
    
    print(f"Finding outlier pairs (min_diff_percentile={args.min_diff_percentile})...")
    pairs = find_outlier_pairs(
        vad_data,
        args.video_dir,
        min_segment_duration=args.min_segment_duration,
        clip_duration=args.clip_duration,
        min_diff_percentile=args.min_diff_percentile,
        seed=args.seed,
    )
    
    save_pairs(pairs, args.output)


if __name__ == '__main__':
    main()

