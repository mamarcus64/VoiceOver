#!/usr/bin/env python3
"""
Build a per-task AU feature dataset from OpenFace CSVs + human smile annotations.

Processes only the unique videos referenced by annotated tasks (~130 videos),
not the full corpus (~3k), making it ~20x faster.

AU12_r_mean >= 1.5 is the manifest's intensityThreshold; all tasks already
satisfy it. Asserted explicitly so that any new AU combinations evaluated on
this dataset are proper subsets of the original AU12-filtered pool.

Outputs
-------
analysis/au_features_dataset.csv
    One row per (task_number, annotator).
    Columns: task metadata, per-AU stats (mean/peak/std/mass for _r,
    mean for _c), fine label, binary label (1=smile, 0=not_a_smile),
    n_annotators, consensus_label, consensus_binary, consensus_agreement.

analysis/au_features_dataset_meta.json
    Summary statistics.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = Path(os.environ.get("VOICEOVER_DATA_DIR", str(PROJECT_DIR / "data")))
OPENFACE_DIR = Path(os.environ.get("OPENFACE_DIR", str(PROJECT_DIR.parent / "threadward_results")))
OUT_DIR = PROJECT_DIR / "analysis"

ANNOTATIONS_DIR = DATA_DIR / "smile_annotations"
MANIFEST_PATH = DATA_DIR / "smile_task_manifest.json"

FPS = 30.0
AU12_MIN = 1.5  # bare minimum; manifest already enforces this

VALID_LABELS = {"genuine", "polite", "masking", "not_a_smile"}
# binary: 1 = any smile category, 0 = not_a_smile
SMILE_LABELS = {"genuine", "polite", "masking"}


def _effective_label(entry: dict) -> str | None:
    """Mirror of smile_agreement.py _effective_label."""
    label = entry.get("label")
    if label == "not_a_smile":
        return "not_a_smile"
    if entry.get("not_a_smile"):
        return "not_a_smile"
    if label in VALID_LABELS:
        return label
    return None


def _ensure_manifest() -> None:
    if MANIFEST_PATH.is_file():
        return
    print("Manifest not found; generating with default params...")
    sys.path.insert(0, str(SCRIPT_DIR))
    from generate_task_manifest import generate_and_write
    generate_and_write()


def load_manifest() -> dict[str, dict]:
    _ensure_manifest()
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    assert manifest["params"]["intensityThreshold"] >= AU12_MIN, (
        f"Manifest intensityThreshold {manifest['params']['intensityThreshold']} < {AU12_MIN}"
    )
    return {str(t["task_number"]): t for t in manifest["tasks"]}


def load_all_annotations() -> dict[str, dict[str, str]]:
    """Returns {task_number_str: {annotator: effective_label}}."""
    by_task: dict[str, dict[str, str]] = defaultdict(dict)
    for path in sorted(ANNOTATIONS_DIR.glob("*.json")):
        annotator = path.stem
        with open(path) as f:
            data = json.load(f)
        for task_key, entry in data.get("annotations", {}).items():
            eff = _effective_label(entry)
            if eff is not None:
                by_task[task_key][annotator] = eff
    return dict(by_task)


def load_openface_csv(video_id: str) -> tuple[np.ndarray, list[str], np.ndarray, list[str], np.ndarray]:
    """Load OpenFace result.csv. Returns (timestamps, au_r_cols, data_r, au_c_cols, data_c)."""
    csv_path = OPENFACE_DIR / video_id / "result.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(str(csv_path))
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    timestamps = df["timestamp"].to_numpy(dtype=np.float64)
    au_r_cols = [c for c in df.columns if c.startswith("AU") and c.endswith("_r")]
    au_c_cols = [c for c in df.columns if c.startswith("AU") and c.endswith("_c")]
    data_r = df[au_r_cols].to_numpy(dtype=np.float64)
    data_c = df[au_c_cols].to_numpy(dtype=np.float64)
    return timestamps, au_r_cols, data_r, au_c_cols, data_c


def compute_au_stats(
    slice_r: np.ndarray,
    au_r_cols: list[str],
    slice_c: np.ndarray,
    au_c_cols: list[str],
) -> dict:
    row = {}
    for i, col in enumerate(au_r_cols):
        au = col[:-2]  # strip "_r"
        vals = slice_r[:, i]
        row[f"{au}_r_mean"] = float(np.mean(vals))
        row[f"{au}_r_peak"] = float(np.max(vals))
        row[f"{au}_r_std"] = float(np.std(vals))
        row[f"{au}_r_mass"] = float(np.sum(vals) / FPS)
    for i, col in enumerate(au_c_cols):
        au = col[:-2]  # strip "_c"
        row[f"{au}_c_mean"] = float(np.mean(slice_c[:, i]))
    return row


def majority_vote(labels: list[str]) -> tuple[str, float]:
    """Majority label and agreement fraction. Ties go to smile (not_a_smile loses ties)."""
    counts: dict[str, int] = defaultdict(int)
    for lab in labels:
        counts[lab] += 1
    # Binary aggregation: smile fraction
    n_smile = sum(v for k, v in counts.items() if k in SMILE_LABELS)
    n_total = len(labels)
    smile_fraction = n_smile / n_total
    consensus = "smile" if smile_fraction >= 0.5 else "not_a_smile"
    agreement = max(smile_fraction, 1 - smile_fraction)
    # For fine label, pick the single most common label
    fine_winner = max(counts, key=counts.__getitem__)
    return consensus, fine_winner, float(agreement)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading manifest...")
    task_lookup = load_manifest()
    print(f"  {len(task_lookup):,} total tasks in manifest")

    print("Loading annotations...")
    by_task = load_all_annotations()
    annotated_keys = sorted(by_task.keys(), key=int)
    print(f"  {len(annotated_keys)} annotated tasks, "
          f"{len(list(ANNOTATIONS_DIR.glob('*.json')))} annotators")

    # Map video → list of annotated task keys
    video_to_tasks: dict[str, list[str]] = defaultdict(list)
    missing_in_manifest: list[str] = []
    for tk in annotated_keys:
        info = task_lookup.get(tk)
        if info is None:
            missing_in_manifest.append(tk)
        else:
            video_to_tasks[info["video_id"]].append(tk)

    if missing_in_manifest:
        print(f"  WARNING: {len(missing_in_manifest)} annotated tasks not in manifest "
              f"(first 5: {missing_in_manifest[:5]})")

    unique_videos = sorted(video_to_tasks.keys())
    print(f"  {len(unique_videos)} unique videos to process "
          f"(from {len(annotated_keys)} annotated tasks)")

    rows: list[dict] = []
    skipped_no_csv = 0
    skipped_no_frames = 0
    skipped_threshold = 0

    for vi, video_id in enumerate(unique_videos, 1):
        task_keys = video_to_tasks[video_id]
        print(f"  [{vi:3d}/{len(unique_videos)}] {video_id} ({len(task_keys)} tasks)...",
              end=" ", flush=True)
        try:
            timestamps, au_r_cols, data_r, au_c_cols, data_c = load_openface_csv(video_id)
        except FileNotFoundError:
            print("CSV not found, skipped")
            skipped_no_csv += len(task_keys)
            continue

        n_ok = 0
        for tk in task_keys:
            info = task_lookup[tk]
            manifest_mean_r = info["mean_r"]

            # Explicit AU12 >= 1.5 guard (manifest already enforces, belt-and-suspenders)
            if manifest_mean_r < AU12_MIN:
                skipped_threshold += 1
                continue

            mask = (timestamps >= info["smile_start"]) & (timestamps <= info["smile_end"])
            if not np.any(mask):
                skipped_no_frames += 1
                continue

            au_stats = compute_au_stats(data_r[mask], au_r_cols, data_c[mask], au_c_cols)

            annotator_labels = by_task[tk]
            n_annotators = len(annotator_labels)
            consensus_binary_str, consensus_fine, consensus_agreement = majority_vote(
                list(annotator_labels.values())
            )
            consensus_binary = 1 if consensus_binary_str == "smile" else 0

            for annotator, fine_label in annotator_labels.items():
                label_binary = 1 if fine_label in SMILE_LABELS else 0
                rows.append({
                    "task_number": int(tk),
                    "video_id": video_id,
                    "smile_start": info["smile_start"],
                    "smile_end": info["smile_end"],
                    "manifest_AU12_mean_r": manifest_mean_r,
                    "manifest_AU12_peak_r": info["peak_r"],
                    "annotator": annotator,
                    "label_fine": fine_label,
                    "label_binary": label_binary,
                    "n_annotators": n_annotators,
                    "consensus_label_binary": consensus_binary_str,
                    "consensus_label_fine": consensus_fine,
                    "consensus_binary": consensus_binary,
                    "consensus_agreement": round(consensus_agreement, 3),
                    **au_stats,
                })
            n_ok += 1

        print(f"{n_ok} ok")

    df = pd.DataFrame(rows)

    # Sort for readability
    df = df.sort_values(["task_number", "annotator"]).reset_index(drop=True)

    out_path = OUT_DIR / "au_features_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df):,} rows → {out_path}")

    # Summary
    n_unique_tasks = int(df["task_number"].nunique())
    consensus_df = df.drop_duplicates("task_number")
    label_dist = consensus_df["consensus_label_binary"].value_counts().to_dict()
    fine_dist = consensus_df["consensus_label_fine"].value_counts().to_dict()
    ann_counts = df.groupby("annotator")["task_number"].count().to_dict()

    meta = {
        "total_rows": len(df),
        "unique_tasks": n_unique_tasks,
        "unique_videos": int(df["video_id"].nunique()),
        "au12_min_threshold": AU12_MIN,
        "manifest_params": task_lookup[list(task_lookup.keys())[0]],  # placeholder
        "skipped_no_csv": skipped_no_csv,
        "skipped_no_frames": skipped_no_frames,
        "skipped_threshold": skipped_threshold,
        "consensus_binary_dist": label_dist,
        "consensus_fine_dist": fine_dist,
        "annotator_task_counts": ann_counts,
        "au_r_columns": [c for c in df.columns if c.startswith("AU") and c.endswith("_r_mean")],
        "au_c_columns": [c for c in df.columns if c.startswith("AU") and c.endswith("_c_mean")],
    }
    meta_path = OUT_DIR / "au_features_dataset_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata → {meta_path}")
    print(json.dumps({k: v for k, v in meta.items() if k not in ("au_r_columns", "au_c_columns")},
                     indent=2))


if __name__ == "__main__":
    main()
