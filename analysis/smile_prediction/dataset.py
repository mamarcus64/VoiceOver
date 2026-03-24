"""
Data loading, soft-label computation, and feature extraction for smile prediction.

Loads smile annotations, computes soft labels with not-a-smile discounting,
extracts and aligns audio/eyegaze VAD signals around each smile event.
"""

import json
import csv
import os
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

SMILE_CLASSES = ["genuine", "polite", "masking"]
CLASS_TO_IDX = {c: i for i, c in enumerate(SMILE_CLASSES)}


@dataclass
class SmileTask:
    task_number: int
    video_id: str
    smile_start: float
    smile_end: float
    soft_label: np.ndarray          # shape (3,) over SMILE_CLASSES, sums to 1
    weight: float                   # 1 - (not_a_smile fraction); 0 means skip
    annotator_count: int
    raw_labels: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Manifest + annotation loading
# ---------------------------------------------------------------------------

def load_manifest(path: Optional[Path] = None) -> dict:
    path = path or DATA_DIR / "smile_task_manifest.json"
    with open(path) as f:
        return json.load(f)


def load_annotations(ann_dir: Optional[Path] = None) -> dict[str, dict]:
    """Return {annotator_name: {task_number_str: record}}."""
    ann_dir = ann_dir or DATA_DIR / "smile_annotations"
    annotators = {}
    for fn in os.listdir(ann_dir):
        if not fn.endswith(".json"):
            continue
        with open(ann_dir / fn) as f:
            data = json.load(f)
        annotators[data["annotator"]] = data["annotations"]
    return annotators


def build_tasks(
    manifest: Optional[dict] = None,
    annotations: Optional[dict[str, dict]] = None,
    min_annotators: int = 1,
) -> list[SmileTask]:
    """
    Merge manifest and annotations into SmileTask objects with soft labels.

    Tasks where weight == 0 (all annotators said not_a_smile) are excluded.
    """
    manifest = manifest or load_manifest()
    annotations = annotations or load_annotations()

    task_by_num = {t["task_number"]: t for t in manifest["tasks"]}

    # Collect per-task labels from all annotators
    task_labels: dict[int, list[str]] = {}
    for _name, anns in annotations.items():
        for k, v in anns.items():
            tn = int(k)
            task_labels.setdefault(tn, []).append(v["label"])

    tasks = []
    for tn, labels in sorted(task_labels.items()):
        if len(labels) < min_annotators:
            continue
        info = task_by_num.get(tn)
        if info is None:
            continue

        n = len(labels)
        not_smile_count = sum(1 for l in labels if l == "not_a_smile")
        smile_labels = [l for l in labels if l != "not_a_smile"]
        weight = len(smile_labels) / n

        if weight == 0:
            continue

        soft = np.zeros(len(SMILE_CLASSES), dtype=np.float32)
        for l in smile_labels:
            if l in CLASS_TO_IDX:
                soft[CLASS_TO_IDX[l]] += 1
        soft /= soft.sum()

        tasks.append(SmileTask(
            task_number=tn,
            video_id=info["video_id"],
            smile_start=info["smile_start"],
            smile_end=info["smile_end"],
            soft_label=soft,
            weight=weight,
            annotator_count=n,
            raw_labels=labels,
        ))

    return tasks


# ---------------------------------------------------------------------------
# VAD signal loading
# ---------------------------------------------------------------------------

def load_audio_vad(video_id: str, data_dir: Optional[Path] = None) -> Optional[dict]:
    """Return {"segments": [{"start","end","valence","arousal","dominance"}, ...]}."""
    data_dir = data_dir or DATA_DIR / "audio_vad"
    path = data_dir / f"{video_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_eyegaze_vad(video_id: str, data_dir: Optional[Path] = None) -> Optional[list[dict]]:
    """Return list of {"timestamp","valence","arousal","dominance"} dicts."""
    data_dir = data_dir or DATA_DIR / "eyegaze_vad"
    path = data_dir / f"{video_id}.csv"
    if not path.exists():
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _audio_to_grid(segments: list[dict], grid: np.ndarray, grid_step: float) -> np.ndarray:
    """
    Map variable-length audio VAD segments onto a regular time grid.
    Returns (T, 4): [valence, arousal, dominance, present_flag].
    """
    T = len(grid)
    out = np.zeros((T, 4), dtype=np.float32)
    if not segments:
        return out

    for seg in segments:
        s, e = seg["start"], seg["end"]
        v = np.array([seg["valence"], seg["arousal"], seg["dominance"]], dtype=np.float32)
        for i, t in enumerate(grid):
            bin_start = t - grid_step / 2
            bin_end = t + grid_step / 2
            overlap_start = max(s, bin_start)
            overlap_end = min(e, bin_end)
            if overlap_end > overlap_start:
                frac = (overlap_end - overlap_start) / grid_step
                out[i, :3] += v * frac
                out[i, 3] += frac

    # Normalize where we have coverage
    mask = out[:, 3] > 0
    out[mask, :3] /= out[mask, 3:4]
    out[:, 3] = mask.astype(np.float32)
    return out


def _eyegaze_to_grid(rows: list[dict], grid: np.ndarray, grid_step: float) -> np.ndarray:
    """
    Snap eyegaze VAD samples to nearest grid point.
    Returns (T, 4): [valence, arousal, dominance, present_flag].
    """
    T = len(grid)
    out = np.zeros((T, 4), dtype=np.float32)
    counts = np.zeros(T, dtype=np.float32)
    if not rows:
        return out

    for r in rows:
        ts = float(r["timestamp"])
        v = np.array([float(r["valence"]), float(r["arousal"]), float(r["dominance"])],
                      dtype=np.float32)
        idx = np.argmin(np.abs(grid - ts))
        if abs(grid[idx] - ts) <= grid_step:
            out[idx, :3] += v
            counts[idx] += 1

    mask = counts > 0
    out[mask, :3] /= counts[mask, None]
    out[:, 3] = mask.astype(np.float32)
    return out


def extract_features(
    task: SmileTask,
    window_before: float = 30.0,
    window_after: float = 30.0,
    grid_step: float = 3.0,
) -> Optional[dict]:
    """
    Extract time-aligned feature sequence for a single smile task.

    Returns dict with:
        "sequence": np.ndarray (T, 11)
            [audio_V, audio_A, audio_D, audio_present,
             eyegaze_V, eyegaze_A, eyegaze_D, eyegaze_present,
             phase_before, phase_during, phase_after]
        "soft_label": np.ndarray (3,)
        "weight": float
        "task_number": int
    """
    audio_data = load_audio_vad(task.video_id)
    eyegaze_data = load_eyegaze_vad(task.video_id)

    t_start = task.smile_start - window_before
    t_end = task.smile_end + window_after
    t_start = max(t_start, 0.0)

    grid = np.arange(t_start, t_end, grid_step, dtype=np.float32)
    if len(grid) == 0:
        return None

    # Audio features (T, 4)
    if audio_data and audio_data.get("segments"):
        audio_feats = _audio_to_grid(audio_data["segments"], grid, grid_step)
    else:
        audio_feats = np.zeros((len(grid), 4), dtype=np.float32)

    # Eyegaze features (T, 4)
    if eyegaze_data:
        eyegaze_feats = _eyegaze_to_grid(eyegaze_data, grid, grid_step)
    else:
        eyegaze_feats = np.zeros((len(grid), 4), dtype=np.float32)

    # Phase indicators (T, 3): before / during / after
    phase = np.zeros((len(grid), 3), dtype=np.float32)
    phase[:, 0] = (grid < task.smile_start).astype(np.float32)
    phase[:, 1] = ((grid >= task.smile_start) & (grid <= task.smile_end)).astype(np.float32)
    phase[:, 2] = (grid > task.smile_end).astype(np.float32)

    sequence = np.concatenate([audio_feats, eyegaze_feats, phase], axis=1)

    return {
        "sequence": sequence,
        "soft_label": task.soft_label,
        "weight": task.weight,
        "task_number": task.task_number,
    }


def extract_aggregated_features(
    task: SmileTask,
    window_before: float = 30.0,
    window_after: float = 30.0,
    grid_step: float = 3.0,
) -> Optional[dict]:
    """
    Compute summary statistics per phase for the aggregated (Tier 1) baseline.

    Returns dict with:
        "features": np.ndarray (36,)
            For each phase (before/during/after):
                For each modality (audio/eyegaze):
                    mean_V, mean_A, mean_D, std_V, std_A, std_D
        "soft_label": np.ndarray (3,)
        "weight": float
        "task_number": int
    """
    raw = extract_features(task, window_before, window_after, grid_step)
    if raw is None:
        return None

    seq = raw["sequence"]  # (T, 11)
    audio_vad = seq[:, :3]
    audio_present = seq[:, 3]
    eyegaze_vad = seq[:, 4:7]
    eyegaze_present = seq[:, 7]
    phase_before = seq[:, 8].astype(bool)
    phase_during = seq[:, 9].astype(bool)
    phase_after = seq[:, 10].astype(bool)

    parts = []
    for phase_mask in [phase_before, phase_during, phase_after]:
        for vad, present in [(audio_vad, audio_present), (eyegaze_vad, eyegaze_present)]:
            mask = phase_mask & (present > 0.5)
            if mask.sum() > 0:
                vals = vad[mask]
                parts.extend([vals.mean(axis=0), vals.std(axis=0)])
            else:
                parts.extend([np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)])

    features = np.concatenate(parts)  # 3 phases * 2 modalities * 2 stats * 3 dims = 36

    return {
        "features": features,
        "soft_label": raw["soft_label"],
        "weight": raw["weight"],
        "task_number": raw["task_number"],
    }
