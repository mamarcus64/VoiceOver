#!/usr/bin/env python3
"""Analyze alignment of audio VAD, eyegaze VAD, and LLM labels against human
smile-valence annotations.

For each modality, finds the thresholds (negative / neutral / positive) that
maximise accuracy on human-annotated tasks.  Then fits a simple logistic
regression combining all modalities with leave-one-out cross-validation.

Usage:
    python scripts/analyze_smile_valence_alignment.py
"""

import json
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
MANIFEST_PATH = DATA / "smile_valence_task_manifest.json"
ANN_DIR = DATA / "smile_valence_annotations"
AUDIO_DIR = DATA / "audio_vad"
EYEGAZE_DIR = DATA / "eyegaze_vad"

VALENCE_ORD = {"negative": -1, "neutral": 0, "positive": 1}
ORD_VALENCE = {-1: "negative", 0: "neutral", 1: "positive"}


# ── Data extraction ───────────────────────────────────────────────────────────

def audio_valence_for_smile(video_id: str, smile_start: float, smile_end: float) -> float | None:
    """Weighted-average audio valence across segments overlapping the smile window."""
    fp = AUDIO_DIR / f"{video_id}.json"
    if not fp.is_file():
        return None
    with open(fp) as f:
        segs = json.load(f)["segments"]
    total_weight = total_val = 0.0
    for s in segs:
        overlap = min(smile_end, s["end"]) - max(smile_start, s["start"])
        if overlap <= 0:
            continue
        total_weight += overlap
        total_val += overlap * s["valence"]
    return total_val / total_weight if total_weight > 0 else None


def eyegaze_valence_for_smile(video_id: str, smile_start: float, smile_end: float,
                               context_s: float = 5.0) -> float | None:
    """Mean eyegaze valence for rows whose timestamp falls near the smile window.

    Uses a context buffer because eyegaze rows are sparse (~2.5 s intervals) and
    smiles are often shorter than one interval.
    """
    fp = EYEGAZE_DIR / f"{video_id}.csv"
    if not fp.is_file():
        return None
    df = pd.read_csv(fp)
    window = df[
        (df["timestamp"] >= smile_start - context_s) &
        (df["timestamp"] <= smile_end + context_s)
    ]
    if window.empty:
        return None
    # Weight rows by proximity: rows inside the smile window get weight 1,
    # rows in the context buffer get weight proportional to their distance.
    weights = []
    for ts in window["timestamp"]:
        if smile_start <= ts <= smile_end:
            weights.append(1.0)
        else:
            dist = min(abs(ts - smile_start), abs(ts - smile_end))
            weights.append(max(0.0, 1.0 - dist / context_s))
    weights = np.array(weights)
    if weights.sum() == 0:
        return None
    return float(np.average(window["valence"].values, weights=weights))


# ── Threshold optimisation ────────────────────────────────────────────────────

def best_thresholds(values: list[float], labels: list[int]) -> tuple[float, float, float]:
    """Find t_lo, t_hi that maximise accuracy when mapping continuous value to
    -1 / 0 / 1 via:  v < t_lo → -1,  v < t_hi → 0,  else → 1.

    Returns (best_t_lo, best_t_hi, best_accuracy).
    """
    vals = np.array(values)
    labs = np.array(labels)
    # Search over all unique value pairs
    candidates = np.unique(vals)
    best_acc, best_t_lo, best_t_hi = -1.0, np.median(vals), np.median(vals)
    for i, t_lo in enumerate(candidates):
        for t_hi in candidates[i:]:
            pred = np.where(vals < t_lo, -1, np.where(vals < t_hi, 0, 1))
            acc = (pred == labs).mean()
            if acc > best_acc:
                best_acc, best_t_lo, best_t_hi = acc, t_lo, t_hi
    return best_t_lo, best_t_hi, best_acc


def apply_thresholds(value: float, t_lo: float, t_hi: float) -> int:
    if value < t_lo:
        return -1
    if value < t_hi:
        return 0
    return 1


# ── Confusion helpers ─────────────────────────────────────────────────────────

def confusion_table(y_true: list[int], y_pred: list[int]) -> str:
    classes = [-1, 0, 1]
    header = "         " + "  ".join(f"{ORD_VALENCE[c]:>8}" for c in classes)
    rows = [header]
    for t in classes:
        row = f"{ORD_VALENCE[t]:>8} "
        for p in classes:
            n = sum(yt == t and yp == p for yt, yp in zip(y_true, y_pred))
            row += f"  {n:>8}"
        rows.append(row)
    return "\n".join(rows)


def per_class_acc(y_true: list[int], y_pred: list[int]) -> str:
    classes = [-1, 0, 1]
    parts = []
    for c in classes:
        idx = [i for i, y in enumerate(y_true) if y == c]
        if not idx:
            continue
        acc = sum(y_pred[i] == c for i in idx) / len(idx)
        parts.append(f"{ORD_VALENCE[c]}={acc:.0%}(n={len(idx)})")
    return "  ".join(parts)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load manifest
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    task_map = {t["task_number"]: t for t in manifest["tasks"]}

    # Load all human annotations
    records = []
    if ANN_DIR.is_dir():
        for fp in sorted(ANN_DIR.glob("*.json")):
            if fp.suffix == ".json" and not fp.stem.endswith(".bak"):
                with open(fp) as f:
                    ann_data = json.load(f)
                annotator = ann_data["annotator"]
                for tn_str, entry in ann_data["annotations"].items():
                    tn = int(tn_str)
                    if tn not in task_map:
                        continue
                    # Skip not-a-smile
                    if entry.get("not_a_smile"):
                        continue
                    nv = entry.get("narrative_valence")
                    sv = entry.get("speaker_valence")
                    if nv not in VALENCE_ORD or sv not in VALENCE_ORD:
                        continue
                    task = task_map[tn]
                    records.append({
                        "task_number": tn,
                        "annotator": annotator,
                        "video_id": task["video_id"],
                        "smile_start": task["smile_start"],
                        "smile_end": task["smile_end"],
                        "human_nv": VALENCE_ORD[nv],
                        "human_sv": VALENCE_ORD[sv],
                        "llm_nv": VALENCE_ORD.get(task.get("llm_narrative_valence", ""), 0),
                        "llm_sv": VALENCE_ORD.get(task.get("llm_speaker_valence", ""), 0),
                    })

    if not records:
        print("No human annotations found. Exiting.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"SMILE VALENCE ALIGNMENT ANALYSIS")
    print(f"{'='*60}")
    print(f"Human annotations: {len(records)}  ({len({r['annotator'] for r in records})} annotator(s))")
    print(f"NOTE: n={len(records)} is small — results are exploratory\n")

    # Extract audio and eyegaze features
    print("Extracting audio and eyegaze features ...")
    for r in records:
        r["audio_v"] = audio_valence_for_smile(r["video_id"], r["smile_start"], r["smile_end"])
        r["eyegaze_v"] = eyegaze_valence_for_smile(r["video_id"], r["smile_start"], r["smile_end"])

    audio_ok = [r for r in records if r["audio_v"] is not None]
    eyegaze_ok = [r for r in records if r["eyegaze_v"] is not None]
    print(f"Audio coverage:   {len(audio_ok)}/{len(records)} tasks")
    print(f"Eyegaze coverage: {len(eyegaze_ok)}/{len(records)} tasks\n")

    for target_field, target_label in [("human_sv", "Speaker's Current Valence"),
                                        ("human_nv", "Narrative Valence")]:
        print(f"\n{'─'*60}")
        print(f"TARGET: {target_label}")
        print(f"{'─'*60}")

        y_human = [r[target_field] for r in records]
        from collections import Counter
        dist = Counter(ORD_VALENCE[y] for y in y_human)
        print(f"Human distribution: {dict(dist)}")

        # ── LLM baseline ────────────────────────────────────────────────────
        llm_field = "llm_sv" if target_field == "human_sv" else "llm_nv"
        y_llm = [r[llm_field] for r in records]
        llm_acc = accuracy_score(y_human, y_llm)
        print(f"\n[LLM]  accuracy={llm_acc:.1%}")
        print(f"       per-class: {per_class_acc(y_human, y_llm)}")
        print("       confusion (rows=human, cols=llm):")
        print(confusion_table(y_human, y_llm))

        # ── Audio ────────────────────────────────────────────────────────────
        if audio_ok:
            a_vals = [r["audio_v"] for r in audio_ok]
            a_human = [r[target_field] for r in audio_ok]
            t_lo, t_hi, best_acc = best_thresholds(a_vals, a_human)
            y_audio_pred = [apply_thresholds(v, t_lo, t_hi) for v in a_vals]
            print(f"\n[Audio VAD]  accuracy={best_acc:.1%}  thresholds: neg<{t_lo:.3f}≤neu<{t_hi:.3f}≤pos")
            print(f"            per-class: {per_class_acc(a_human, y_audio_pred)}")
            print("            confusion (rows=human, cols=audio):")
            print(confusion_table(a_human, y_audio_pred))
        else:
            print("\n[Audio VAD]  no data")
            t_lo = t_hi = None

        # ── Eyegaze ──────────────────────────────────────────────────────────
        if eyegaze_ok:
            e_vals = [r["eyegaze_v"] for r in eyegaze_ok]
            e_human = [r[target_field] for r in eyegaze_ok]
            et_lo, et_hi, best_acc_e = best_thresholds(e_vals, e_human)
            y_eye_pred = [apply_thresholds(v, et_lo, et_hi) for v in e_vals]
            print(f"\n[Eyegaze VAD]  accuracy={best_acc_e:.1%}  thresholds: neg<{et_lo:.3f}≤neu<{et_hi:.3f}≤pos")
            print(f"              per-class: {per_class_acc(e_human, y_eye_pred)}")
            print("              confusion (rows=human, cols=eyegaze):")
            print(confusion_table(e_human, y_eye_pred))
        else:
            print("\n[Eyegaze VAD]  no data")
            et_lo = et_hi = None

        # ── Combined: LLM + Audio + Eyegaze ──────────────────────────────────
        # Use only tasks that have ALL features
        combined = [r for r in records
                    if r["audio_v"] is not None and r["eyegaze_v"] is not None]
        print(f"\n[Combined]  tasks with all features: {len(combined)}")
        if len(combined) < 3:
            print("  Too few samples for combined model — skipping.")
            continue

        X = np.array([
            [r[llm_field],        # LLM valence (ordinal)
             r["llm_nv"] if target_field == "human_sv" else r["llm_sv"],  # cross-modal LLM
             r["audio_v"],        # audio valence (continuous)
             r["eyegaze_v"]]      # eyegaze valence (continuous)
            for r in combined
        ])
        y = np.array([r[target_field] for r in combined])

        # Leave-one-out cross-validation
        loo = LeaveOneOut()
        y_pred_loo = []
        y_true_loo = []
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]
            if len(np.unique(y_train)) < 2:
                # Only one class in training fold — predict majority
                y_pred_loo.append(int(np.bincount(y_train + 1).argmax() - 1))
            else:
                clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=0)
                clf.fit(X_train, y_train)
                y_pred_loo.append(int(clf.predict(X_test)[0]))
            y_true_loo.append(int(y[test_idx[0]]))

        loo_acc = accuracy_score(y_true_loo, y_pred_loo)
        print(f"  LOO-CV accuracy = {loo_acc:.1%}")
        print(f"  per-class: {per_class_acc(y_true_loo, y_pred_loo)}")
        print("  confusion (rows=human, cols=combined):")
        print(confusion_table(y_true_loo, y_pred_loo))

        # Also fit on all data to show coefficients
        if len(np.unique(y)) >= 2:
            clf_full = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=0)
            clf_full.fit(X, y)
            feat_names = ["llm_sv_or_nv", "llm_crossmodal", "audio_valence", "eyegaze_valence"]
            print("  Feature importance (coef magnitude, multinomial):")
            coef_abs = np.abs(clf_full.coef_).mean(axis=0)
            for name, c in sorted(zip(feat_names, coef_abs), key=lambda x: -x[1]):
                print(f"    {name:>25}: {c:.4f}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
