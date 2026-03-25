"""
Phase 1b: Alignment analysis — compare LLM annotations vs human annotations.

Loads LLM annotation output and compares against human soft labels to measure
how well the LLM agrees with human annotators.

Usage:
    python -m analysis.smile_prediction.llm_alignment <llm_annotations.json>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .dataset import build_tasks, SMILE_CLASSES, CLASS_TO_IDX


def parse_args():
    p = argparse.ArgumentParser(description="LLM vs human alignment analysis")
    p.add_argument("llm_file", type=str, help="Path to llm_annotations JSON output")
    p.add_argument("--min-annotators", type=int, default=1,
                   help="Only compare tasks with this many human annotators")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.llm_file) as f:
        llm_data = json.load(f)

    llm_results = {r["task_number"]: r for r in llm_data["results"]
                   if "classification" in r}
    print(f"LLM annotations: {len(llm_results)} tasks")

    human_tasks = build_tasks(min_annotators=args.min_annotators, label_smoothing=0.0)
    human_by_tn = {t.task_number: t for t in human_tasks}

    # Overlap
    overlap = set(llm_results.keys()) & set(human_by_tn.keys())
    print(f"Human annotations (min_ann={args.min_annotators}): {len(human_by_tn)} tasks")
    print(f"Overlap: {len(overlap)} tasks")

    if not overlap:
        print("No overlapping tasks to compare.")
        return

    # --- Hard agreement ---
    correct = 0
    total = 0
    from collections import Counter
    confusion = Counter()

    for tn in sorted(overlap):
        human = human_by_tn[tn]
        llm = llm_results[tn]

        human_cls = SMILE_CLASSES[human.soft_label.argmax()]
        llm_cls = llm["classification"]
        # Map not_a_smile to the LLM's second-best for hard comparison
        if llm_cls == "not_a_smile":
            conf = llm.get("soft_label_3class", {})
            if conf:
                llm_cls = max(conf, key=conf.get)
            else:
                llm_cls = "genuine"

        confusion[(human_cls, llm_cls)] += 1
        if human_cls == llm_cls:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"\n=== Hard Agreement (LLM top class vs human majority) ===")
    print(f"Accuracy: {correct}/{total} = {accuracy:.3f}")

    # Confusion matrix
    print(f"\nConfusion matrix (rows=human, cols=LLM):")
    header = f"{'':>10s}" + "".join(f"{c:>10s}" for c in SMILE_CLASSES)
    print(header)
    for hc in SMILE_CLASSES:
        row = f"{hc:>10s}"
        for lc in SMILE_CLASSES:
            row += f"{confusion.get((hc, lc), 0):>10d}"
        print(row)

    # Per-class
    print(f"\nPer-class:")
    for c in SMILE_CLASSES:
        tp = confusion.get((c, c), 0)
        fn = sum(confusion.get((c, lc), 0) for lc in SMILE_CLASSES if lc != c)
        fp = sum(confusion.get((hc, c), 0) for hc in SMILE_CLASSES if hc != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"  {c:>10s}: P={prec:.3f} R={rec:.3f} F1={f1:.3f} (n={tp+fn})")

    # --- Distributional agreement (soft labels) ---
    print(f"\n=== Distributional Agreement ===")

    brier_scores = []
    kl_divs = []

    for tn in sorted(overlap):
        human = human_by_tn[tn]
        llm = llm_results[tn]

        h = human.soft_label  # (3,) already normalized
        l_dict = llm.get("soft_label_3class", {})
        l = np.array([l_dict.get(c, 1/3) for c in SMILE_CLASSES], dtype=np.float32)
        l = l / l.sum()  # ensure normalized

        # Brier score
        brier = ((h - l) ** 2).sum()
        brier_scores.append(brier)

        # KL divergence (human || LLM)
        h_safe = np.clip(h, 1e-8, 1.0)
        l_safe = np.clip(l, 1e-8, 1.0)
        kl = (h_safe * np.log(h_safe / l_safe)).sum()
        kl_divs.append(kl)

    print(f"Brier score (human vs LLM): {np.mean(brier_scores):.4f} "
          f"(std={np.std(brier_scores):.4f})")
    print(f"KL divergence (human || LLM): {np.mean(kl_divs):.4f} "
          f"(std={np.std(kl_divs):.4f})")

    # --- Compare to prior baseline ---
    # What if LLM just predicted the human class prior?
    all_human_labels = np.stack([human_by_tn[tn].soft_label for tn in overlap])
    prior = all_human_labels.mean(axis=0)
    prior_brier = []
    for tn in overlap:
        h = human_by_tn[tn].soft_label
        brier = ((h - prior) ** 2).sum()
        prior_brier.append(brier)

    print(f"\nPrior baseline Brier: {np.mean(prior_brier):.4f}")
    print(f"LLM Brier skill vs prior: {1 - np.mean(brier_scores) / np.mean(prior_brier):+.3f}")

    # --- Focus on multi-annotator tasks ---
    multi_overlap = [tn for tn in overlap if human_by_tn[tn].annotator_count >= 2]
    if multi_overlap:
        print(f"\n=== Multi-annotator subset ({len(multi_overlap)} tasks) ===")
        correct_m = sum(1 for tn in multi_overlap
                        if SMILE_CLASSES[human_by_tn[tn].soft_label.argmax()]
                        == (llm_results[tn]["classification"]
                            if llm_results[tn]["classification"] != "not_a_smile"
                            else max(llm_results[tn].get("soft_label_3class", {}),
                                     key=llm_results[tn].get("soft_label_3class", {}).get,
                                     default="genuine")))
        print(f"Accuracy: {correct_m}/{len(multi_overlap)} = "
              f"{correct_m / len(multi_overlap):.3f}")

        brier_m = []
        for tn in multi_overlap:
            h = human_by_tn[tn].soft_label
            l_dict = llm_results[tn].get("soft_label_3class", {})
            l = np.array([l_dict.get(c, 1/3) for c in SMILE_CLASSES], dtype=np.float32)
            l = l / l.sum()
            brier_m.append(((h - l) ** 2).sum())
        print(f"Brier: {np.mean(brier_m):.4f}")

    # --- Show some examples ---
    print(f"\n=== Example comparisons (first 10) ===")
    for tn in sorted(overlap)[:10]:
        h = human_by_tn[tn]
        l = llm_results[tn]
        h_str = ", ".join(f"{c}={h.soft_label[i]:.2f}" for i, c in enumerate(SMILE_CLASSES))
        l_3 = l.get("soft_label_3class", {})
        l_str = ", ".join(f"{c}={l_3.get(c, 0):.2f}" for c in SMILE_CLASSES)
        match = "✓" if (SMILE_CLASSES[h.soft_label.argmax()] ==
                        (l["classification"] if l["classification"] != "not_a_smile"
                         else max(l_3, key=l_3.get, default="genuine"))) else "✗"
        print(f"  Task {tn:>3d} {match}  Human: [{h_str}]  LLM: [{l_str}]  "
              f"({l.get('reasoning', '')[:60]})")


if __name__ == "__main__":
    main()
