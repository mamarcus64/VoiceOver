#!/usr/bin/env python3
"""
Evaluate memory_type accuracy on the 100-task validation set (tasks 100-199).

The 100 tasks are FIXED — they come from data/recall_facts_annotations/tasks.json
which must NOT be regenerated or modified. Ground truth is derived once from that
file (llm_memory_type), with 16 known-wrong tasks flipped, and is hardcoded below.

Workflow:
  1. Save a timestamped snapshot of the current prompt_template.txt
  2. (Optional --reset) Delete annotation files for the validation transcripts
  3. Re-annotate the 8 validation transcripts via annotate_recall/annotate.py
  4. For each fixed task, look up the new label from the annotation files
  5. Compare against ground truth, print results + confidence threshold table
  6. Save results to eval_results/

Saved to: data/recall_facts_annotations/eval_results/
  <timestamp>_prompt.txt    -- prompt in use during this run
  <timestamp>_results.json  -- per-task breakdown + summary

Usage:
    python scripts/eval_memory_type_validation.py               # re-annotate, keep existing files
    python scripts/eval_memory_type_validation.py --reset       # delete annotation files first
    python scripts/eval_memory_type_validation.py --model openai/gpt-4o
    python scripts/eval_memory_type_validation.py --score-only  # skip re-annotation, just score
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT       = Path(__file__).resolve().parent.parent
ANNOTATE_SCRIPT = REPO_ROOT / "annotate_recall" / "annotate.py"
PROMPT_TEMPLATE = REPO_ROOT / "annotate_recall" / "prompt_template.txt"
TASKS_FILE      = REPO_ROOT / "data" / "recall_facts_annotations" / "tasks.json"
ANNOTATIONS_DIR = REPO_ROOT / "data" / "llm_annotated_recall_facts"
RESULTS_DIR     = REPO_ROOT / "data" / "recall_facts_annotations" / "eval_results"

# Transcripts that contain the 100 validation tasks — derived from tasks.json, do not change.
VALIDATION_STEMS = [
    "10.4", "10.5", "10.6",
    "10010.1", "10098.8", "10119.1", "10162.1", "10162.6",
]

# Tasks where the ground truth is the FLIP of the original llm_memory_type in tasks.json.
WRONG_TASK_IDS = {
    114, 119, 121, 122, 126, 147, 149,
    159, 165, 168, 174, 185, 187, 188, 189, 199,
}

BASELINE = 84  # confirmed accuracy of the original prompt on these tasks

# ── Frozen ground truth ───────────────────────────────────────────────────────
# Derived once from data/recall_facts_annotations/tasks.json (restored original):
#   GT = llm_memory_type for the 84 confirmed-correct tasks
#   GT = flip(llm_memory_type) for the 16 WRONG_TASK_IDS
# Do NOT re-derive this from tasks.json at runtime.
GROUND_TRUTH = {
    100: "external",  101: "external",  102: "external",  103: "internal",
    104: "internal",  105: "external",  106: "external",  107: "external",
    108: "internal",  109: "internal",  110: "external",  111: "external",
    112: "external",  113: "external",  114: "external",  115: "external",
    116: "external",  117: "internal",  118: "external",  119: "external",
    120: "external",  121: "external",  122: "external",  123: "internal",
    124: "internal",  125: "internal",  126: "internal",  127: "external",
    128: "internal",  129: "internal",  130: "external",  131: "internal",
    132: "external",  133: "internal",  134: "external",  135: "external",
    136: "external",  137: "external",  138: "internal",  139: "internal",
    140: "internal",  141: "internal",  142: "external",  143: "internal",
    144: "external",  145: "external",  146: "internal",  147: "external",
    148: "internal",  149: "external",  150: "internal",  151: "external",
    152: "external",  153: "external",  154: "internal",  155: "external",
    156: "external",  157: "internal",  158: "external",  159: "internal",
    160: "internal",  161: "internal",  162: "external",  163: "external",
    164: "internal",  165: "external",  166: "external",  167: "internal",
    168: "external",  169: "internal",  170: "internal",  171: "internal",
    172: "internal",  173: "internal",  174: "external",  175: "internal",
    176: "internal",  177: "internal",  178: "external",  179: "internal",
    180: "internal",  181: "internal",  182: "internal",  183: "internal",
    184: "external",  185: "external",  186: "internal",  187: "external",
    188: "external",  189: "internal",  190: "external",  191: "internal",
    192: "external",  193: "internal",  194: "internal",  195: "external",
    196: "internal",  197: "external",  198: "internal",  199: "external",
}


# ── Annotation lookup ─────────────────────────────────────────────────────────

def load_annotation_index() -> dict[str, dict[str, dict]]:
    """
    Load all 8 validation transcript annotation files into a nested dict:
      { transcript_id: { sentence_text: {memory_type, confidence} } }
    If an annotation file is missing, that transcript is absent from the index.
    """
    index: dict[str, dict[str, dict]] = {}
    for stem in VALIDATION_STEMS:
        p = ANNOTATIONS_DIR / f"{stem}.json"
        if not p.exists():
            print(f"  [WARN] Annotation file missing: {p.name}")
            continue
        try:
            doc = json.loads(p.read_text())
        except Exception as e:
            print(f"  [WARN] Cannot read {p.name}: {e}")
            continue
        by_text: dict[str, dict] = {}
        for sent in doc.get("sentences", []):
            text = sent.get("text", "").strip()
            if text and text not in by_text:
                by_text[text] = {
                    "memory_type": sent.get("memory_type", "external"),
                }
        index[stem] = by_text
    return index


# ── Scoring ───────────────────────────────────────────────────────────────────

def compute_accuracy(ann_index: dict[str, dict[str, dict]]) -> dict:
    """
    Score all 100 fixed tasks against GROUND_TRUTH using fresh annotations
    looked up from ann_index by (transcript_id, sentence_text).
    """
    tasks_data = json.loads(TASKS_FILE.read_text())
    mt_tasks   = [t for t in tasks_data["tasks"] if t["type"] == "memory_type"]

    per_task: dict[int, dict] = {}
    missing: list[int] = []

    for t in mt_tasks:
        tid   = t["id"]
        text  = t["sentence_text"].strip()
        tstem = t["transcript_id"]

        if tid not in GROUND_TRUTH:
            continue

        ann = ann_index.get(tstem, {}).get(text)
        if ann is None:
            missing.append(tid)
            continue

        per_task[tid] = {
            "sentence":       text,
            "transcript_id":  tstem,
            "ground_truth":   GROUND_TRUTH[tid],
            "llm":            ann["memory_type"],
            "correct":        ann["memory_type"] == GROUND_TRUTH[tid],
            "was_wrong_task": tid in WRONG_TASK_IDS,
        }

    if missing:
        print(f"  [WARN] {len(missing)} tasks not found in annotation files: {missing}")

    correct = sum(1 for v in per_task.values() if v["correct"])

    # Cohen's kappa
    n = len(per_task)
    kappa = 0.0
    if n > 0:
        po = correct / n
        p_gt_int  = sum(1 for v in per_task.values() if v["ground_truth"] == "internal") / n
        p_llm_int = sum(1 for v in per_task.values() if v["llm"] == "internal") / n
        pe = p_gt_int * p_llm_int + (1 - p_gt_int) * (1 - p_llm_int)
        kappa = (po - pe) / (1 - pe) if pe < 1.0 else 1.0

    return {"correct": correct, "total": n, "kappa": round(kappa, 3), "per_task": per_task}


# ── Subprocess helper ─────────────────────────────────────────────────────────

def run(cmd: list, **kwargs):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-oss-120b",
                        help="Model ID (default: openai/gpt-oss-120b)")
    parser.add_argument("--reset", action="store_true",
                        help="Delete existing annotation files for the validation transcripts first")
    parser.add_argument("--score-only", action="store_true",
                        help="Skip re-annotation; score from whatever annotation files exist")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # ── 1. snapshot the prompt ─────────────────────────────────────────────
    prompt_snapshot = RESULTS_DIR / f"{timestamp}_prompt.txt"
    shutil.copy(PROMPT_TEMPLATE, prompt_snapshot)
    print(f"[1] Prompt snapshot → eval_results/{prompt_snapshot.name}")

    if not args.score_only:
        # ── 2. optionally reset ────────────────────────────────────────────
        if args.reset:
            print(f"\n[2] Resetting {len(VALIDATION_STEMS)} validation transcripts…")
            for stem in VALIDATION_STEMS:
                run([sys.executable, str(ANNOTATE_SCRIPT), "--reset-file", stem])
        else:
            print("\n[2] Skipped reset (pass --reset to delete existing annotation files first)")

        # ── 3. re-annotate ────────────────────────────────────────────────
        print(f"\n[3] Re-annotating {len(VALIDATION_STEMS)} transcripts (model={args.model})…")
        for stem in VALIDATION_STEMS:
            print(f"\n  → {stem}")
            run([sys.executable, str(ANNOTATE_SCRIPT),
                 "--file", f"{stem}.json",
                 "--model", args.model])
    else:
        print("\n[2-3] Skipped (--score-only)")

    # ── 4. score ──────────────────────────────────────────────────────────
    print("\n[4] Loading annotations and scoring…")
    ann_index = load_annotation_index()
    results   = compute_accuracy(ann_index)

    correct = results["correct"]
    total   = results["total"]
    pct     = correct / total * 100 if total else 0
    delta   = correct - BASELINE

    kappa = results["kappa"]

    print(f"\n{'='*55}")
    print(f"  Accuracy : {correct}/{total} = {pct:.1f}%")
    print(f"  Kappa    : {kappa:.3f}")
    print(f"  Baseline : {BASELINE}/100 = {BASELINE}.0%")
    print(f"  Change   : {delta:+d} tasks")
    print(f"{'='*55}")

    # disagreements only
    wrong = {tid: p for tid, p in results["per_task"].items() if not p["correct"]}
    print(f"\n  {'ID':>4}  {'GT':8} {'LLM':8}  sentence")
    for tid in sorted(wrong.keys()):
        p = wrong[tid]
        print(f"  {tid:>4}  {p['ground_truth']:8} {p['llm']:8}  '{p['sentence']}'")

    # ── 5. save results ────────────────────────────────────────────────────
    results["timestamp"]       = timestamp
    results["model"]           = args.model
    results["prompt_snapshot"] = prompt_snapshot.name
    results["accuracy_pct"]    = round(pct, 1)
    results["kappa"]           = kappa
    results["baseline"]        = BASELINE
    results["delta"]           = delta

    results_file = RESULTS_DIR / f"{timestamp}_results.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"\n[5] Results saved → eval_results/{results_file.name}")


if __name__ == "__main__":
    main()
