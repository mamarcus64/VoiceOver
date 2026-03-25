"""
Phase 1: LLM-as-Annotator for smile classification.

For each smile task, extracts transcript context and asks the LLM to classify
the smile as genuine/polite/masking/not_a_smile with a confidence distribution.

Supports:
  - Single deterministic pass (temperature=0)
  - Multi-pass soft labels (N passes at temperature>0, aggregated)
  - Resumable output (skips tasks already in the output file)

Usage:
    # Classify the 268 annotated tasks (deterministic)
    python -m analysis.smile_prediction.llm_annotate --mode annotated

    # Classify all 18K tasks
    python -m analysis.smile_prediction.llm_annotate --mode all

    # Multi-pass soft labels (5 runs at temp=0.7)
    python -m analysis.smile_prediction.llm_annotate --mode annotated --passes 5 --temperature 0.7

    # Limit to first N tasks for testing
    python -m analysis.smile_prediction.llm_annotate --mode annotated --limit 5
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np

from .dataset import build_tasks, load_manifest, SMILE_CLASSES, SmileTask, DATA_DIR
from .transcript_context import extract_context
from .llm_utils import load_api_key, batch_call, DEFAULT_MODEL

SCRIPT_DIR = Path(__file__).resolve().parent

SYSTEM_PROMPT = """\
You are an expert in nonverbal communication analyzing smiles in Holocaust \
survivor testimony interviews. You understand that smiles in this context can \
serve very different functions — from genuine amusement to social courtesy to \
masking painful emotions.

You will receive transcript context around a moment where a smile was detected \
on the interviewee's face. Based on what is being said, classify the smile.\
"""

USER_PROMPT_TEMPLATE = """\
A smile was detected on the interviewee's face during this moment in a \
Holocaust survivor testimony interview.

{context}

Based on the conversational context, classify this smile into one of these categories:

- **genuine**: Authentic positive emotion — amusement, joy, warmth, fond memory, \
humor. The content naturally evokes a real smile.
- **polite**: Social courtesy — acknowledging the interviewer, conversational \
lubricant, polite agreement. Not driven by the emotional content.
- **masking**: Covering difficult emotions — smiling while discussing pain, loss, \
trauma, or discomfort. A coping mechanism or deflection.
- **not_a_smile**: The context suggests this is unlikely to be a meaningful smile \
(e.g., a facial movement during speech).

Return a JSON object with:
- "classification": your best single label (one of: genuine, polite, masking, not_a_smile)
- "confidence": a probability distribution over all four classes (must sum to 1.0)
- "reasoning": 1-2 sentences explaining your classification

Example: {{"classification": "masking", "confidence": {{"genuine": 0.1, "polite": 0.05, "masking": 0.8, "not_a_smile": 0.05}}, "reasoning": "The interviewee smiles while recounting a traumatic separation, suggesting emotional masking."}}\
"""


def parse_args():
    p = argparse.ArgumentParser(description="LLM smile annotation")
    p.add_argument("--mode", choices=["annotated", "all"], default="annotated",
                   help="annotated = only human-annotated tasks; all = full manifest")
    p.add_argument("--passes", type=int, default=1,
                   help="Number of LLM passes per task (>1 for soft labels)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="LLM temperature (use >0 with --passes >1)")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N tasks (for testing)")
    p.add_argument("--concurrency", type=int, default=50)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--out", type=str, default=None,
                   help="Output JSON path (default: auto-generated)")
    return p.parse_args()


def build_task_list(mode: str, limit: int | None) -> list[SmileTask]:
    """Build the list of tasks to annotate."""
    if mode == "annotated":
        tasks = build_tasks(min_annotators=1, label_smoothing=0.0)
    else:
        manifest = load_manifest()
        tasks = []
        for t in manifest["tasks"]:
            tasks.append(SmileTask(
                task_number=t["task_number"],
                video_id=t["video_id"],
                smile_start=t["smile_start"],
                smile_end=t["smile_end"],
                soft_label=np.zeros(3, dtype=np.float32),
                weight=0.0,
                annotator_count=0,
            ))

    if limit:
        tasks = tasks[:limit]
    return tasks


def build_prompts(tasks: list[SmileTask], n_passes: int) -> tuple[list[dict], dict]:
    """
    Build LLM prompts for all tasks × passes.

    Returns (prompts, context_map) where context_map[task_number] = extracted context.
    """
    prompts = []
    context_map = {}
    skipped = 0

    for task in tasks:
        ctx = extract_context(task)
        if ctx is None:
            skipped += 1
            continue
        context_map[task.task_number] = ctx

        for pass_idx in range(n_passes):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                    context=ctx["full_context"]
                )},
            ]
            prompts.append({
                "messages": messages,
                "metadata": {
                    "task_number": task.task_number,
                    "video_id": task.video_id,
                    "pass_idx": pass_idx,
                },
            })

    if skipped:
        print(f"  Skipped {skipped} tasks (no transcript)")
    return prompts, context_map


def aggregate_results(
    raw_results: list[dict],
    tasks: list[SmileTask],
    context_map: dict,
    n_passes: int,
) -> list[dict]:
    """Aggregate LLM results, merging multi-pass runs into soft labels."""
    CLASSES_4 = ["genuine", "polite", "masking", "not_a_smile"]

    # Group by task_number
    by_task: dict[int, list[dict]] = {}
    for r in raw_results:
        tn = r["metadata"]["task_number"]
        by_task.setdefault(tn, []).append(r)

    output = []
    for task in tasks:
        tn = task.task_number
        if tn not in by_task:
            continue

        runs = by_task[tn]
        successful_runs = [r for r in runs if r["result"] is not None]

        if not successful_runs:
            output.append({
                "task_number": tn,
                "video_id": task.video_id,
                "error": runs[0].get("error", "unknown"),
            })
            continue

        # Collect classifications and confidence distributions
        classifications = []
        confidence_sum = {c: 0.0 for c in CLASSES_4}
        reasonings = []

        for r in successful_runs:
            result = r["result"]
            cls = result.get("classification", "genuine")
            classifications.append(cls)
            conf = result.get("confidence", {})
            for c in CLASSES_4:
                confidence_sum[c] += conf.get(c, 0.0)
            reasoning = result.get("reasoning", "")
            if reasoning:
                reasonings.append(reasoning)

        n = len(successful_runs)
        # Average confidence across passes
        avg_confidence = {c: confidence_sum[c] / n for c in CLASSES_4}

        # Vote-based classification
        from collections import Counter
        vote_counts = Counter(classifications)
        majority_cls = vote_counts.most_common(1)[0][0]

        # Compute the 3-class soft label (excluding not_a_smile)
        smile_conf = {c: avg_confidence[c] for c in ["genuine", "polite", "masking"]}
        smile_total = sum(smile_conf.values())
        if smile_total > 0:
            soft_label_3 = {c: smile_conf[c] / smile_total for c in SMILE_CLASSES}
        else:
            soft_label_3 = {c: 1.0 / 3 for c in SMILE_CLASSES}

        # Not-a-smile discount (analogous to human annotation)
        not_smile_frac = avg_confidence.get("not_a_smile", 0.0)
        weight = 1.0 - not_smile_frac

        ctx = context_map.get(tn, {})

        entry = {
            "task_number": tn,
            "video_id": task.video_id,
            "classification": majority_cls,
            "confidence_4class": avg_confidence,
            "soft_label_3class": soft_label_3,
            "weight": weight,
            "n_passes": n,
            "vote_counts": dict(vote_counts),
            "speaker_at_smile": ctx.get("speaker_at_smile", "unknown"),
            "has_laugh_marker": ctx.get("has_laugh_marker", False),
            "reasoning": reasonings[0] if reasonings else "",
        }

        # Include usage stats
        total_prompt = sum(r["usage"].get("prompt_tokens", 0) for r in successful_runs)
        total_completion = sum(r["usage"].get("completion_tokens", 0) for r in successful_runs)
        entry["total_prompt_tokens"] = total_prompt
        entry["total_completion_tokens"] = total_completion

        output.append(entry)

    return output


def main():
    args = parse_args()

    print(f"=== LLM Smile Annotation ===")
    print(f"Mode: {args.mode}, Passes: {args.passes}, Temp: {args.temperature}")
    print(f"Model: {args.model}, Concurrency: {args.concurrency}")
    if args.limit:
        print(f"Limit: {args.limit} tasks")
    print()

    api_key = load_api_key()

    tasks = build_task_list(args.mode, args.limit)
    print(f"Tasks to process: {len(tasks)}")

    print("Extracting transcript context...")
    prompts, context_map = build_prompts(tasks, args.passes)
    print(f"  {len(prompts)} API calls to make ({len(context_map)} tasks × {args.passes} passes)")
    print()

    if not prompts:
        print("No prompts to send. Exiting.")
        return

    print("Calling OpenRouter...")
    t0 = time.time()
    raw_results = asyncio.run(batch_call(
        prompts, api_key,
        model=args.model,
        temperature=args.temperature,
        concurrency=args.concurrency,
    ))
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    errors = sum(1 for r in raw_results if r["error"] is not None)
    if errors:
        print(f"  {errors} errors")

    total_prompt_tokens = sum(r["usage"].get("prompt_tokens", 0) for r in raw_results)
    total_completion_tokens = sum(r["usage"].get("completion_tokens", 0) for r in raw_results)
    print(f"  Tokens: {total_prompt_tokens:,} prompt + {total_completion_tokens:,} completion")
    print()

    print("Aggregating results...")
    output = aggregate_results(raw_results, tasks, context_map, args.passes)

    # Summary
    from collections import Counter
    cls_dist = Counter(r.get("classification") for r in output if "classification" in r)
    print(f"\nClassification distribution:")
    for c in ["genuine", "polite", "masking", "not_a_smile"]:
        print(f"  {c:>12s}: {cls_dist.get(c, 0)}")

    # Save
    if args.out is None:
        suffix = f"_{args.mode}"
        if args.passes > 1:
            suffix += f"_{args.passes}pass"
        if args.limit:
            suffix += f"_limit{args.limit}"
        out_path = SCRIPT_DIR / f"llm_annotations{suffix}.json"
    else:
        out_path = Path(args.out)

    with open(out_path, "w") as f:
        json.dump({
            "args": vars(args),
            "n_tasks": len(output),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "elapsed_seconds": elapsed,
            "results": output,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
