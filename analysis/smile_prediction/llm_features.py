"""
Phase 2: LLM-as-Feature-Extractor for smile prediction.

Asks the LLM to extract structured semantic features from transcript context
around each smile event. These features can then be combined with VAD signals
in the ML pipeline.

Extracted features (all 0–10 scale):
  - humor_level: How humorous/funny is the content?
  - emotional_weight: How emotionally heavy/traumatic is the content?
  - social_courtesy: How much social/politeness signaling is happening?
  - coping_indicator: Signs of coping mechanisms (deflection, minimization)?
  - nostalgia: Warm reminiscence about the past?
  - interviewer_prompt: Was this a reaction to the interviewer's question/comment?
  - topic_shift: Is the speaker transitioning topics?
  - self_referential: Is the speaker talking about themselves/their experience?
  - positive_content: Is the literal content positive?
  - negative_content: Is the literal content negative/sad/painful?

These 10 features provide a rich semantic signal about the conversational context.

Usage:
    python -m analysis.smile_prediction.llm_features --mode annotated --limit 5
    python -m analysis.smile_prediction.llm_features --mode all
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from .dataset import DATA_DIR, SMILE_CLASSES, SmileTask
from .llm_annotate import build_task_list
from .transcript_context import extract_context
from .llm_utils import load_api_key, batch_call, DEFAULT_MODEL

SCRIPT_DIR = Path(__file__).resolve().parent

FEATURE_NAMES = [
    "humor_level",
    "emotional_weight",
    "social_courtesy",
    "coping_indicator",
    "nostalgia",
    "interviewer_prompt",
    "topic_shift",
    "self_referential",
    "positive_content",
    "negative_content",
]

SYSTEM_PROMPT = """\
You are an expert linguist analyzing conversational context in Holocaust \
survivor testimony interviews. You will extract structured features from \
transcript segments to characterize the semantic and emotional context.\
"""

USER_PROMPT_TEMPLATE = """\
A smile was detected on the interviewee's face during this moment in a \
Holocaust survivor testimony interview. Analyze the conversational context \
and extract these features.

{context}

Rate each feature on a 0-10 integer scale:

- **humor_level** (0-10): How humorous, funny, or lighthearted is the content?
- **emotional_weight** (0-10): How emotionally heavy, painful, or traumatic is the content?
- **social_courtesy** (0-10): How much social/politeness signaling is present? \
(acknowledging interviewer, pleasantries, conversational filler)
- **coping_indicator** (0-10): Signs of emotional coping — deflection, \
minimization, ironic distance, or making light of difficult topics?
- **nostalgia** (0-10): Warm reminiscence or fond memory recall?
- **interviewer_prompt** (0-10): Is this smile a reaction to the interviewer's \
question, comment, or behavior (vs. self-generated)?
- **topic_shift** (0-10): Is a topic transition happening near the smile?
- **self_referential** (0-10): Is the speaker discussing their own personal \
experience (vs. general/others' experiences)?
- **positive_content** (0-10): Is the literal semantic content positive, \
happy, or uplifting?
- **negative_content** (0-10): Is the literal semantic content negative, sad, \
painful, or related to suffering?

Also provide a short "topic_tag" (2-5 words) describing the topic being discussed.

Return a JSON object with integer values for each feature and a string for "topic_tag".
Example: {{"humor_level": 7, "emotional_weight": 2, "social_courtesy": 1, \
"coping_indicator": 0, "nostalgia": 3, "interviewer_prompt": 0, "topic_shift": 0, \
"self_referential": 8, "positive_content": 6, "negative_content": 1, \
"topic_tag": "childhood game memory"}}\
"""


def parse_args():
    p = argparse.ArgumentParser(description="LLM feature extraction")
    p.add_argument("--mode", choices=["annotated", "all"], default="annotated")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=50)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def build_prompts(tasks: list[SmileTask]) -> tuple[list[dict], dict]:
    prompts = []
    context_map = {}
    skipped = 0

    for task in tasks:
        ctx = extract_context(task)
        if ctx is None:
            skipped += 1
            continue
        context_map[task.task_number] = ctx

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
            },
        })

    if skipped:
        print(f"  Skipped {skipped} tasks (no transcript)")
    return prompts, context_map


def process_results(raw_results: list[dict], tasks: list[SmileTask]) -> list[dict]:
    """Parse and validate LLM feature responses."""
    output = []
    for r in raw_results:
        tn = r["metadata"]["task_number"]
        vid = r["metadata"]["video_id"]

        if r["result"] is None:
            output.append({"task_number": tn, "video_id": vid, "error": r["error"]})
            continue

        entry = {"task_number": tn, "video_id": vid}
        result = r["result"]

        features = {}
        for feat in FEATURE_NAMES:
            val = result.get(feat, 5)
            try:
                val = int(val)
            except (ValueError, TypeError):
                val = 5
            features[feat] = max(0, min(10, val))

        entry["features"] = features
        entry["topic_tag"] = str(result.get("topic_tag", "unknown"))

        usage = r.get("usage", {})
        entry["prompt_tokens"] = usage.get("prompt_tokens", 0)
        entry["completion_tokens"] = usage.get("completion_tokens", 0)

        output.append(entry)

    return output


def main():
    args = parse_args()

    print(f"=== LLM Feature Extraction ===")
    print(f"Mode: {args.mode}, Model: {args.model}, Concurrency: {args.concurrency}")
    if args.limit:
        print(f"Limit: {args.limit} tasks")
    print()

    api_key = load_api_key()
    tasks = build_task_list(args.mode, args.limit)
    print(f"Tasks: {len(tasks)}")

    print("Building prompts...")
    prompts, context_map = build_prompts(tasks)
    print(f"  {len(prompts)} API calls")
    print()

    if not prompts:
        print("No prompts to send. Exiting.")
        return

    print("Calling OpenRouter...")
    t0 = time.time()
    raw_results = asyncio.run(batch_call(
        prompts, api_key,
        model=args.model,
        temperature=0.0,
        concurrency=args.concurrency,
    ))
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    errors = sum(1 for r in raw_results if r["error"] is not None)
    if errors:
        print(f"  {errors} errors")

    total_prompt = sum(r["usage"].get("prompt_tokens", 0) for r in raw_results)
    total_comp = sum(r["usage"].get("completion_tokens", 0) for r in raw_results)
    print(f"  Tokens: {total_prompt:,} prompt + {total_comp:,} completion")

    print("Processing results...")
    output = process_results(raw_results, tasks)

    # Feature statistics
    valid = [r for r in output if "features" in r]
    if valid:
        print(f"\n=== Feature Statistics ({len(valid)} tasks) ===")
        import numpy as np
        for feat in FEATURE_NAMES:
            vals = [r["features"][feat] for r in valid]
            print(f"  {feat:>20s}: mean={np.mean(vals):.1f} std={np.std(vals):.1f} "
                  f"[{np.min(vals)}-{np.max(vals)}]")

        # Topic tag distribution
        from collections import Counter
        tags = Counter(r["topic_tag"] for r in valid)
        print(f"\nTop 10 topic tags:")
        for tag, count in tags.most_common(10):
            print(f"  {tag:>30s}: {count}")

    # Save
    if args.out is None:
        suffix = f"_{args.mode}"
        if args.limit:
            suffix += f"_limit{args.limit}"
        out_path = SCRIPT_DIR / f"llm_features{suffix}.json"
    else:
        out_path = Path(args.out)

    with open(out_path, "w") as f:
        json.dump({
            "args": vars(args),
            "n_tasks": len(output),
            "feature_names": FEATURE_NAMES,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_comp,
            "elapsed_seconds": elapsed,
            "results": output,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
