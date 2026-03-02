#!/usr/bin/env python3
"""
LLM-based transcript quality pass using OpenRouter API.

Uses a "corrections-only" approach: the LLM returns only a list of utterance
indices that need speaker role changes. We apply changes programmatically,
so text/timestamps/word arrays are never at risk of corruption.

Resumable: skips files whose output already exists in the output directory.
Parallel: runs up to --concurrency (default 50) OpenRouter calls at once.
"""

import argparse
import asyncio
import copy
import json
import os
import sys
from pathlib import Path

import aiohttp

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """\
You are a transcript quality checker for Holocaust survivor interview videos.

You will receive a numbered list of utterances from an interview transcript. \
Each utterance has an index, a speaker role ("interviewer" or "interviewee"), \
a speaker tag (initials), and the text.

The INTERVIEWER asks questions, gives directions, introduces the session, \
and makes short prompts. The INTERVIEWEE tells their story, answers \
questions, and shares personal experiences, memories, and narratives.

Your task: identify utterances where the speaker role is WRONG. Return a \
JSON object with key "corrections" — an array of objects, each with:
  - "index": the utterance index (integer)
  - "correct_speaker": the role it SHOULD be ("interviewer" or "interviewee")
  - "reason": brief explanation (one sentence)

If no corrections are needed, return: {"corrections": []}

Only flag clear misattributions. Do NOT flag ambiguous cases.\
"""

USER_PROMPT_TEMPLATE = """\
Review this transcript ({num_utterances} utterances) and identify any \
speaker role misattributions.

{utterance_list}

Return JSON: {{"corrections": [{{"index": N, "correct_speaker": "...", "reason": "..."}}]}}
If nothing needs fixing: {{"corrections": []}}\
"""


def parse_args():
    p = argparse.ArgumentParser(description="LLM-based transcript quality pass")
    p.add_argument(
        "--input-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "transcripts"),
    )
    p.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "transcripts_llm"),
    )
    p.add_argument("--limit", type=int, default=None, help="Only process first N files")
    p.add_argument("--model", default="openai/gpt-oss-120b")
    p.add_argument("--provider", default=None, help="e.g. 'Cerebras'")
    p.add_argument("--diff-report", action="store_true")
    p.add_argument("--concurrency", type=int, default=50)
    p.add_argument("--no-resume", action="store_true", help="Re-process even if output exists")
    return p.parse_args()


def build_utterance_list(utterances):
    """Build a compact numbered list for the prompt (no word arrays, no timestamps)."""
    lines = []
    for i, u in enumerate(utterances):
        speaker = u["speaker"]
        tag = u.get("tag", "?")
        text = u["text"]
        if len(text) > 200:
            text = text[:200] + "..."
        lines.append(f"[{i}] {speaker} ({tag}): {text}")
    return "\n".join(lines)


async def call_openrouter(session, utterances, model, api_key, provider=None):
    """Ask the LLM for corrections only. Returns (corrections_list, usage)."""
    utterance_list = build_utterance_list(utterances)

    user_msg = USER_PROMPT_TEMPLATE.format(
        num_utterances=len(utterances),
        utterance_list=utterance_list,
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
    }

    if provider:
        payload["provider"] = {"only": [provider]}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with session.post(
        OPENROUTER_URL, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()

    usage = data.get("usage", {})
    content = data["choices"][0]["message"]["content"]
    parsed = json.loads(content)

    corrections = parsed.get("corrections", [])
    if not isinstance(corrections, list):
        corrections = []

    return corrections, usage


def apply_corrections(utterances, corrections):
    """Apply speaker corrections to the transcript. Returns (corrected, applied_diffs)."""
    corrected = copy.deepcopy(utterances)

    interviewer_tags = set()
    interviewee_tags = set()
    for u in utterances:
        tag = u.get("tag")
        if tag:
            if u["speaker"] == "interviewer":
                interviewer_tags.add(tag)
            else:
                interviewee_tags.add(tag)

    applied = []
    for c in corrections:
        idx = c.get("index")
        new_speaker = c.get("correct_speaker")

        if not isinstance(idx, int) or idx < 0 or idx >= len(corrected):
            continue
        if new_speaker not in ("interviewer", "interviewee"):
            continue
        if corrected[idx]["speaker"] == new_speaker:
            continue

        old_speaker = corrected[idx]["speaker"]
        old_tag = corrected[idx].get("tag")

        corrected[idx]["speaker"] = new_speaker

        if new_speaker == "interviewer" and interviewer_tags:
            corrected[idx]["tag"] = sorted(interviewer_tags)[0]
        elif new_speaker == "interviewee" and interviewee_tags:
            corrected[idx]["tag"] = sorted(interviewee_tags)[0]

        applied.append({
            "index": idx,
            "text_preview": corrected[idx]["text"][:120],
            "old_speaker": old_speaker,
            "new_speaker": new_speaker,
            "old_tag": old_tag,
            "new_tag": corrected[idx].get("tag"),
            "reason": c.get("reason", ""),
        })

    return corrected, applied


def write_diff_report(all_diffs, output_path, stats):
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("LLM Transcript Quality Pass — Diff Report\n")
        f.write("=" * 80 + "\n\n")
        for k in ("processed", "changed", "unchanged", "errors", "total_diffs",
                   "total_prompt_tokens", "total_completion_tokens"):
            f.write(f"{k:>25s}: {stats.get(k, 0)}\n")
        f.write("\n")

        if not all_diffs:
            f.write("No changes were made to any transcripts.\n")
            return

        current_vid = None
        for vid, d in all_diffs:
            if vid != current_vid:
                current_vid = vid
                f.write("-" * 60 + "\n")
                f.write(f"File: {vid}.json\n")
                f.write("-" * 60 + "\n")
            f.write(f"  Utterance #{d['index']}:\n")
            f.write(f"    Text: \"{d['text_preview']}...\"\n")
            f.write(f"    speaker: {d['old_speaker']!r} -> {d['new_speaker']!r}\n")
            f.write(f"    tag: {d['old_tag']!r} -> {d['new_tag']!r}\n")
            if d["reason"]:
                f.write(f"    reason: {d['reason']}\n")
            f.write("\n")


async def process_one(session, semaphore, fpath, output_dir, model, api_key, provider, label):
    """Process a single transcript file. Returns (video_id, diffs, usage, error)."""
    video_id = fpath.stem
    async with semaphore:
        try:
            with open(fpath) as f:
                original = json.load(f)

            corrections, usage = await call_openrouter(
                session, original, model, api_key, provider
            )

            corrected, applied = apply_corrections(original, corrections)

            out_path = output_dir / f"{video_id}.json"
            with open(out_path, "w") as f:
                json.dump(corrected, f, indent=2)

            status = f"CHANGED ({len(applied)} utterances)" if applied else "OK (no changes)"
            print(f"  {label} {video_id}: {status}", flush=True)

            return video_id, applied, usage, None

        except Exception as e:
            print(f"  {label} {video_id}: ERROR: {e}", flush=True)
            return video_id, [], {}, e


async def async_main():
    args = parse_args()

    api_key = os.environ.get("VOICES_OPENROUTER_KEY")
    if not api_key:
        print("ERROR: VOICES_OPENROUTER_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    if args.limit:
        json_files = json_files[: args.limit]

    total_files = len(json_files)

    if not args.no_resume:
        already_done = {p.stem for p in output_dir.glob("*.json") if not p.name.startswith("_")}
        pending_files = [f for f in json_files if f.stem not in already_done]
        skipped = total_files - len(pending_files)
    else:
        pending_files = json_files
        skipped = 0

    print(f"Input dir:    {input_dir}")
    print(f"Output dir:   {output_dir}")
    print(f"Model:        {args.model}")
    print(f"Provider:     {args.provider or '(default)'}")
    print(f"Concurrency:  {args.concurrency}")
    print(f"Total files:  {total_files}")
    if skipped:
        print(f"Skipped (already done): {skipped}")
    print(f"Files to process: {len(pending_files)}")
    print()

    if not pending_files:
        print("Nothing to do — all files already processed.")
        return

    stats = {
        "processed": 0, "changed": 0, "unchanged": 0, "errors": 0,
        "total_diffs": 0, "total_prompt_tokens": 0, "total_completion_tokens": 0,
        "skipped": skipped,
    }
    all_diffs = []

    semaphore = asyncio.Semaphore(args.concurrency)

    connector = aiohttp.TCPConnector(limit=args.concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for idx, fpath in enumerate(pending_files):
            label = f"[{idx + 1}/{len(pending_files)}]"
            tasks.append(
                process_one(session, semaphore, fpath, output_dir, args.model, api_key, args.provider, label)
            )

        results = await asyncio.gather(*tasks)

    for video_id, applied, usage, error in results:
        stats["processed"] += 1
        stats["total_prompt_tokens"] += usage.get("prompt_tokens", 0)
        stats["total_completion_tokens"] += usage.get("completion_tokens", 0)

        if error:
            stats["errors"] += 1
        elif applied:
            stats["changed"] += 1
            stats["total_diffs"] += len(applied)
            all_diffs.extend((video_id, d) for d in applied)
        else:
            stats["unchanged"] += 1

    print()
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    for k, v in stats.items():
        print(f"  {k:>25s}: {v}")

    if args.diff_report:
        report_path = output_dir / "_diff_report.txt"
        write_diff_report(all_diffs, report_path, stats)
        print(f"\nDiff report written to: {report_path}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
