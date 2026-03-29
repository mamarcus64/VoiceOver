#!/usr/bin/env python3
"""
Two-pass correction of interviewer → interviewee mislabelings in transcripts.

PASS 1 (batch screening):
  For each file, send all "interviewer"-labeled utterances with ≥5 words to the LLM.
  The LLM returns one of three verdicts per utterance:
    "correct"    – leave as-is
    "flip"       – definitely interviewee, flip immediately
    "uncertain"  – needs surrounding context

PASS 2 (context-aware):
  For each "uncertain" utterance, send it with ±2 surrounding utterances from the
  full transcript. LLM returns "correct" or "flip".

Both passes edit files in-place. Progress is tracked in a JSON sidecar file so
the script is fully resumable.

Usage:
  VOICES_OPENROUTER_KEY=sk-... python scripts/fix_interviewer_labels.py
  VOICES_OPENROUTER_KEY=sk-... python scripts/fix_interviewer_labels.py --limit 10
  VOICES_OPENROUTER_KEY=sk-... python scripts/fix_interviewer_labels.py --concurrency 40
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

# ── Prompts ───────────────────────────────────────────────────────────────────

PASS1_SYSTEM = """\
You are a transcript quality checker for Holocaust survivor interview videos.

The INTERVIEWER asks questions, gives brief prompts, and facilitates the conversation.
The INTERVIEWEE tells their personal story, recounts memories, and gives extended \
first-person narratives.

You will receive a list of utterances that are currently labeled "interviewer". \
Some are correctly labeled; others are actually the interviewee's narration that \
was misattributed. Your task is to classify EACH utterance with one of:

  "correct"   – this utterance is genuinely the interviewer speaking
  "flip"      – this is clearly the interviewee (personal narrative, "my father", \
"we were", "I said", extended memory, etc.)
  "uncertain" – ambiguous; needs surrounding context to decide

Return a JSON object: {"verdicts": [{"index": N, "verdict": "..."}, ...]}
Include every utterance index you were given.\
"""

PASS1_USER_TEMPLATE = """\
Classify each of these currently-labeled "interviewer" utterances.

{utterance_list}

Return JSON: {{"verdicts": [{{"index": N, "verdict": "correct|flip|uncertain"}}]}}\
"""

PASS2_SYSTEM = """\
You are a transcript quality checker for Holocaust survivor interview videos.

The INTERVIEWER asks questions and brief prompts.
The INTERVIEWEE gives personal narratives and extended memories.

You will see a short excerpt from a transcript. One utterance is marked [REVIEW].
Decide whether [REVIEW] is correctly labeled as "interviewer" or is actually \
the interviewee speaking.

Return JSON: {"verdict": "correct"} or {"verdict": "flip"}\
"""

PASS2_USER_TEMPLATE = """\
Context excerpt (± 2 utterances around the one under review):

{context_block}

Is the utterance marked [REVIEW] correctly labeled as "interviewer", \
or is it actually the interviewee?

Return JSON: {{"verdict": "correct"}} or {{"verdict": "flip"}}\
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

MIN_WORDS = 5  # utterances with fewer words are skipped entirely


def word_count(text: str) -> int:
    return len(text.split())


def nearest_interviewee_tag(utterances: list, flip_idx: int) -> str | None:
    """Return the tag of the nearest interviewee utterance before flip_idx."""
    for i in range(flip_idx - 1, -1, -1):
        if utterances[i]["speaker"] == "interviewee" and utterances[i].get("tag"):
            return utterances[i]["tag"]
    # fall back to any interviewee tag in the file
    for u in utterances:
        if u["speaker"] == "interviewee" and u.get("tag"):
            return u["tag"]
    return None


def build_pass1_list(utterances: list, ir_indices: list[int]) -> str:
    lines = []
    for idx in ir_indices:
        u = utterances[idx]
        text = u["text"]
        if len(text) > 300:
            text = text[:300] + "…"
        lines.append(f"[{idx}] interviewer: {text}")
    return "\n".join(lines)


def build_pass2_context(utterances: list, target_idx: int, window: int = 2) -> str:
    lo = max(0, target_idx - window)
    hi = min(len(utterances) - 1, target_idx + window)
    lines = []
    for i in range(lo, hi + 1):
        u = utterances[i]
        text = u["text"]
        if len(text) > 300:
            text = text[:300] + "…"
        marker = " [REVIEW]" if i == target_idx else ""
        lines.append(f"[{i}] {u['speaker']}{marker}: {text}")
    return "\n".join(lines)


def apply_flips(utterances: list, flip_indices: list[int]) -> list[dict]:
    """Flip speaker on given indices in-place (on a deep copy). Returns diffs."""
    diffs = []
    for idx in flip_indices:
        u = utterances[idx]
        old_tag = u.get("tag")
        new_tag = nearest_interviewee_tag(utterances, idx)
        u["speaker"] = "interviewee"
        if new_tag:
            u["tag"] = new_tag
        diffs.append({
            "index": idx,
            "text_preview": u["text"][:120],
            "old_tag": old_tag,
            "new_tag": new_tag,
        })
    return diffs


# ── OpenRouter calls ──────────────────────────────────────────────────────────

async def _post(session: aiohttp.ClientSession, payload: dict, api_key: str,
                provider: str | None) -> dict:
    if provider:
        payload["provider"] = {"only": [provider]}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with session.post(
        OPENROUTER_URL, json=payload, headers=headers,
        timeout=aiohttp.ClientTimeout(total=120),
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


def _extract_json(content: str) -> dict:
    """Robustly extract a JSON object from an LLM response string.

    Tries in order:
      1. Direct parse.
      2. Strip markdown fences then parse.
      3. Find the outermost {...} block and parse that.
    Returns {} on total failure.
    """
    text = content.strip()
    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2. Strip markdown code fences
    stripped = text.lstrip("`").rstrip("`")
    if stripped.startswith("json"):
        stripped = stripped[4:]
    try:
        return json.loads(stripped.strip())
    except json.JSONDecodeError:
        pass
    # 3. Find outermost { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {}


async def call_pass1(session, ir_indices: list[int], utterances: list,
                     model: str, api_key: str, provider: str | None
                     ) -> tuple[list[int], list[int], dict]:
    """Returns (flip_indices, uncertain_indices, usage)."""
    ulist = build_pass1_list(utterances, ir_indices)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": PASS1_SYSTEM},
            {"role": "user", "content": PASS1_USER_TEMPLATE.format(utterance_list=ulist)},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
    }
    data = await _post(session, payload, api_key, provider)
    usage = data.get("usage", {})
    parsed = _extract_json(data["choices"][0]["message"]["content"])
    verdicts = parsed.get("verdicts", [])
    if not isinstance(verdicts, list):
        verdicts = []

    flip_indices, uncertain_indices = [], []
    ir_set = set(ir_indices)
    for v in verdicts:
        idx = v.get("index")
        verdict = v.get("verdict", "correct")
        if not isinstance(idx, int) or idx not in ir_set:
            continue
        if verdict == "flip":
            flip_indices.append(idx)
        elif verdict == "uncertain":
            uncertain_indices.append(idx)
    return flip_indices, uncertain_indices, usage


async def call_pass2(session, target_idx: int, utterances: list,
                     model: str, api_key: str, provider: str | None
                     ) -> tuple[bool, dict]:
    """Returns (should_flip, usage)."""
    ctx = build_pass2_context(utterances, target_idx, window=2)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": PASS2_SYSTEM},
            {"role": "user", "content": PASS2_USER_TEMPLATE.format(context_block=ctx)},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
    }
    data = await _post(session, payload, api_key, provider)
    usage = data.get("usage", {})
    parsed = _extract_json(data["choices"][0]["message"]["content"])
    verdict = parsed.get("verdict", "correct")
    return verdict == "flip", usage


# ── Per-file processing ───────────────────────────────────────────────────────

async def process_file(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    fpath: Path,
    model: str,
    api_key: str,
    provider: str | None,
    label: str,
) -> dict:
    """Process one transcript file through both passes. Returns a result dict."""
    video_id = fpath.stem
    result = {
        "video_id": video_id,
        "pass1_flipped": [],
        "pass1_uncertain": [],
        "pass2_flipped": [],
        "pass2_correct": [],
        "total_flipped": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "error": None,
    }

    async with semaphore:
        try:
            with open(fpath) as f:
                utterances = json.load(f)

            # Collect reviewable interviewer indices (≥ MIN_WORDS)
            ir_indices = [
                i for i, u in enumerate(utterances)
                if u["speaker"] == "interviewer" and word_count(u["text"]) >= MIN_WORDS
            ]

            if not ir_indices:
                print(f"  {label} {video_id}: skip (no reviewable IR utterances)", flush=True)
                result["skipped"] = True
                return result

            working = copy.deepcopy(utterances)

            # ── Pass 1 ────────────────────────────────────────────────────────
            flip1, uncertain, usage1 = await call_pass1(
                session, ir_indices, working, model, api_key, provider
            )
            result["prompt_tokens"] += usage1.get("prompt_tokens", 0)
            result["completion_tokens"] += usage1.get("completion_tokens", 0)

            if flip1:
                diffs = apply_flips(working, flip1)
                result["pass1_flipped"] = [d["index"] for d in diffs]

            result["pass1_uncertain"] = uncertain

            # ── Pass 2 ────────────────────────────────────────────────────────
            for uidx in uncertain:
                should_flip, usage2 = await call_pass2(
                    session, uidx, working, model, api_key, provider
                )
                result["prompt_tokens"] += usage2.get("prompt_tokens", 0)
                result["completion_tokens"] += usage2.get("completion_tokens", 0)
                if should_flip:
                    apply_flips(working, [uidx])
                    result["pass2_flipped"].append(uidx)
                else:
                    result["pass2_correct"].append(uidx)

            # ── Write back if anything changed ────────────────────────────────
            total_flipped = len(result["pass1_flipped"]) + len(result["pass2_flipped"])
            result["total_flipped"] = total_flipped

            if total_flipped > 0:
                with open(fpath, "w") as f:
                    json.dump(working, f, indent=2)

            status = (
                f"CHANGED ({total_flipped} flipped, {len(uncertain)} uncertain)"
                if total_flipped else
                f"OK ({len(uncertain)} uncertain → kept)"
            )
            print(f"  {label} {video_id}: {status}", flush=True)

        except Exception as e:
            result["error"] = str(e)
            print(f"  {label} {video_id}: ERROR: {e}", flush=True)

    return result


# ── Tracker ───────────────────────────────────────────────────────────────────

def load_tracker(tracker_path: Path) -> dict:
    if tracker_path.is_file():
        with open(tracker_path) as f:
            return json.load(f)
    return {"done": {}, "stats": {}}


def save_tracker(tracker_path: Path, tracker: dict) -> None:
    with open(tracker_path, "w") as f:
        json.dump(tracker, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fix interviewer → interviewee mislabelings")
    p.add_argument(
        "--transcript-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "transcripts_llm"),
    )
    p.add_argument("--limit", type=int, default=None, help="Only process first N files")
    p.add_argument("--model", default="openai/gpt-oss-120b")
    p.add_argument("--provider", default=None)
    p.add_argument("--concurrency", type=int, default=40)
    p.add_argument("--rerun", action="store_true", help="Re-process already-completed files")
    return p.parse_args()


async def async_main():
    args = parse_args()

    api_key = os.environ.get("VOICES_OPENROUTER_KEY")
    if not api_key:
        print("ERROR: VOICES_OPENROUTER_KEY not set.", file=sys.stderr)
        sys.exit(1)

    transcript_dir = Path(args.transcript_dir).resolve()
    tracker_path = transcript_dir / "_fix_interviewer_tracker.json"

    tracker = load_tracker(tracker_path)
    # Only skip files that completed without error
    if not args.rerun:
        done_ids: set[str] = {
            vid for vid, res in tracker["done"].items() if not res.get("error")
        }
    else:
        done_ids = set()

    all_files = sorted(f for f in transcript_dir.glob("*.json") if not f.name.startswith("_"))
    if args.limit:
        all_files = all_files[: args.limit]

    pending = [f for f in all_files if f.stem not in done_ids]
    skipped_count = len(all_files) - len(pending)

    print(f"Transcript dir:  {transcript_dir}")
    print(f"Model:           {args.model}")
    print(f"Provider:        {args.provider or '(default)'}")
    print(f"Concurrency:     {args.concurrency}")
    print(f"Total files:     {len(all_files)}")
    print(f"Already done:    {skipped_count}")
    print(f"To process:      {len(pending)}")
    print(f"Tracker:         {tracker_path.name}")
    print()

    if not pending:
        print("Nothing to do — all files already processed.")
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency)

    total_stats = {
        "processed": 0, "changed": 0, "unchanged": 0, "errors": 0,
        "total_flipped": 0, "prompt_tokens": 0, "completion_tokens": 0,
    }

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for idx, fpath in enumerate(pending):
            label = f"[{idx + 1}/{len(pending)}]"
            tasks.append(
                process_file(session, semaphore, fpath, args.model, api_key, args.provider, label)
            )
        results = await asyncio.gather(*tasks)

    for res in results:
        vid = res["video_id"]
        tracker["done"][vid] = res
        total_stats["processed"] += 1
        total_stats["prompt_tokens"] += res.get("prompt_tokens", 0)
        total_stats["completion_tokens"] += res.get("completion_tokens", 0)
        if res.get("error"):
            total_stats["errors"] += 1
        elif res.get("total_flipped", 0) > 0:
            total_stats["changed"] += 1
            total_stats["total_flipped"] += res["total_flipped"]
        else:
            total_stats["unchanged"] += 1

    tracker["stats"] = total_stats
    save_tracker(tracker_path, tracker)

    print()
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    for k, v in total_stats.items():
        print(f"  {k:>25s}: {v}")
    print(f"\nTracker saved to: {tracker_path}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
