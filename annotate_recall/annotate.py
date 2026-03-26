#!/usr/bin/env python3
"""
Resumable parallel LLM annotation script — sentence-level recall/topic labeling.

60 concurrent API requests across all transcripts and chunks simultaneously.
Each request retries independently. Progress tracked by per-chunk output files.

Usage:
    python annotate.py [options]

Options:
    --chunk-words N      Target words per chunk (default: 800)
    --max-concurrent N   Max parallel API calls (default: 60)
    --limit N            Process at most N transcript files
    --file FILE          Process a single file (e.g. 10.1.json)
    --dry-run            No API calls; write mock outputs (no disk writes)
    --analyze-only       Print chunking statistics and exit
    --combine-only       Combine chunk files into per-transcript JSONs and exit
    --reset-file STEM    Delete output and chunk files for one transcript
    --normalize-outputs  Fix any invalid topic labels in existing outputs

Chunking options (--chunk-words):
    300  → ~61K API calls  (most focused)
    500  → ~38K API calls
    800  → ~24K API calls  (default, good balance)
   1500  → ~13K API calls  (fewer calls, wider context)

Output structure:
    data/llm_annotated_recall_facts/
        chunks/
            {stem}__{chunk_idx:05d}.json   ← one per completed chunk (progress tracking)
        {stem}.json                         ← combined per-transcript output
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
import glob
from pathlib import Path

import aiohttp

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIR = REPO_ROOT / "data" / "transcripts_llm"
OUTPUT_DIR = REPO_ROOT / "data" / "llm_annotated_recall_facts"
CHUNKS_DIR = OUTPUT_DIR / "chunks"
PROMPT_TEMPLATE_FILE = Path(__file__).resolve().parent / "prompt_template.txt"

# ── Model / API ───────────────────────────────────────────────────────────────
MODEL = "openai/gpt-4o-mini"          # override with --model; default is fast+cheap
DEFAULT_MODEL = "openai/gpt-4o-mini"  # user requested openai/gpt-oss-120b (alias below)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_CHUNK_WORDS = 800
DEFAULT_MAX_CONCURRENT = 60
MAX_RETRIES = 5
BASE_RETRY_DELAY = 2.0  # seconds, doubles each attempt

VALID_TOPICS = {
    "Captivity",
    "Daily life (childhood)",
    "Daily life (imprisonment)",
    "Feelings and thoughts",
    "Forced labor",
    "Government",
    "Health",
    "Liberation",
    "Post-conflict",
    "Refugee experiences",
    "Parents",
    "Other",
}


# ── API key ───────────────────────────────────────────────────────────────────
def load_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        env_file = REPO_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    if not key:
        sys.exit("ERROR: OPENROUTER_API_KEY not found in environment or .env")
    return key


def load_prompt_template() -> str:
    return PROMPT_TEMPLATE_FILE.read_text()


# ── Sentence splitting ────────────────────────────────────────────────────────
# Handles transcript disfluencies; keeps non-verbal cues whole.
_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z\[\(])')

def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    # Non-verbal markers like [LAUGHTER] stay as one "sentence"
    parts = _SENT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# ── Chunking ──────────────────────────────────────────────────────────────────
def build_sentence_chunks(
    segments: list[dict], chunk_words: int
) -> list[list[dict]]:
    """
    Split interviewee segments into sentences, then greedily group into
    chunks of ~chunk_words words (never splitting a sentence).

    Returns: list of chunks, each chunk = list of sentence dicts:
        {segment_idx, sentence_idx_in_seg, text, word_count}
    """
    flat: list[dict] = []
    for seg_idx, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        sentences = split_sentences(text) if text else [text]
        if not sentences:
            sentences = [text]
        for s_idx, s in enumerate(sentences):
            flat.append({
                "segment_idx": seg_idx,
                "sentence_idx_in_seg": s_idx,
                "text": s,
                "word_count": len(s.split()),
            })

    if not flat:
        return []

    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_words = 0

    for sent in flat:
        w = sent["word_count"]
        if current_words + w > chunk_words and current:
            chunks.append(current)
            current = [sent]
            current_words = w
        else:
            current.append(sent)
            current_words += w

    if current:
        chunks.append(current)

    return chunks


# ── Chunk file helpers ────────────────────────────────────────────────────────
def chunk_key(stem: str, chunk_idx: int) -> str:
    return f"{stem}__{chunk_idx:05d}"


def chunk_path(stem: str, chunk_idx: int) -> Path:
    return CHUNKS_DIR / f"{chunk_key(stem, chunk_idx)}.json"


def is_chunk_done(stem: str, chunk_idx: int) -> bool:
    p = chunk_path(stem, chunk_idx)
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text())
        return "annotations" in d
    except Exception:
        return False


def load_chunk(stem: str, chunk_idx: int) -> dict | None:
    p = chunk_path(stem, chunk_idx)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ── LLM response parsing ──────────────────────────────────────────────────────
def validate_and_normalize_annotations(
    annotations: list[dict],
) -> list[dict]:
    out = []
    for ann in annotations:
        topics = ann.get("topics", [])
        fixed_topics = []
        for t in topics:
            if t in VALID_TOPICS:
                fixed_topics.append(t)
            else:
                if "Other" not in fixed_topics:
                    fixed_topics.append("Other")
        out.append({
            "idx": ann.get("idx"),
            "recall": bool(ann.get("recall", False)),
            "topics": fixed_topics,
        })
    return out


def parse_llm_response(response_text: str, n_expected: int) -> list[dict]:
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        end = next((i for i, l in enumerate(lines[1:], 1) if l.strip() == "```"), len(lines))
        text = "\n".join(lines[1:end])

    parsed = json.loads(text)

    # Accept both {"sentences": [...]} and a bare list
    if isinstance(parsed, list):
        raw = parsed
    elif "sentences" in parsed:
        raw = parsed["sentences"]
    else:
        # Try first list value
        for v in parsed.values():
            if isinstance(v, list):
                raw = v
                break
        else:
            raise ValueError(f"Cannot find sentence list in response keys: {list(parsed.keys())}")

    annotations = validate_and_normalize_annotations(raw)

    # Warn if indices are missing
    returned_idxs = {a["idx"] for a in annotations}
    missing = [i for i in range(n_expected) if i not in returned_idxs]
    if missing:
        print(f"    [WARN] Response missing {len(missing)} sentence indices: {missing[:10]}...")

    return annotations


# ── Async API call ────────────────────────────────────────────────────────────
async def call_openrouter_async(
    session: aiohttp.ClientSession,
    prompt: str,
    api_key: str,
    model: str,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://voiceover-project",
        "X-Title": "VoiceOver Recall Annotation",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        # 16384 tokens to accommodate reasoning models (gpt-oss-120b uses ~3k reasoning tokens)
        "max_tokens": 16384,
        "response_format": {"type": "json_object"},
    }
    async with session.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=180),
    ) as resp:
        if resp.status == 429:
            retry_after = int(resp.headers.get("Retry-After", "30"))
            raise aiohttp.ClientResponseError(
                resp.request_info, resp.history,
                status=429, message=f"Rate limited, retry after {retry_after}s"
            )
        resp.raise_for_status()
        data = await resp.json()
        return data["choices"][0]["message"]["content"]


async def annotate_chunk_async(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    stem: str,
    chunk_idx: int,
    sentences: list[dict],
    api_key: str,
    prompt_template: str,
    model: str,
    counter: dict,
    dry_run: bool = False,
) -> bool:
    """Annotate a single chunk with independent retry. Returns True on success."""
    if is_chunk_done(stem, chunk_idx):
        return True

    # Build prompt: list of numbered sentences
    numbered = "\n".join(f"[{i}] {s['text']}" for i, s in enumerate(sentences))
    prompt = prompt_template.replace("{{SENTENCES}}", numbered)

    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if dry_run:
                    await asyncio.sleep(0.01)  # simulate work
                    annotations = [
                        {"idx": i, "recall": True, "topics": ["Other"]}
                        for i in range(len(sentences))
                    ]
                    elapsed = 0.01
                else:
                    t0 = time.monotonic()
                    content = await call_openrouter_async(session, prompt, api_key, model)
                    elapsed = time.monotonic() - t0
                    annotations = parse_llm_response(content, len(sentences))
                    # Retry if model skipped too many sentences (< 90% returned)
                    n_returned = len({a["idx"] for a in annotations})
                    if n_returned < max(1, len(sentences) * 0.90):
                        raise ValueError(
                            f"Incomplete response: {n_returned}/{len(sentences)} sentences returned"
                        )

                record = {
                    "stem": stem,
                    "chunk_idx": chunk_idx,
                    "model": model,
                    "elapsed_s": round(elapsed, 1),
                    "sentences_input": sentences,
                    "annotations": annotations,
                }
                if not dry_run:
                    chunk_path(stem, chunk_idx).write_text(
                        json.dumps(record, ensure_ascii=False, indent=2)
                    )

                counter["done"] += 1
                total = counter["total"]
                pct = counter["done"] * 100 // total if total else 0
                print(
                    f"  [{counter['done']}/{total} {pct}%] {stem} chunk {chunk_idx}"
                    f" ({len(sentences)} sents, {elapsed:.1f}s)"
                )
                return True

            except aiohttp.ClientResponseError as e:
                if e.status == 429:
                    delay = 30.0
                    print(f"  [429] {stem}:{chunk_idx} rate limited, waiting {delay}s (attempt {attempt})")
                elif 400 <= e.status < 500:
                    print(f"  [HTTP {e.status}] {stem}:{chunk_idx} non-retryable: {e}")
                    return False
                else:
                    delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                    print(f"  [HTTP {e.status}] {stem}:{chunk_idx} attempt {attempt}/{MAX_RETRIES}, retry in {delay:.0f}s")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(delay)

            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                print(f"  [NET] {stem}:{chunk_idx} {type(e).__name__} attempt {attempt}/{MAX_RETRIES}, retry in {delay:.0f}s")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(delay)

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                print(f"  [PARSE] {stem}:{chunk_idx} {e} attempt {attempt}/{MAX_RETRIES}, retry in {delay:.0f}s")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(delay)

    print(f"  [FAIL] {stem}:{chunk_idx} failed after {MAX_RETRIES} attempts")
    counter["errors"] += 1
    return False


# ── Combine chunk files → per-transcript JSON ─────────────────────────────────
def combine_transcript(stem: str, n_chunks: int, chunk_words: int = DEFAULT_CHUNK_WORDS) -> bool:
    """Read all chunk files for a transcript and write the combined output."""
    all_sentences: list[dict] = []

    for idx in range(n_chunks):
        record = load_chunk(stem, idx)
        if record is None:
            print(f"  [WARN] Missing chunk {idx} for {stem}, skipping combine")
            return False
        sent_inputs = record["sentences_input"]
        annotations = {a["idx"]: a for a in record["annotations"]}
        for i, sent in enumerate(sent_inputs):
            ann = annotations.get(i, {})
            all_sentences.append({
                "segment_idx": sent["segment_idx"],
                "sentence_idx_in_seg": sent["sentence_idx_in_seg"],
                "text": sent["text"],
                "recall": ann.get("recall", False),
                "topics": ann.get("topics", []),
            })

    # Read model from first chunk file if available
    first_chunk = load_chunk(stem, 0)
    model_used = (
        first_chunk.get("model") or "openai/gpt-oss-120b"
        if first_chunk else "openai/gpt-oss-120b"
    )

    output = {
        "transcript_id": stem,
        "source_file": f"{stem}.json",
        "model": model_used,
        "chunk_words_setting": chunk_words,
        "n_chunks": n_chunks,
        "sentences": all_sentences,
    }
    out_path = OUTPUT_DIR / f"{stem}.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    return True


def combine_all(
    stems_with_chunks: list[tuple[str, int]],
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    verbose: bool = True,
) -> int:
    """Combine all completed transcripts. Returns count of combined files."""
    n = 0
    for stem, n_chunks in stems_with_chunks:
        all_done = all(is_chunk_done(stem, i) for i in range(n_chunks))
        if all_done:
            ok = combine_transcript(stem, n_chunks, chunk_words=chunk_words)
            if ok and verbose:
                print(f"  Combined {stem} ({n_chunks} chunks)")
            if ok:
                n += 1
    return n


# ── Chunking analysis ─────────────────────────────────────────────────────────
def print_chunking_analysis(chunk_words: int) -> None:
    files = sorted(glob.glob(str(TRANSCRIPT_DIR / "*.json")))
    word_counts = []
    chunk_counts = []
    sent_counts = []

    for f in files:
        try:
            d = json.loads(Path(f).read_text())
            segs = [s for s in d if s.get("speaker") == "interviewee"]
            if not segs:
                continue
            total_words = sum(len(s.get("words", [])) for s in segs)
            chunks = build_sentence_chunks(segs, chunk_words)
            total_sents = sum(len(c) for c in chunks)
            word_counts.append(total_words)
            chunk_counts.append(len(chunks))
            sent_counts.append(total_sents)
        except Exception:
            pass

    if not word_counts:
        return

    word_counts.sort()
    chunk_counts.sort()
    sent_counts.sort()

    print(f"\n{'='*60}")
    print(f"CHUNKING ANALYSIS: chunk_words={chunk_words}")
    print(f"{'='*60}")
    print(f"Transcripts analyzed: {len(word_counts)}")
    print(f"Words/transcript: min={word_counts[0]}, max={word_counts[-1]}, "
          f"mean={sum(word_counts)//len(word_counts)}, median={word_counts[len(word_counts)//2]}")
    print(f"Chunks/transcript: min={chunk_counts[0]}, max={chunk_counts[-1]}, "
          f"mean={sum(chunk_counts)/len(chunk_counts):.1f}, median={chunk_counts[len(chunk_counts)//2]}")
    print(f"Sentences/transcript: mean={sum(sent_counts)//len(sent_counts)}, median={sent_counts[len(sent_counts)//2]}")
    total_calls = sum(chunk_counts)
    avg_sents_per_chunk = sum(sent_counts) / total_calls if total_calls else 0
    print(f"Total API calls needed: {total_calls:,}")
    print(f"Avg sentences per chunk: {avg_sents_per_chunk:.1f}")
    # Rough time estimates
    for secs_per_call in [5, 15, 30]:
        serial_h = total_calls * secs_per_call / 3600
        parallel_h = total_calls * secs_per_call / 60 / 3600  # 60 concurrent
        print(f"  @{secs_per_call}s/call: serial={serial_h:.0f}h, parallel@60={parallel_h:.1f}h")
    print(f"{'='*60}\n")


# ── Utilities ─────────────────────────────────────────────────────────────────
def normalize_existing_outputs() -> None:
    files = sorted(OUTPUT_DIR.glob("*.json"))
    fixed = 0
    for p in files:
        d = json.loads(p.read_text())
        changed = False
        for sent in d.get("sentences", []):
            bad = [t for t in sent.get("topics", []) if t not in VALID_TOPICS]
            if bad:
                sent["topics"] = [t if t in VALID_TOPICS else "Other" for t in sent["topics"]]
                # Deduplicate
                seen = []
                for t in sent["topics"]:
                    if t not in seen:
                        seen.append(t)
                sent["topics"] = seen
                fixed += len(bad)
                changed = True
        if changed:
            p.write_text(json.dumps(d, ensure_ascii=False, indent=2))
    print(f"Fixed {fixed} invalid topic labels.")


def reset_file(stem: str) -> None:
    # Delete chunk files
    for p in CHUNKS_DIR.glob(f"{stem}__*.json"):
        p.unlink()
        print(f"  Deleted chunk: {p.name}")
    # Delete combined output
    out = OUTPUT_DIR / f"{stem}.json"
    if out.exists():
        out.unlink()
        print(f"  Deleted output: {out.name}")


# ── Main async orchestration ──────────────────────────────────────────────────
async def run_async(args) -> None:
    api_key = load_api_key()
    prompt_template = load_prompt_template()
    model = getattr(args, "model", DEFAULT_MODEL)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    # Gather transcript files
    if args.file:
        files = [TRANSCRIPT_DIR / args.file]
    else:
        files = sorted(TRANSCRIPT_DIR.glob("*.json"))
    if args.limit:
        files = files[:args.limit]

    # Build full work queue
    work_items: list[tuple[str, int, list[dict]]] = []  # (stem, chunk_idx, sentences)
    stems_with_chunks: list[tuple[str, int]] = []
    skipped = 0
    already_done_transcripts = 0

    for f in files:
        try:
            transcript = json.loads(f.read_text())
        except Exception as e:
            print(f"  [ERROR] Cannot read {f.name}: {e}")
            continue

        segs = [s for s in transcript if s.get("speaker") == "interviewee"]
        if not segs:
            skipped += 1
            continue

        chunks = build_sentence_chunks(segs, args.chunk_words)
        if not chunks:
            skipped += 1
            continue

        stem = f.stem
        stems_with_chunks.append((stem, len(chunks)))

        # Check if transcript is fully done
        if all(is_chunk_done(stem, i) for i in range(len(chunks))):
            already_done_transcripts += 1
            # Still include in combine pass
            continue

        for chunk_idx, sentences in enumerate(chunks):
            if not is_chunk_done(stem, chunk_idx):
                work_items.append((stem, chunk_idx, sentences))

    total_pending = len(work_items)
    print(f"Transcripts: {already_done_transcripts} complete, "
          f"{len(stems_with_chunks) - already_done_transcripts} in-progress/pending, "
          f"{skipped} skipped (no interviewee content)")
    print(f"Pending chunks: {total_pending}")
    print(f"Model: {model} | chunk-words: {args.chunk_words} | max-concurrent: {args.max_concurrent}")
    if args.dry_run:
        print("MODE: DRY RUN (no API calls, no disk writes)")
    print()

    if total_pending == 0:
        print("Nothing to do.")
    else:
        counter = {"done": 0, "errors": 0, "total": total_pending}
        semaphore = asyncio.Semaphore(args.max_concurrent)

        connector = aiohttp.TCPConnector(limit=args.max_concurrent + 10)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                annotate_chunk_async(
                    session, semaphore, stem, chunk_idx, sentences,
                    api_key, prompt_template, model, counter, dry_run=args.dry_run,
                )
                for stem, chunk_idx, sentences in work_items
            ]
            t0 = time.monotonic()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.monotonic() - t0

        successes = sum(1 for r in results if r is True)
        failures = sum(1 for r in results if r is not True)
        print(f"\nChunks: {successes} succeeded, {failures} failed in {elapsed:.0f}s")
        if total_pending > 0:
            print(f"Throughput: {successes / elapsed * 60:.0f} chunks/min")

    # Combine completed transcripts
    print("\nCombining completed transcripts...")
    n_combined = combine_all(
        stems_with_chunks,
        chunk_words=args.chunk_words,
        verbose=args.verbose if hasattr(args, "verbose") else True,
    )
    print(f"Combined {n_combined} transcript(s) → {OUTPUT_DIR}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel recall/topic annotation for interview transcripts"
    )
    parser.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b",
                        help="OpenRouter model ID (default: openai/gpt-oss-120b)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--combine-only", action="store_true",
                        help="Only run the combine step (no new API calls)")
    parser.add_argument("--normalize-outputs", action="store_true")
    parser.add_argument("--reset-file", type=str, default=None, metavar="STEM")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    args.verbose = args.verbose and not args.quiet

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    if args.normalize_outputs:
        normalize_existing_outputs()
        return

    if args.reset_file:
        reset_file(args.reset_file)
        return

    if args.analyze_only:
        for cw in [300, 500, 800, 1500]:
            print_chunking_analysis(cw)
        return

    if args.combine_only:
        # Rebuild stems_with_chunks from existing chunk files
        stems: dict[str, int] = {}
        for p in CHUNKS_DIR.glob("*.json"):
            parts = p.stem.rsplit("__", 1)
            if len(parts) == 2:
                stem, idx_str = parts
                stems[stem] = max(stems.get(stem, 0), int(idx_str) + 1)
        n = combine_all(list(stems.items()), chunk_words=args.chunk_words, verbose=True)
        print(f"Combined {n} transcript(s).")
        return

    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()
