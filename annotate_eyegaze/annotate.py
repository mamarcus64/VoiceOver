#!/usr/bin/env python3
"""
Resumable parallel LLM annotation for transcripts that have associated eyegaze data.

Uses up to two OpenRouter accounts (OPENROUTER_API_KEY + SECOND_OPENROUTER_API_KEY),
each capped at KEY_CONCURRENCY (40) concurrent requests, for 80 total.

Usage:
    python annotate_eyegaze/annotate.py [options]

Options:
    --chunk-words N      Target words per chunk (default: 300)
    --limit N            Process at most N transcript files (not chunks)
    --file FILE          Process a single file (e.g. 10.1.json)
    --model MODEL        OpenRouter model string (default: openai/gpt-oss-120b)
    --cerebras           Use Cerebras API instead of OpenRouter (requires CEREBRAS_API_KEY)
    --dry-run            No API calls; write mock outputs
    --combine-only       Combine finished chunk files and exit
    --reset-file STEM    Delete output and chunk files for one transcript
    --time-logs          Suppress per-chunk logs; print one progress line per minute

Output:
    data/llm_annotated_eyegaze/
        chunks/
            {stem}__{chunk_idx:05d}.json
        {stem}.json
"""

import argparse
import ast
import asyncio
import json
import os
import random
import re
import sys
import time
import glob
from pathlib import Path

import aiohttp

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT       = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIR  = REPO_ROOT / "data" / "transcripts_llm"
EYEGAZE_VAD_DIR = REPO_ROOT / "data" / "eyegaze_vad"
OUTPUT_DIR      = REPO_ROOT / "data" / "llm_annotated_eyegaze"
CHUNKS_DIR      = OUTPUT_DIR / "chunks"
PROMPT_TEMPLATE_FILE = Path(__file__).resolve().parent / "prompt_template.txt"

# ── Model / API ───────────────────────────────────────────────────────────────
DEFAULT_MODEL           = "openai/gpt-oss-120b"
CEREBRAS_MODEL          = "gpt-oss-120b"          # no namespace prefix on Cerebras
OPENROUTER_URL          = "https://openrouter.ai/api/v1/chat/completions"
CEREBRAS_URL            = "https://api.cerebras.ai/v1/chat/completions"
DEFAULT_CHUNK_WORDS     = 300
KEY_CONCURRENCY         = 40   # max simultaneous requests per OpenRouter key
CEREBRAS_CONCURRENCY    = 25   # rate limiter is the real throttle; keep slots above rate*avg_latency
MAX_RETRIES             = 5
BASE_RETRY_DELAY        = 2.0
API_TIMEOUT_S           = 900  # seconds; gpt-oss-120b reasoning can be very slow
CEREBRAS_TIMEOUT_S      = 120  # Cerebras is ~10x faster

VALID_MEMORY_TYPES = {"internal", "external"}
VALID_CONTENT_DOMAINS = {"pre-war", "wartime", "liberation", "post-war", "present-day", "other"}
VALID_TEMPORAL_SYNTAX = {"strict_past", "habitual_past", "present_narration", "present_reliving", "present_reflection"}
VALID_NARRATIVE_STRUCTURES = {"orientation", "complicating_action", "evaluation", "resolution", "other"}
VALID_NARRATIVE_VALENCE = {"positive", "negative", "neutral", "mixed"}
VALID_PRESENT_DAY_VALENCE = {"positive", "negative", "neutral", "mixed"}
VALID_TOPICS = {
    "Captivity", "Daily life (childhood)", "Daily life (imprisonment)",
    "Feelings and thoughts", "Forced labor", "Government", "Health",
    "Liberation", "Post-conflict", "Refugee experiences", "Parents",
}


# ── API keys ──────────────────────────────────────────────────────────────────
def _read_env_file() -> dict[str, str]:
    env_file = REPO_ROOT / ".env"
    result: dict[str, str] = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                result[k.strip()] = v.strip()
    return result


def load_openrouter_keys() -> list[str]:
    """Return a list of available OpenRouter API keys (1 or 2)."""
    env = _read_env_file()
    keys = []
    for name in ("OPENROUTER_API_KEY", "SECOND_OPENROUTER_API_KEY"):
        k = os.environ.get(name, "") or env.get(name, "")
        if k:
            keys.append(k)
    if not keys:
        sys.exit("ERROR: No OpenRouter API keys found in environment or .env "
                 "(expected OPENROUTER_API_KEY and/or SECOND_OPENROUTER_API_KEY)")
    return keys


def load_cerebras_key() -> str:
    env = _read_env_file()
    k = os.environ.get("CEREBRAS_API_KEY", "") or env.get("CEREBRAS_API_KEY", "")
    if not k:
        sys.exit("ERROR: CEREBRAS_API_KEY not found in environment or .env")
    return k


def load_prompt_template() -> str:
    return PROMPT_TEMPLATE_FILE.read_text()


# ── Sentence splitting ────────────────────────────────────────────────────────
_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z\[\(])')

def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# ── Chunking ──────────────────────────────────────────────────────────────────
def build_sentence_chunks(segments: list[dict], chunk_words: int) -> list[list[dict]]:
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
def chunk_path(stem: str, chunk_idx: int) -> Path:
    return CHUNKS_DIR / f"{stem}__{chunk_idx:05d}.json"


def is_chunk_done(stem: str, chunk_idx: int) -> bool:
    p = chunk_path(stem, chunk_idx)
    if not p.exists():
        return False
    try:
        return "annotations" in json.loads(p.read_text())
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
def validate_and_normalize(annotations: list[dict]) -> list[dict]:
    out = []
    for ann in annotations:
        mt = ann.get("memory_type", "external")
        if mt not in VALID_MEMORY_TYPES:
            mt = "external"

        cd = ann.get("content_domain", "other")
        if cd not in VALID_CONTENT_DOMAINS:
            cd = "other"

        ts = ann.get("temporal_syntax", "strict_past")
        if ts not in VALID_TEMPORAL_SYNTAX:
            ts = "strict_past"

        ns = ann.get("narrative_structure", "other")
        if ns not in VALID_NARRATIVE_STRUCTURES:
            ns = "other"

        nv = ann.get("narrative_valence", "neutral")
        if nv not in VALID_NARRATIVE_VALENCE:
            nv = "neutral"

        pdv = ann.get("present_day_valence", "neutral")
        if pdv not in VALID_PRESENT_DAY_VALENCE:
            pdv = "neutral"

        topics = [t for t in ann.get("topics", []) if t in VALID_TOPICS]

        out.append({
            "idx":                 ann.get("idx"),
            "memory_type":         mt,
            "content_domain":      cd,
            "temporal_syntax":     ts,
            "narrative_structure": ns,
            "narrative_valence":   nv,
            "present_day_valence": pdv,
            "topics":              topics,
        })
    return out


def parse_llm_response(response_text: str, n_expected: int) -> list[dict]:
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        end = next((i for i, l in enumerate(lines[1:], 1) if l.strip() == "```"), len(lines))
        text = "\n".join(lines[1:end])

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)

    if isinstance(parsed, list):
        raw = parsed
    elif "sentences" in parsed:
        raw = parsed["sentences"]
    else:
        for v in parsed.values():
            if isinstance(v, list):
                raw = v
                break
        else:
            raise ValueError(f"Cannot find sentence list in response: {list(parsed.keys())}")

    annotations = validate_and_normalize(raw)

    returned_idxs = {a["idx"] for a in annotations}
    missing = [i for i in range(n_expected) if i not in returned_idxs]
    if missing:
        print(f"    [WARN] Response missing {len(missing)} indices: {missing[:10]}")

    return annotations


# ── Token-bucket rate limiter ─────────────────────────────────────────────────
class RateLimiter:
    """Asyncio token-bucket rate limiter.

    Limits to `rate` requests per second by making callers wait until a
    token is available.  Safe for concurrent use (protected by an asyncio Lock).
    """

    def __init__(self, rate: float) -> None:
        self._rate  = rate          # tokens added per second
        self._tokens = rate         # start full
        self._last  = time.monotonic()
        self._lock  = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            self._tokens = min(self._rate, self._tokens + (now - self._last) * self._rate)
            self._last = now
            if self._tokens < 1:
                wait = (1 - self._tokens) / self._rate
                await asyncio.sleep(wait)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


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
        "X-Title": "VoiceOver Eyegaze Annotation",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 32768,
        "response_format": {"type": "json_object"},
    }
    async with session.post(OPENROUTER_URL, headers=headers, json=payload,
                            timeout=aiohttp.ClientTimeout(total=API_TIMEOUT_S)) as resp:
        if resp.status == 429:
            retry_after = int(resp.headers.get("Retry-After", "30"))
            raise aiohttp.ClientResponseError(
                resp.request_info, resp.history,
                status=429, message=f"Rate limited, retry after {retry_after}s"
            )
        resp.raise_for_status()
        data = await resp.json()
        return data["choices"][0]["message"]["content"]


async def call_cerebras_async(
    session: aiohttp.ClientSession,
    prompt: str,
    api_key: str,
    rate_limiter: RateLimiter,
) -> str:
    await rate_limiter.acquire()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CEREBRAS_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 8192,   # reasoning (hidden) + JSON output; Cerebras reserves this upfront vs TPM quota
        "response_format": {"type": "json_object"},
    }
    async with session.post(CEREBRAS_URL, headers=headers, json=payload,
                            timeout=aiohttp.ClientTimeout(total=CEREBRAS_TIMEOUT_S)) as resp:
        if resp.status == 429:
            retry_after = int(resp.headers.get("Retry-After", "30"))
            raise aiohttp.ClientResponseError(
                resp.request_info, resp.history,
                status=429, message=f"Rate limited, retry after {retry_after}s"
            )
        resp.raise_for_status()
        data = await resp.json()
        return data["choices"][0]["message"]["content"]


# ── Progress logger (used with --time-logs) ───────────────────────────────────
async def progress_logger(counter: dict, t_start: float, stop_event: asyncio.Event) -> None:
    while True:
        try:
            # Wake up after 60s OR immediately when stop_event is set
            await asyncio.wait_for(stop_event.wait(), timeout=60)
            break  # stop_event fired — we're done
        except asyncio.TimeoutError:
            pass   # 60s elapsed normally — print progress and loop

        elapsed = time.monotonic() - t_start
        done    = counter["done"]
        total   = counter["total"]
        errors  = counter["errors"]
        rate    = done / elapsed if elapsed > 1 else 0
        pct     = done / total * 100 if total > 0 else 0
        remaining = (total - done) / rate if rate > 0 else float("inf")
        rem_str = f"{remaining / 60:.0f}min" if remaining != float("inf") else "?"
        print(
            f"[+{elapsed / 60:.1f}min] {done}/{total} ({pct:.1f}%) | "
            f"{rate * 60:.0f} chunks/min | {errors} errors | ~{rem_str} remaining",
            flush=True,
        )


# ── Per-chunk annotation ──────────────────────────────────────────────────────
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
    stem_n_chunks: int,
    chunk_words: int,
    combine_locks: dict,
    dry_run: bool = False,
    prev_sentence: str | None = None,
    verbose: bool = True,
    use_cerebras: bool = False,
    cerebras_rate_limiter: RateLimiter | None = None,
) -> bool:
    if is_chunk_done(stem, chunk_idx):
        return True

    numbered = "\n".join(f"[{i}] {s['text']}" for i, s in enumerate(sentences))
    if prev_sentence:
        sentences_block = f"[context only, do not label] {prev_sentence}\n\n{numbered}"
    else:
        sentences_block = numbered
    prompt = prompt_template.replace("{{SENTENCES}}", sentences_block)

    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if dry_run:
                    await asyncio.sleep(0.01)
                    annotations = [
                        {"idx": i, "memory_type": "internal", "content_domain": "wartime",
                         "temporal_syntax": "strict_past", "narrative_structure": "complicating_action",
                         "narrative_valence": "neutral", "present_day_valence": "neutral",
                         "topics": []}
                        for i in range(len(sentences))
                    ]
                    elapsed = 0.01
                else:
                    t0 = time.monotonic()
                    if use_cerebras:
                        content = await call_cerebras_async(
                            session, prompt, api_key, cerebras_rate_limiter
                        )
                    else:
                        content = await call_openrouter_async(session, prompt, api_key, model)
                    elapsed = time.monotonic() - t0
                    annotations = parse_llm_response(content, len(sentences))
                    n_returned = len({a["idx"] for a in annotations})
                    if n_returned < max(1, len(sentences) * 0.90):
                        raise ValueError(
                            f"Incomplete response: {n_returned}/{len(sentences)} returned"
                        )

                record = {
                    "stem": stem, "chunk_idx": chunk_idx, "model": model,
                    "elapsed_s": round(elapsed, 1),
                    "sentences_input": sentences,
                    "annotations": annotations,
                }
                if not dry_run:
                    chunk_path(stem, chunk_idx).write_text(
                        json.dumps(record, ensure_ascii=False, indent=2)
                    )

                counter["done"] += 1
                if verbose:
                    total = counter["total"]
                    pct = counter["done"] * 100 // total if total else 0
                    print(f"  [{counter['done']}/{total} {pct}%] {stem} chunk {chunk_idx}"
                          f" ({len(sentences)} sents, {elapsed:.1f}s)")

                if not dry_run and all(is_chunk_done(stem, i) for i in range(stem_n_chunks)):
                    lock = combine_locks.setdefault(stem, asyncio.Lock())
                    async with lock:
                        out_path = OUTPUT_DIR / f"{stem}.json"
                        if not out_path.exists():
                            ok = combine_transcript(stem, stem_n_chunks, chunk_words)
                            if ok and verbose:
                                print(f"  [COMBINED] {stem} ({stem_n_chunks} chunks)")

                return True

            except aiohttp.ClientResponseError as e:
                if e.status == 429:
                    # Honour Retry-After if present, plus random jitter so all
                    # simultaneous 429s don't retry at exactly the same moment.
                    m = re.search(r"retry after (\d+)s", str(e), re.IGNORECASE)
                    base = float(m.group(1)) if m else 5.0
                    delay = base + random.uniform(0, 20)
                    print(f"  [429] {stem}:{chunk_idx} rate limited, waiting {delay:.0f}s (attempt {attempt})")
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


# ── Combine chunks → per-transcript JSON ──────────────────────────────────────
def combine_transcript(stem: str, n_chunks: int, chunk_words: int = DEFAULT_CHUNK_WORDS) -> bool:
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
                "segment_idx":           sent["segment_idx"],
                "sentence_idx_in_seg":   sent["sentence_idx_in_seg"],
                "text":                  sent["text"],
                "memory_type":           ann.get("memory_type", "external"),
                "content_domain":        ann.get("content_domain", "other"),
                "temporal_syntax":       ann.get("temporal_syntax", "strict_past"),
                "narrative_structure":  ann.get("narrative_structure", "other"),
                "narrative_valence":    ann.get("narrative_valence", "neutral"),
                "present_day_valence":  ann.get("present_day_valence", "neutral"),
                "topics":               ann.get("topics", []),
            })

    first_chunk = load_chunk(stem, 0)
    model_used = first_chunk.get("model", DEFAULT_MODEL) if first_chunk else DEFAULT_MODEL

    output = {
        "transcript_id":      stem,
        "source_file":        f"{stem}.json",
        "model":              model_used,
        "chunk_words_setting": chunk_words,
        "n_chunks":           n_chunks,
        "sentences":          all_sentences,
    }
    (OUTPUT_DIR / f"{stem}.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))
    return True


def combine_all(stems_with_chunks: list[tuple[str, int]],
                chunk_words: int = DEFAULT_CHUNK_WORDS,
                skip_existing: bool = False) -> int:
    n = 0
    for stem, n_chunks in stems_with_chunks:
        if skip_existing and (OUTPUT_DIR / f"{stem}.json").exists():
            continue
        if all(is_chunk_done(stem, i) for i in range(n_chunks)):
            ok = combine_transcript(stem, n_chunks, chunk_words=chunk_words)
            if ok:
                print(f"  Combined {stem} ({n_chunks} chunks)")
                n += 1
    return n


# ── Utilities ─────────────────────────────────────────────────────────────────
def reset_file(stem: str) -> None:
    for p in CHUNKS_DIR.glob(f"{stem}__*.json"):
        p.unlink()
        print(f"  Deleted chunk: {p.name}")
    out = OUTPUT_DIR / f"{stem}.json"
    if out.exists():
        out.unlink()
        print(f"  Deleted output: {out.name}")


def get_eyegaze_stems() -> set[str]:
    """Return stems that have an eyegaze_vad CSV file."""
    return {p.stem for p in EYEGAZE_VAD_DIR.glob("*.csv")}


# ── Main async orchestration ──────────────────────────────────────────────────
async def run_async(args) -> None:
    use_cerebras = getattr(args, "cerebras", False)
    prompt_template = load_prompt_template()
    model = getattr(args, "model", DEFAULT_MODEL)
    verbose = not args.time_logs

    if use_cerebras:
        cerebras_key = load_cerebras_key()
        key_entries: list[tuple[str, asyncio.Semaphore]] = [
            (cerebras_key, asyncio.Semaphore(CEREBRAS_CONCURRENCY))
        ]
        total_concurrency = CEREBRAS_CONCURRENCY
        # Cerebras reserves (input + max_tokens) against TPM quota before each request.
        # 1M TPM / 10,692 tokens/req / 60s = 1.56 req/s theoretical max.
        # Use 1.1 req/s (70% utilization) to leave comfortable headroom.
        cerebras_limiter: RateLimiter | None = RateLimiter(rate=1.05)
        print(f"Provider: Cerebras ({total_concurrency} concurrent slots, ≤1.05 req/s ~66 chunks/min)")
    else:
        or_keys = load_openrouter_keys()
        key_entries = [
            (k, asyncio.Semaphore(KEY_CONCURRENCY)) for k in or_keys
        ]
        total_concurrency = len(key_entries) * KEY_CONCURRENCY
        cerebras_limiter = None
        print(f"Provider: OpenRouter — {len(or_keys)} key(s) "
              f"({total_concurrency} total concurrent slots)")

    print(f"Model: {CEREBRAS_MODEL if use_cerebras else model} | chunk-words: {args.chunk_words}")
    if args.dry_run:
        print("MODE: DRY RUN")
    if args.time_logs:
        print("Logging: one progress line per minute")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    eyegaze_stems = get_eyegaze_stems()

    if args.file:
        files = [TRANSCRIPT_DIR / args.file]
    else:
        all_files = sorted(TRANSCRIPT_DIR.glob("*.json"))
        files = [f for f in all_files if f.stem in eyegaze_stems]
    if args.limit:
        files = files[:args.limit]

    work_items: list[tuple[str, int, list[dict], str | None]] = []
    stems_with_chunks: list[tuple[str, int]] = []
    skipped = 0
    already_done = 0

    for f in files:
        if f.stem not in eyegaze_stems:
            skipped += 1
            continue
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

        # Skip if the combined output already exists (safe even if chunk size changed)
        if (OUTPUT_DIR / f"{stem}.json").exists():
            already_done += 1
            continue

        if all(is_chunk_done(stem, i) for i in range(len(chunks))):
            already_done += 1
            continue

        for chunk_idx, sentences in enumerate(chunks):
            if not is_chunk_done(stem, chunk_idx):
                prev_sent = chunks[chunk_idx - 1][-1]["text"] if chunk_idx > 0 else None
                work_items.append((stem, chunk_idx, sentences, prev_sent))

    total_pending = len(work_items)
    print(f"\nEyegaze transcripts: {already_done} complete, "
          f"{len(stems_with_chunks) - already_done} in-progress/pending, "
          f"{skipped} skipped")
    print(f"Pending chunks: {total_pending}\n")

    if total_pending == 0:
        print("Nothing to do.")
    else:
        counter = {"done": 0, "errors": 0, "total": total_pending}
        combine_locks: dict = {}
        stem_chunks_map = dict(stems_with_chunks)

        connector = aiohttp.TCPConnector(limit=total_concurrency + 10)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Distribute chunks round-robin across key slots
            chunk_tasks = [
                annotate_chunk_async(
                    session,
                    semaphore=key_entries[i % len(key_entries)][1],
                    stem=stem,
                    chunk_idx=chunk_idx,
                    sentences=sentences,
                    api_key=key_entries[i % len(key_entries)][0],
                    prompt_template=prompt_template,
                    model=model,
                    counter=counter,
                    stem_n_chunks=stem_chunks_map[stem],
                    chunk_words=args.chunk_words,
                    combine_locks=combine_locks,
                    dry_run=args.dry_run,
                    prev_sentence=prev_sent,
                    verbose=verbose,
                    use_cerebras=use_cerebras,
                    cerebras_rate_limiter=cerebras_limiter,
                )
                for i, (stem, chunk_idx, sentences, prev_sent) in enumerate(work_items)
            ]

            t0 = time.monotonic()

            if args.time_logs:
                stop_event = asyncio.Event()
                logger = asyncio.create_task(progress_logger(counter, t0, stop_event))
                results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
                stop_event.set()
                await logger
            else:
                results = await asyncio.gather(*chunk_tasks, return_exceptions=True)

            elapsed = time.monotonic() - t0

        successes = sum(1 for r in results if r is True)
        failures  = sum(1 for r in results if r is not True)
        print(f"\nChunks: {successes} succeeded, {failures} failed in {elapsed:.0f}s")
        if successes > 0:
            print(f"Throughput: {successes / elapsed * 60:.0f} chunks/min")

    print("\nChecking for uncombined transcripts...")
    n = combine_all(stems_with_chunks, chunk_words=args.chunk_words, skip_existing=True)
    if n:
        print(f"Combined {n} additional transcript(s).")
    else:
        print("All transcripts already combined.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel eyegaze-transcript annotation"
    )
    parser.add_argument("--chunk-words",  type=int, default=DEFAULT_CHUNK_WORDS)
    parser.add_argument("--limit",        type=int, default=None)
    parser.add_argument("--file",         type=str, default=None)
    parser.add_argument("--model",        type=str, default=DEFAULT_MODEL)
    parser.add_argument("--cerebras",     action="store_true",
                        help="Use Cerebras instead of OpenRouter (requires CEREBRAS_API_KEY)")
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--combine-only", action="store_true")
    parser.add_argument("--reset-file",   type=str, default=None, metavar="STEM")
    parser.add_argument("--time-logs",    action="store_true",
                        help="Suppress per-chunk logs; print one progress line per minute")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    if args.reset_file:
        reset_file(args.reset_file)
        return

    if args.combine_only:
        stems: dict[str, int] = {}
        for p in CHUNKS_DIR.glob("*.json"):
            parts = p.stem.rsplit("__", 1)
            if len(parts) == 2:
                stem, idx_str = parts
                stems[stem] = max(stems.get(stem, 0), int(idx_str) + 1)
        n = combine_all(list(stems.items()), chunk_words=args.chunk_words)
        print(f"Combined {n} transcript(s).")
        return

    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()
