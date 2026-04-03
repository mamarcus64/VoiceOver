"""
Enrich llm_annotated_eyegaze JSON files with first_word_ms and last_word_ms
per sentence, derived from word-level timings in transcripts_llm.

segment_idx maps to the Nth interviewee-only segment in the transcript.
The transcript ms values are used as-is (transcript time base).
"""

import json, os, re, unicodedata
from pathlib import Path
from multiprocessing import Pool, cpu_count

EYEGAZE_DIR  = Path("/Users/marcus/Desktop/usc/VoiceOver/data/llm_annotated_eyegaze")
TRANSCRIPT_DIR = Path("/Users/marcus/Desktop/usc/VoiceOver/data/transcripts_llm")
OFFSETS_FILE = Path("/Users/marcus/Desktop/usc/VoiceOver/data/transcript_offsets.json")

offsets = json.loads(OFFSETS_FILE.read_text())


# ── Text normalisation ────────────────────────────────────────────────────────
_PUNCT = re.compile(r"[^\w']", re.UNICODE)  # keep word-chars and apostrophes

def norm(text: str) -> list[str]:
    """Lowercase, strip leading/trailing punctuation, split on whitespace."""
    tokens = []
    for tok in text.split():
        t = _PUNCT.sub("", tok.lower()).strip("'")
        if t:
            tokens.append(t)
    return tokens


def flatten_words(words: list[dict]) -> list[tuple[str, int]]:
    """
    Expand transcript word entries (which can be multi-word tokens) into a flat
    list of (normalised_token, ms).  All sub-tokens of a multi-word entry share
    the same ms timestamp.
    """
    flat = []
    for w in words:
        ms  = w["ms"]
        for tok in norm(w["text"]):
            flat.append((tok, ms))
    return flat


# ── Sentence → word-span matcher ─────────────────────────────────────────────
def find_span(sent_tokens: list[str],
              flat_words: list[tuple[str, int]],
              start_hint: int = 0) -> tuple[int | None, int | None, int]:
    """
    Find sent_tokens as a contiguous sub-sequence in flat_words, starting the
    search at start_hint.  Returns (first_ms, last_ms, next_search_start).
    If not found, returns (None, None, start_hint).
    """
    n = len(sent_tokens)
    if n == 0:
        return None, None, start_hint

    for i in range(start_hint, len(flat_words) - n + 1):
        if all(flat_words[i + j][0] == sent_tokens[j] for j in range(n)):
            first_ms = flat_words[i][1]
            last_ms  = flat_words[i + n - 1][1]
            return first_ms, last_ms, i + n  # advance past this match
    return None, None, start_hint            # no match found


# ── Per-file processing ───────────────────────────────────────────────────────
def process_file(fname: str) -> tuple[str, int, int]:
    """Return (fname, sentences_matched, sentences_total)."""
    eyegaze_path   = EYEGAZE_DIR / fname
    transcript_path = TRANSCRIPT_DIR / fname

    if not transcript_path.exists():
        return fname, 0, 0

    data = json.loads(eyegaze_path.read_text())
    tr   = json.loads(transcript_path.read_text())

    offset = offsets.get(data["transcript_id"], 0)

    # Build interviewee-segment index
    interviewee_segs = [s for s in tr if s["speaker"] == "interviewee"]

    # Pre-flatten words per segment
    flat_per_seg: dict[int, list[tuple[str, int]]] = {}
    for seg_idx, seg in enumerate(interviewee_segs):
        flat_per_seg[seg_idx] = flatten_words(seg.get("words", []))

    # Process sentences in order; keep a per-segment search cursor
    cursors: dict[int, int] = {}
    matched = 0

    for sent in data["sentences"]:
        seg_idx = sent["segment_idx"]
        flat = flat_per_seg.get(seg_idx, [])
        if not flat:
            sent["first_word_ms"] = None
            sent["last_word_ms"]  = None
            continue

        cursor = cursors.get(seg_idx, 0)
        sent_toks = norm(sent["text"])

        first_ms, last_ms, next_cursor = find_span(sent_toks, flat, cursor)

        # If exact match failed, retry from the beginning (handles rare reorder)
        if first_ms is None and cursor > 0:
            first_ms, last_ms, next_cursor = find_span(sent_toks, flat, 0)

        if first_ms is not None:
            sent["first_word_ms"] = first_ms + offset
            sent["last_word_ms"]  = last_ms  + offset
            cursors[seg_idx]      = next_cursor
            matched += 1
        else:
            sent["first_word_ms"] = None
            sent["last_word_ms"]  = None

    eyegaze_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return fname, matched, len(data["sentences"])


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    files = sorted(f for f in os.listdir(EYEGAZE_DIR) if f.endswith(".json"))
    print(f"Processing {len(files)} files with {cpu_count()} workers...", flush=True)

    total_matched = 0
    total_sents   = 0
    failures      = []

    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        for i, (fname, matched, total) in enumerate(
                pool.imap_unordered(process_file, files, chunksize=20)):
            total_matched += matched
            total_sents   += total
            if total > 0 and matched < total * 0.5:
                failures.append((fname, matched, total))
            if (i + 1) % 500 == 0:
                pct = 100 * total_matched / max(total_sents, 1)
                print(f"  {i+1}/{len(files)} files | "
                      f"{total_matched:,}/{total_sents:,} sentences matched ({pct:.1f}%)",
                      flush=True)

    pct = 100 * total_matched / max(total_sents, 1)
    print(f"\nDone: {total_matched:,}/{total_sents:,} sentences timestamped ({pct:.1f}%)")
    if failures:
        print(f"\nLow-match files ({len(failures)}):")
        for f, m, t in failures[:20]:
            print(f"  {f}: {m}/{t} ({100*m/t:.0f}%)")
