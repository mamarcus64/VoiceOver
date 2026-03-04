#!/usr/bin/env python3
"""Standardize XML transcript files into structured JSON.

Reads ~5130 XML transcripts (one per video segment), parses speaker-tagged
paragraphs with word-level timestamps, and writes one JSON file per video
containing an array of utterance objects with speaker role, tag, text,
timestamps, and per-word timing.
"""

import os
import re
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

TRANSCRIPT_DIR = "/home/mjma/voices/test_data/transcripts"
OUTPUT_DIR = "/home/mjma/voices/VoiceOver/data/transcripts"
OFFSETS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "transcript_offsets.json")

INTERVIEWER_TAGS = {
    'CREW', 'INT', 'INT 1', 'INT 2', 'INT1', 'INT2', 'UNIDENTIFIED SPEAKER'
}

SPEAKER_TAG_RE = re.compile(
    r'^\s*([A-Za-z][A-Za-z0-9 ._-]{0,50}?)\s*:\s*(.*)\Z', re.DOTALL
)

# Matches text that is entirely non-verbal bracketed markers (not uncertain transcription [? ... ?])
NON_VERBAL_RE = re.compile(r'^\s*(\[(?!\?\s)[^\]]*\]\s*)+$')

CUE_START = re.compile(
    r"^\s*(i'd like to|let's|can you|could you|would you|please|tell me|talk about|"
    r"describe|walk me through|so[, ]|okay[, ]|alright[, ])",
    re.IGNORECASE,
)
PAST_NARR = re.compile(
    r"\b(was|were|did|had|went|saw|lived|worked|took|came|told|started|began|stayed)\b",
    re.IGNORECASE,
)


def _clean_text(text):
    """Remove bracketed content (for classification heuristics only)."""
    return re.sub(r"\[.*?\]", "", text).strip()


def _classify_untagged_paragraph(text, prev_first_role, next_first_role):
    """Classify an untagged paragraph as interviewer or interviewee.

    Uses the same priority rules and weighted-vote fallback as the original
    extract_transcript_features.py implementation.
    """
    t = (text or "").strip()
    tok_len = len(t.split())
    has_q = "?" in t
    cue_start = bool(CUE_START.search(t))
    past_narr = bool(PAST_NARR.search(t))

    # Priority rules
    if prev_first_role == 'interviewer' and next_first_role == 'interviewee':
        return 'interviewer'
    if has_q:
        return 'interviewer'
    if cue_start:
        return 'interviewer'
    if prev_first_role == 'interviewee' and next_first_role == 'interviewer':
        return 'interviewee'

    # Weighted vote fallback
    score_i, score_e = 0, 0
    if next_first_role == 'interviewer':
        score_i += 1
    if prev_first_role == 'interviewer':
        score_i += 2
    if cue_start:
        score_i += 2
    if has_q:
        score_i += 3
    if tok_len <= 8 and re.match(
        r"^\s*(i\b|we\b|let's\b|okay\b|alright\b|so\b)", t, re.IGNORECASE
    ):
        score_i += 1
    if prev_first_role == 'interviewee':
        score_e += 2
    if next_first_role == 'interviewee':
        score_e += 1
    if tok_len >= 12:
        score_e += 1
    if past_narr:
        score_e += 1

    if score_i > score_e:
        return 'interviewer'
    return 'interviewee'


def _parse_span_tag(text):
    """If text matches 'TAG: rest', return (role, tag, after_text); else three Nones."""
    m = SPEAKER_TAG_RE.match((text or "").strip())
    if not m:
        return None, None, None
    tag = m.group(1).strip()
    after = m.group(2).strip()
    role = 'interviewer' if tag.upper() in INTERVIEWER_TAGS else 'interviewee'
    return role, tag, after


def _first_span_role(p):
    """Return the speaker role from the first span's tag, or None."""
    spans = list(p.findall('span'))
    if not spans:
        return None
    role, _, _ = _parse_span_tag(spans[0].text or "")
    return role


def _paragraph_clean_text(p):
    """Join cleaned span texts for classification heuristics."""
    words = []
    for s in p.findall('span'):
        t = _clean_text(s.text or "")
        if t:
            words.append(t)
    return " ".join(words)


def _build_utterance(group):
    """Build an utterance dict from a group of (role, tag, text, ms) entries."""
    role, tag = group[0][0], group[0][1]
    words = [{'text': e[2], 'ms': e[3]} for e in group]
    full_text = ' '.join(e[2] for e in group)

    utt = {
        'speaker': role,
        'tag': tag,
        'text': full_text,
        'start_ms': group[0][3],
        'end_ms': None,
        'words': words,
    }
    if NON_VERBAL_RE.match(full_text):
        utt['type'] = 'non_verbal'
    return utt


def parse_xml_to_utterances(xml_path, offset_ms=0):
    """Parse an XML transcript into a list of structured utterance dicts.

    If offset_ms is provided, it is subtracted from every raw 'm' value to
    convert tape-time to video-time.  (offset_ms = xml_ms − video_ms, so
    corrected = raw_ms − offset_ms.)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    paragraphs = list(root.findall('.//p'))
    clean_texts = [_paragraph_clean_text(p) for p in paragraphs]
    first_roles = [_first_span_role(p) for p in paragraphs]

    entries = []
    last_known_tags = {}

    for idx, p in enumerate(paragraphs):
        spans = list(p.findall('span'))
        if not spans:
            continue

        prev_role = first_roles[idx - 1] if idx > 0 else None
        next_role = first_roles[idx + 1] if idx + 1 < len(paragraphs) else None

        role0, tag0, after0 = _parse_span_tag(spans[0].text or "")

        if role0 is not None:
            current_role = role0
            current_tag = tag0
            last_known_tags[current_role] = current_tag
            if after0:
                ms_str = spans[0].attrib.get('m')
                if ms_str:
                    entries.append((current_role, current_tag, after0, int(ms_str) - offset_ms))
            start_idx = 1
        else:
            inferred_role = _classify_untagged_paragraph(
                clean_texts[idx], prev_role, next_role
            )
            current_role = inferred_role
            current_tag = last_known_tags.get(inferred_role)
            start_idx = 0

        for i in range(start_idx, len(spans)):
            raw = (spans[i].text or "").strip()
            if not raw:
                continue

            ms_str = spans[i].attrib.get('m')
            if not ms_str:
                continue

            try:
                ms = int(ms_str) - offset_ms
            except ValueError:
                continue

            tag_role, tag_name, tag_after = _parse_span_tag(raw)
            if tag_role is not None:
                current_role = tag_role
                current_tag = tag_name
                last_known_tags[current_role] = current_tag
                if tag_after:
                    entries.append((current_role, current_tag, tag_after, ms))
                continue

            entries.append((current_role, current_tag, raw, ms))

    if not entries:
        return []

    # Group consecutive entries by (role, tag)
    utterances = []
    current_group = [entries[0]]

    for entry in entries[1:]:
        prev = current_group[-1]
        if entry[0] == prev[0] and entry[1] == prev[1]:
            current_group.append(entry)
        else:
            utterances.append(_build_utterance(current_group))
            current_group = [entry]
    utterances.append(_build_utterance(current_group))

    # Compute end_ms: next utterance's start_ms, or last word ms + 1000
    for i in range(len(utterances) - 1):
        utterances[i]['end_ms'] = utterances[i + 1]['start_ms']
    if utterances:
        last = utterances[-1]
        last_word_ms = last['words'][-1]['ms'] if last['words'] else last['start_ms']
        last['end_ms'] = last_word_ms + 1000

    return utterances


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    offsets = {}
    offsets_path = os.path.normpath(OFFSETS_PATH)
    if os.path.isfile(offsets_path):
        with open(offsets_path) as f:
            offsets = json.load(f)
        print(f"Loaded {len(offsets)} time offsets from {offsets_path}")
    else:
        print(f"No offsets file found at {offsets_path} — using raw timestamps")

    xml_files = sorted(f for f in os.listdir(TRANSCRIPT_DIR) if f.endswith('.xml'))
    print(f"Found {len(xml_files)} XML files to process")

    successes = 0
    failures = []
    no_offset = 0
    total_utterances = 0
    sample_file = None
    sample_utterances = None

    for filename in tqdm(xml_files, desc="Processing transcripts"):
        xml_path = os.path.join(TRANSCRIPT_DIR, filename)
        video_id = os.path.splitext(filename)[0]
        out_path = os.path.join(OUTPUT_DIR, f"{video_id}.json")

        offset_ms = offsets.get(video_id, 0)
        if offset_ms == 0 and offsets:
            no_offset += 1

        try:
            utterances = parse_xml_to_utterances(xml_path, offset_ms=offset_ms)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(utterances, f, ensure_ascii=False, indent=2)
            successes += 1
            total_utterances += len(utterances)
            if sample_file is None and utterances:
                sample_file = f"{video_id}.json"
                sample_utterances = utterances[:5]
        except Exception as e:
            failures.append((filename, str(e)))

    print(f"\n{'=' * 60}")
    print(f"Total files processed successfully: {successes}")
    print(f"Total files failed: {len(failures)}")
    if no_offset:
        print(f"Videos with no offset (used raw timestamps): {no_offset}")
    if failures:
        print("\nFailed files:")
        for fname, err in failures:
            print(f"  {fname}: {err}")
    print(f"\nTotal utterances generated: {total_utterances}")

    if sample_file and sample_utterances:
        print(f"\nSample output from {sample_file} (first 5 utterances):")
        print(json.dumps(sample_utterances, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
