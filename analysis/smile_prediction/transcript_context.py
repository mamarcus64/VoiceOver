"""
Extract transcript context around smile events.

For each smile task, pulls the surrounding utterances and highlights
the words spoken during the smile interval.
"""

import json
from pathlib import Path
from typing import Optional

from .dataset import DATA_DIR, SmileTask


def load_transcript(video_id: str, data_dir: Optional[Path] = None) -> Optional[list[dict]]:
    """Load LLM-cleaned transcript for a video."""
    data_dir = data_dir or DATA_DIR / "transcripts_llm"
    path = data_dir / f"{video_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def extract_context(
    task: SmileTask,
    context_utterances_before: int = 2,
    context_utterances_after: int = 1,
    max_words: int = 250,
) -> Optional[dict]:
    """
    Extract transcript context around a smile event.

    Returns dict with:
        "before": list of utterance strings before the smile
        "during": the utterance containing the smile, with smile words marked
        "during_words": just the words spoken during the smile
        "after": list of utterance strings after the smile
        "speaker_at_smile": "interviewer" or "interviewee"
        "has_laugh_marker": whether [LAUGHS] appears near the smile
        "full_context": single formatted string for LLM prompting
    """
    transcript = load_transcript(task.video_id)
    if not transcript:
        return None

    smile_start_ms = task.smile_start * 1000
    smile_end_ms = task.smile_end * 1000

    # Find the utterance containing the smile
    containing_idx = None
    for i, u in enumerate(transcript):
        if u["start_ms"] <= smile_end_ms and u["end_ms"] >= smile_start_ms:
            containing_idx = i
            break

    if containing_idx is None:
        # Smile falls in a gap — find nearest utterance
        best_dist = float("inf")
        for i, u in enumerate(transcript):
            mid = (u["start_ms"] + u["end_ms"]) / 2
            smile_mid = (smile_start_ms + smile_end_ms) / 2
            dist = abs(mid - smile_mid)
            if dist < best_dist:
                best_dist = dist
                containing_idx = i
        if containing_idx is None:
            return None

    u_during = transcript[containing_idx]

    # Extract words during the smile
    smile_words = []
    words = u_during.get("words", [])
    for w in words:
        wt = w["ms"]
        if smile_start_ms - 3000 <= wt <= smile_end_ms + 3000:
            smile_words.append(w["text"])

    # Trim to max_words centered on the smile if the utterance is very long
    all_words = [w["text"] for w in words]
    if len(all_words) > max_words:
        # Find the center word index
        center_idx = 0
        for wi, w in enumerate(words):
            if w["ms"] >= smile_start_ms:
                center_idx = wi
                break
        half = max_words // 2
        start = max(0, center_idx - half)
        end = min(len(all_words), start + max_words)
        start = max(0, end - max_words)
        all_words = all_words[start:end]
        trimmed = True
    else:
        trimmed = False

    during_text = " ".join(all_words)
    if trimmed:
        during_text = "..." + during_text + "..."

    # Gather before/after utterances
    before_utts = []
    for i in range(max(0, containing_idx - context_utterances_before), containing_idx):
        u = transcript[i]
        text = u["text"]
        if len(text) > 300:
            text = text[:300] + "..."
        before_utts.append(f"[{u['speaker']}]: {text}")

    after_utts = []
    for i in range(containing_idx + 1,
                   min(len(transcript), containing_idx + 1 + context_utterances_after)):
        u = transcript[i]
        text = u["text"]
        if len(text) > 300:
            text = text[:300] + "..."
        after_utts.append(f"[{u['speaker']}]: {text}")

    # Mark smile words in context
    smile_word_str = " ".join(smile_words) if smile_words else "(no word-level data at smile time)"

    # Check for laugh markers
    search_range = " ".join(smile_words) if smile_words else during_text
    has_laugh = "[LAUGHS]" in search_range or "[LAUGHTER]" in search_range

    # Build formatted context string
    parts = []
    if before_utts:
        parts.append("BEFORE THE SMILE:")
        parts.extend(before_utts)
        parts.append("")

    parts.append(f"DURING THE SMILE (speaker: {u_during['speaker']}):")
    parts.append(f"[{u_during['speaker']}]: {during_text}")
    parts.append(f">>> Words at smile moment: {smile_word_str}")
    parts.append("")

    if after_utts:
        parts.append("AFTER THE SMILE:")
        parts.extend(after_utts)

    return {
        "before": before_utts,
        "during_text": during_text,
        "during_words": smile_word_str,
        "after": after_utts,
        "speaker_at_smile": u_during["speaker"],
        "has_laugh_marker": has_laugh,
        "full_context": "\n".join(parts),
    }
