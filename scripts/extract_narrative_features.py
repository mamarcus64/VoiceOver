#!/usr/bin/env python3
"""
Deterministic, resumable LLM extraction of narrative features around smiles.

This script reads `data/detected_smiles.json`, pulls time-based transcript context
from `data/transcripts_llm/`, calls OpenRouter concurrently, and writes one
interpretable JSON file per smile plus a combined JSON output.

Designed for long-running jobs:
- fixed source order (original order from detected_smiles.json)
- resumable per-smile outputs
- concurrent OpenRouter requests
- robust parsing for JSON wrapped in prose or markdown fences
- progress logging to both stdout and a log file

Examples:
    python scripts/extract_narrative_features.py --limit 10
    python scripts/extract_narrative_features.py --min-score 0.75 --concurrency 40
    python scripts/extract_narrative_features.py --combine-only
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import difflib
import json
import os
import re
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
INPUT_SMILES = DATA_DIR / "detected_smiles.json"
TRANSCRIPT_DIR = DATA_DIR / "transcripts_llm"
OUTPUT_DIR = DATA_DIR / "llm_narrative_features"
ITEMS_DIR = OUTPUT_DIR / "items"
LOGS_DIR = OUTPUT_DIR / "logs"
PROMPT_FILE = Path(__file__).resolve().parent / "narrative_features_prompt.txt"
REQUEST_LOGS_SUBDIR = "requests"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_CONCURRENCY = 30
DEFAULT_WINDOW_BEFORE = 20.0
DEFAULT_WINDOW_AFTER = 15.0
DEFAULT_MAX_RETRIES = 5
DEFAULT_PROGRESS_EVERY = 10
DEFAULT_ANALYSIS_MODULO = 20
DEFAULT_ANALYSIS_OFFSET = 0

VALID_CONTENT_DOMAIN = {
    "pre_war",
    "wartime_or_camp",
    "liberation",
    "post_war",
    "present_day",
    "other",
}

VALID_TEMPORAL_SYNTAX = {
    "strict_past",
    "habitual_past",
    "present_narration",
    "present_reflection",
    "mixed",
}

VALID_NARRATIVE_STRUCTURE = {
    "orientation",
    "complicating_action",
    "evaluation",
    "resolution_coda",
    "other",
}

VALID_VALENCE = {
    "very_negative",
    "negative",
    "neutral",
    "positive",
    "very_positive",
}

CONTENT_DOMAIN_ALIASES = {
    "prewar": "pre_war",
    "pre-war": "pre_war",
    "pre war": "pre_war",
    "wartime": "wartime_or_camp",
    "war": "wartime_or_camp",
    "camp": "wartime_or_camp",
    "wartime/camp": "wartime_or_camp",
    "wartime or camp": "wartime_or_camp",
    "war_or_camp": "wartime_or_camp",
    "postwar": "post_war",
    "post-war": "post_war",
    "post war": "post_war",
    "present-day": "present_day",
    "present day": "present_day",
    "current_day": "present_day",
    "current": "present_day",
}

TEMPORAL_SYNTAX_ALIASES = {
    "simple_past": "strict_past",
    "past": "strict_past",
    "bounded_past": "strict_past",
    "habitual": "habitual_past",
    "ongoing_past": "habitual_past",
    "present-tense narration": "present_narration",
    "present tense narration": "present_narration",
    "present narration": "present_narration",
    "dramatic_present": "present_narration",
    "present-tense reflection": "present_reflection",
    "present tense reflection": "present_reflection",
    "reflection": "present_reflection",
}

NARRATIVE_STRUCTURE_ALIASES = {
    "orientation/scene-setting": "orientation",
    "scene_setting": "orientation",
    "scene-setting": "orientation",
    "scene setting": "orientation",
    "complicating": "complicating_action",
    "action": "complicating_action",
    "complicating action": "complicating_action",
    "resolution": "resolution_coda",
    "coda": "resolution_coda",
    "resolution/coda": "resolution_coda",
}

VALENCE_ALIASES = {
    "very negative": "very_negative",
    "very-positive": "very_positive",
    "very positive": "very_positive",
}


class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        stamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(stamped)
        self._fh.write(stamped + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class RateLimitError(Exception):
    def __init__(self, retry_after: int, message: str):
        super().__init__(message)
        self.retry_after = retry_after


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract narrative features around smiles")
    p.add_argument("--input", type=Path, default=INPUT_SMILES)
    p.add_argument("--transcript-dir", type=Path, default=TRANSCRIPT_DIR)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--prompt-file", type=Path, default=PROMPT_FILE)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--window-before", type=float, default=DEFAULT_WINDOW_BEFORE)
    p.add_argument("--window-after", type=float, default=DEFAULT_WINDOW_AFTER)
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    p.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--min-score", type=float, default=None)
    p.add_argument("--request-timeout", type=float, default=240.0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--analysis-modulo", type=int, default=DEFAULT_ANALYSIS_MODULO)
    p.add_argument("--analysis-offset", type=int, default=DEFAULT_ANALYSIS_OFFSET)
    p.add_argument("--combine-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--overwrite-existing", action="store_true")
    return p.parse_args()


def load_api_key() -> str:
    for env_name in ("OPENROUTER_API_KEY", "VOICES_OPENROUTER_KEY"):
        value = os.environ.get(env_name)
        if value:
            return value

    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key in {"OPENROUTER_API_KEY", "VOICES_OPENROUTER_KEY"} and value:
                return value

    raise RuntimeError("OpenRouter API key not found in environment or .env")


def load_smiles(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    smiles = data.get("smiles")
    if not isinstance(smiles, list):
        raise ValueError(f"Expected list at {path}: 'smiles'")
    return smiles


@lru_cache(maxsize=256)
def load_transcript_cached(transcript_dir_str: str, video_id: str) -> list[dict[str, Any]] | None:
    path = Path(transcript_dir_str) / f"{video_id}.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _words_in_range(words: list[dict[str, Any]], start_ms: float, end_ms: float) -> list[str]:
    selected = []
    for word in words:
        ms = word.get("ms")
        text = word.get("text")
        if ms is None or text is None:
            continue
        if start_ms <= ms <= end_ms:
            selected.append(str(text))
    return selected


def _excerpt_for_segment(segment: dict[str, Any], start_ms: float, end_ms: float) -> str:
    words = segment.get("words") or []
    excerpt_words = _words_in_range(words, start_ms, end_ms)
    if excerpt_words:
        return " ".join(excerpt_words).strip()
    return str(segment.get("text", "")).strip()


def build_context(
    smile: dict[str, Any],
    smile_index: int,
    transcript_dir: Path,
    window_before: float,
    window_after: float,
) -> dict[str, Any] | None:
    video_id = str(smile["video_id"])
    transcript = load_transcript_cached(str(transcript_dir), video_id)
    if not transcript:
        return None

    smile_start = float(smile["start_ts"])
    smile_end = float(smile["end_ts"])
    window_start_s = max(0.0, smile_start - window_before)
    window_end_s = smile_end + window_after
    window_start_ms = window_start_s * 1000.0
    window_end_ms = window_end_s * 1000.0
    smile_start_ms = smile_start * 1000.0
    smile_end_ms = smile_end * 1000.0

    selected_segments: list[dict[str, Any]] = []
    smile_words: list[str] = []

    for segment in transcript:
        seg_start = float(segment.get("start_ms", 0))
        seg_end = float(segment.get("end_ms", 0))
        if seg_end < window_start_ms or seg_start > window_end_ms:
            continue

        text = _excerpt_for_segment(segment, window_start_ms, window_end_ms)
        if not text:
            continue

        speaker = str(segment.get("speaker", "unknown"))
        selected_segments.append(
            {
                "speaker": speaker,
                "start_ms": int(seg_start),
                "end_ms": int(seg_end),
                "text": text,
            }
        )

        overlap_words = _words_in_range(segment.get("words") or [], smile_start_ms, smile_end_ms)
        smile_words.extend(overlap_words)

    if not selected_segments:
        return None

    if not smile_words:
        expanded = 1500.0
        for segment in transcript:
            overlap_words = _words_in_range(
                segment.get("words") or [],
                smile_start_ms - expanded,
                smile_end_ms + expanded,
            )
            smile_words.extend(overlap_words)
            if smile_words:
                break

    context_lines = [
        f"SMILE INDEX: {smile_index}",
        (
            "WINDOW: "
            f"{window_start_s:.3f}s to {window_end_s:.3f}s "
            f"(smile at {smile_start:.3f}s to {smile_end:.3f}s)"
        ),
        "",
        "TRANSCRIPT CONTEXT:",
    ]
    for seg in selected_segments:
        seg_start_s = seg["start_ms"] / 1000.0
        seg_end_s = seg["end_ms"] / 1000.0
        context_lines.append(
            f"[{seg_start_s:.3f}-{seg_end_s:.3f}] {seg['speaker']}: {seg['text']}"
        )
    context_lines.extend(
        [
            "",
            "WORDS AT THE SMILE MOMENT:",
            " ".join(smile_words) if smile_words else "(no word-level words captured at the smile moment)",
        ]
    )

    return {
        "window_start_s": round(window_start_s, 3),
        "window_end_s": round(window_end_s, 3),
        "smile_words": smile_words,
        "segments": selected_segments,
        "context_text": "\n".join(context_lines),
    }


def render_prompt(
    prompt_template: str,
    smile_index: int,
    smile: dict[str, Any],
    context_text: str,
    window_before: float,
    window_after: float,
    analysis_requested: bool,
) -> str:
    analysis_instruction = (
        "Include an \"analysis\" field with 2-5 sentences explaining the reasoning behind "
        "the labels, distinguishing narrated emotion from present-day stance when relevant."
        if analysis_requested
        else "Do not include any analysis, reasoning, summary, or explanation field. "
             "Return only the five categorical labels."
    )
    output_schema = (
        '{\n'
        '  "analysis": "2-5 sentences explaining the reasoning behind the labels, distinguishing narrated emotion from present-day stance when relevant.",\n'
        '  "content_domain": "pre_war",\n'
        '  "temporal_syntax": "strict_past",\n'
        '  "narrative_structure": "orientation",\n'
        '  "narrative_valence": "negative",\n'
        '  "present_day_valence": "neutral"\n'
        '}'
        if analysis_requested
        else
        '{\n'
        '  "content_domain": "pre_war",\n'
        '  "temporal_syntax": "strict_past",\n'
        '  "narrative_structure": "orientation",\n'
        '  "narrative_valence": "negative",\n'
        '  "present_day_valence": "neutral"\n'
        '}'
    )
    replacements = {
        "{{SMILE_INDEX}}": str(smile_index),
        "{{VIDEO_ID}}": str(smile["video_id"]),
        "{{SMILE_START_SECONDS}}": f"{float(smile['start_ts']):.3f}",
        "{{SMILE_END_SECONDS}}": f"{float(smile['end_ts']):.3f}",
        "{{SMILE_SCORE}}": f"{float(smile.get('score', 0.0)):.4f}",
        "{{WINDOW_BEFORE_SECONDS}}": f"{window_before:.1f}",
        "{{WINDOW_AFTER_SECONDS}}": f"{window_after:.1f}",
        "{{CONTEXT}}": context_text,
        "{{ANALYSIS_INSTRUCTION}}": analysis_instruction,
        "{{OUTPUT_SCHEMA}}": output_schema,
    }
    prompt = prompt_template
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    if "{{ANALYSIS_INSTRUCTION}}" not in prompt_template:
        prompt += "\n\n" + analysis_instruction
    if "{{OUTPUT_SCHEMA}}" not in prompt_template:
        prompt += "\n\nUse this exact schema:\n" + output_schema
    return prompt


def canonicalize_label(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"", "none", "null", "n/a", "na"}:
        return ""
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    parts = [part for part in text.split("_") if part]
    deduped = []
    for part in parts:
        if not deduped or deduped[-1] != part:
            deduped.append(part)
    text = "_".join(deduped)
    return text


def normalize_value(value: Any, valid: set[str], aliases: dict[str, str]) -> str | None:
    raw = str(value).strip()
    if not raw:
        return None
    canonical = canonicalize_label(raw)
    if not canonical:
        return None

    candidate_map = {canonicalize_label(v): v for v in valid}
    for alias, target in aliases.items():
        candidate_map[canonicalize_label(alias)] = target

    if canonical in candidate_map:
        return candidate_map[canonical]

    prefix_matches = {
        target
        for key, target in candidate_map.items()
        if (key.startswith(canonical) or canonical.startswith(key)) and len(canonical) >= 6
    }
    if len(prefix_matches) == 1:
        return next(iter(prefix_matches))

    close = difflib.get_close_matches(canonical, list(candidate_map.keys()), n=1, cutoff=0.82)
    if close:
        return candidate_map[close[0]]

    return None


def extract_json_candidate(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty response")

    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            for i in range(1, len(lines)):
                if lines[i].strip() == "```":
                    inner = "\n".join(lines[1:i]).strip()
                    if inner.lower().startswith("json"):
                        inner = inner[4:].strip()
                    if inner:
                        return inner

    start_positions = [idx for idx, ch in enumerate(stripped) if ch in "{["]
    if not start_positions:
        return stripped

    for start in start_positions:
        opening = stripped[start]
        closing = "}" if opening == "{" else "]"
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(stripped)):
            ch = stripped[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == opening:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    return stripped[start : idx + 1]

    return stripped


def parse_json_like(text: str) -> Any:
    candidate = extract_json_candidate(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return ast.literal_eval(candidate)


def normalize_annotation(parsed: Any, require_analysis: bool) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected top-level object, got {type(parsed).__name__}")

    analysis = str(parsed.get("analysis", "")).strip()
    if not analysis:
        analysis = str(parsed.get("reasoning", parsed.get("summary", ""))).strip()

    normalized = {
        "analysis": analysis,
        "content_domain": normalize_value(
            parsed.get("content_domain"),
            VALID_CONTENT_DOMAIN,
            CONTENT_DOMAIN_ALIASES,
        ),
        "temporal_syntax": normalize_value(
            parsed.get("temporal_syntax"),
            VALID_TEMPORAL_SYNTAX,
            TEMPORAL_SYNTAX_ALIASES,
        ),
        "narrative_structure": normalize_value(
            parsed.get("narrative_structure"),
            VALID_NARRATIVE_STRUCTURE,
            NARRATIVE_STRUCTURE_ALIASES,
        ),
        "narrative_valence": normalize_value(
            parsed.get("narrative_valence"),
            VALID_VALENCE,
            VALENCE_ALIASES,
        ),
        "present_day_valence": normalize_value(
            parsed.get("present_day_valence"),
            VALID_VALENCE,
            VALENCE_ALIASES,
        ),
    }

    errors = []
    if require_analysis and not normalized["analysis"]:
        errors.append("Missing 'analysis'")
    for key in (
        "content_domain",
        "temporal_syntax",
        "narrative_structure",
        "narrative_valence",
        "present_day_valence",
    ):
        if normalized[key] is None:
            errors.append(f"Invalid or missing '{key}': {parsed.get(key)!r}")

    return normalized, errors


async def call_openrouter(
    session: httpx.AsyncClient,
    prompt: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
) -> tuple[str, dict[str, Any]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://voiceover-project",
        "X-Title": "VoiceOver Narrative Feature Extraction",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    resp = await session.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=timeout_s,
    )
    if resp.status_code == 429:
        retry_after = int(resp.headers.get("Retry-After", "30"))
        raise RateLimitError(retry_after, f"Rate limited, retry after {retry_after}s")
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return content, usage


def item_path(items_dir: Path, smile_index: int) -> Path:
    return items_dir / f"{smile_index:06d}.json"


def request_log_path(output_dir: Path, smile_index: int) -> Path:
    return output_dir / "logs" / REQUEST_LOGS_SUBDIR / f"{smile_index:06d}.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def item_is_done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data.get("status") == "ok" and "annotation" in data
    except Exception:
        return False


def build_smile_selection(
    smiles: list[dict[str, Any]],
    start_index: int,
    limit: int | None,
    min_score: float | None,
) -> list[tuple[int, dict[str, Any]]]:
    selected: list[tuple[int, dict[str, Any]]] = []
    for idx in range(start_index, len(smiles)):
        smile = smiles[idx]
        if min_score is not None and float(smile.get("score", 0.0)) < min_score:
            continue
        selected.append((idx, smile))
        if limit is not None and len(selected) >= limit:
            break
    return selected


def should_request_analysis(smile_index: int, modulo: int, offset: int) -> bool:
    if modulo <= 0:
        return False
    return smile_index % modulo == offset % modulo


def combine_outputs(args: argparse.Namespace, selected_indices: set[int] | None = None) -> Path:
    items_dir = args.output_dir / "items"
    output_path = args.output_dir / "narrative_features_combined.json"

    records: list[dict[str, Any]] = []
    for path in sorted(items_dir.glob("*.json")):
        with path.open(encoding="utf-8") as f:
            record = json.load(f)
        if record.get("status") != "ok":
            continue
        smile_index = int(record["smile_index"])
        if selected_indices is not None and smile_index not in selected_indices:
            continue
        ann = record["annotation"]
        records.append(
            {
                "smile_index": smile_index,
                "video_id": record["video_id"],
                "smile": record["smile"],
                "context_window_seconds": record["context_window_seconds"],
                "analysis_requested": record.get("analysis_requested", False),
                "content_domain": ann["content_domain"],
                "temporal_syntax": ann["temporal_syntax"],
                "narrative_structure": ann["narrative_structure"],
                "narrative_valence": ann["narrative_valence"],
                "present_day_valence": ann["present_day_valence"],
                "elapsed_s": record.get("elapsed_s"),
                "usage": record.get("usage", {}),
            }
        )
        if ann.get("analysis"):
            records[-1]["analysis"] = ann["analysis"]

    combined = {
        "run_metadata": {
            "input": str(args.input),
            "transcript_dir": str(args.transcript_dir),
            "prompt_file": str(args.prompt_file),
            "model": args.model,
            "temperature": args.temperature,
            "analysis_modulo": args.analysis_modulo,
            "analysis_offset": args.analysis_offset,
            "window_before_seconds": args.window_before,
            "window_after_seconds": args.window_after,
            "min_score": args.min_score,
            "start_index": args.start_index,
            "limit": args.limit,
            "n_analysis_requested": sum(1 for r in records if r.get("analysis_requested")),
            "n_results": len(records),
        },
        "results": records,
    }

    output_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


async def run_async(args: argparse.Namespace, logger: Logger) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items_dir = args.output_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs" / REQUEST_LOGS_SUBDIR).mkdir(parents=True, exist_ok=True)

    smiles = load_smiles(args.input)
    selected = build_smile_selection(smiles, args.start_index, args.limit, args.min_score)
    selected_indices = {idx for idx, _ in selected}

    logger.log(f"Loaded {len(smiles):,} smiles from {args.input}")
    logger.log(f"Selected {len(selected):,} smiles to consider")

    if args.combine_only:
        output_path = combine_outputs(args, selected_indices if selected else None)
        logger.log(f"Combined output written to {output_path}")
        return

    prompt_template = args.prompt_file.read_text(encoding="utf-8")

    pending: list[tuple[int, dict[str, Any]]] = []
    skipped_existing = 0
    skipped_missing_transcript = 0

    for smile_index, smile in selected:
        out_path = item_path(items_dir, smile_index)
        if not args.overwrite_existing and not args.no_resume and item_is_done(out_path):
            skipped_existing += 1
            continue
        transcript_path = args.transcript_dir / f"{smile['video_id']}.json"
        if not transcript_path.exists():
            skipped_missing_transcript += 1
            continue
        pending.append((smile_index, smile))

    logger.log(f"Already complete and skipped: {skipped_existing:,}")
    logger.log(f"Skipped due to missing transcript: {skipped_missing_transcript:,}")
    logger.log(f"Pending API calls: {len(pending):,}")

    if args.dry_run:
        if pending:
            sample_index, sample_smile = pending[0]
            context = build_context(
                sample_smile,
                sample_index,
                args.transcript_dir,
                args.window_before,
                args.window_after,
            )
            if context is None:
                logger.log("Dry run sample had no context.")
            else:
                prompt = render_prompt(
                    prompt_template,
                    sample_index,
                    sample_smile,
                    context["context_text"],
                    args.window_before,
                    args.window_after,
                    should_request_analysis(
                        sample_index, args.analysis_modulo, args.analysis_offset
                    ),
                )
                logger.log("Dry run prompt preview:")
                print(prompt)
        output_path = combine_outputs(args, selected_indices if selected else None)
        logger.log(f"Combined output written to {output_path}")
        return

    if not pending:
        logger.log("No pending calls. Combining outputs.")
        output_path = combine_outputs(args, selected_indices if selected else None)
        logger.log(f"Combined output written to {output_path}")
        return

    api_key = load_api_key()
    queue: asyncio.Queue[tuple[int, dict[str, Any]]] = asyncio.Queue()
    for item in pending:
        queue.put_nowait(item)

    counter = {
        "done": 0,
        "errors": 0,
        "started_at": time.monotonic(),
        "total": len(pending),
    }
    counter_lock = asyncio.Lock()

    async def worker(worker_id: int, session: httpx.AsyncClient) -> None:
        while True:
            try:
                smile_index, smile = queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            out_path = item_path(items_dir, smile_index)
            req_log_path = request_log_path(args.output_dir, smile_index)
            try:
                context = build_context(
                    smile,
                    smile_index,
                    args.transcript_dir,
                    args.window_before,
                    args.window_after,
                )
                if context is None:
                    async with counter_lock:
                        counter["errors"] += 1
                    logger.log(f"[SKIP] {smile_index:06d} no transcript context")
                    write_json(
                        req_log_path,
                        {
                            "status": "skipped_no_context",
                            "smile_index": smile_index,
                            "video_id": str(smile["video_id"]),
                            "smile": smile,
                        },
                    )
                    continue

                analysis_requested = should_request_analysis(
                    smile_index, args.analysis_modulo, args.analysis_offset
                )
                prompt = render_prompt(
                    prompt_template,
                    smile_index,
                    smile,
                    context["context_text"],
                    args.window_before,
                    args.window_after,
                    analysis_requested,
                )
                request_payload = {
                    "status": "pending",
                    "smile_index": smile_index,
                    "video_id": str(smile["video_id"]),
                    "analysis_requested": analysis_requested,
                    "smile": {
                        "start_ts": float(smile["start_ts"]),
                        "end_ts": float(smile["end_ts"]),
                        "peak_r": smile.get("peak_r"),
                        "mean_r": smile.get("mean_r"),
                        "score": smile.get("score"),
                        "model": smile.get("model"),
                    },
                    "context_window_seconds": {
                        "before": args.window_before,
                        "after": args.window_after,
                    },
                    "context_bounds_seconds": {
                        "start": context["window_start_s"],
                        "end": context["window_end_s"],
                    },
                    "context_text": context["context_text"],
                    "context_segments": context["segments"],
                    "smile_words": context["smile_words"],
                    "prompt_text": prompt,
                    "attempts": [],
                }
                write_json(req_log_path, request_payload)

                last_error = None
                for attempt in range(1, args.max_retries + 1):
                    try:
                        t0 = time.monotonic()
                        raw_text, usage = await call_openrouter(
                            session=session,
                            prompt=prompt,
                            api_key=api_key,
                            model=args.model,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            timeout_s=args.request_timeout,
                        )
                        elapsed = time.monotonic() - t0
                        parsed = parse_json_like(raw_text)
                        annotation, validation_errors = normalize_annotation(
                            parsed, require_analysis=analysis_requested
                        )
                        if validation_errors:
                            raise ValueError("; ".join(validation_errors))

                        record = {
                            "status": "ok",
                            "smile_index": smile_index,
                            "video_id": str(smile["video_id"]),
                            "smile": {
                                "start_ts": float(smile["start_ts"]),
                                "end_ts": float(smile["end_ts"]),
                                "peak_r": smile.get("peak_r"),
                                "mean_r": smile.get("mean_r"),
                                "score": smile.get("score"),
                                "model": smile.get("model"),
                            },
                            "context_window_seconds": {
                                "before": args.window_before,
                                "after": args.window_after,
                            },
                            "context_bounds_seconds": {
                                "start": context["window_start_s"],
                                "end": context["window_end_s"],
                            },
                            "context_text": context["context_text"],
                            "context_segments": context["segments"],
                            "smile_words": context["smile_words"],
                            "analysis_requested": analysis_requested,
                            "model": args.model,
                            "temperature": args.temperature,
                            "elapsed_s": round(elapsed, 2),
                            "usage": usage,
                            "raw_response_text": raw_text,
                            "annotation": annotation,
                        }
                        write_json(out_path, record)
                        request_payload["status"] = "ok"
                        request_payload["elapsed_s"] = round(elapsed, 2)
                        request_payload["usage"] = usage
                        request_payload["raw_response_text"] = raw_text
                        request_payload["annotation"] = annotation
                        request_payload["attempts"].append(
                            {
                                "attempt": attempt,
                                "status": "ok",
                                "elapsed_s": round(elapsed, 2),
                            }
                        )
                        write_json(req_log_path, request_payload)

                        async with counter_lock:
                            counter["done"] += 1
                            done = counter["done"]
                            total = counter["total"]
                            errors = counter["errors"]
                            elapsed_total = time.monotonic() - counter["started_at"]
                            rate = done / elapsed_total if elapsed_total > 0 else 0.0
                            remaining = total - done
                            eta_s = remaining / rate if rate > 0 else 0.0
                            should_log = done % args.progress_every == 0 or done == total

                        if should_log:
                            logger.log(
                                f"[PROGRESS] {done}/{total} done, {errors} errors, "
                                f"{elapsed_total:.1f}s elapsed, ~{eta_s/60:.1f}m remaining"
                            )
                        logger.log(
                            f"[OK] {smile_index:06d} {smile['video_id']} "
                            f"{float(smile['start_ts']):.3f}-{float(smile['end_ts']):.3f}s "
                            f"in {elapsed:.1f}s (worker {worker_id}, analysis={analysis_requested})"
                        )
                        last_error = None
                        break

                    except RateLimitError as e:
                        last_error = str(e)
                        delay = max(30.0, float(e.retry_after))
                        request_payload["attempts"].append(
                            {
                                "attempt": attempt,
                                "status": "retry_rate_limit",
                                "error": last_error,
                                "sleep_s": round(delay, 2),
                            }
                        )
                        write_json(req_log_path, request_payload)
                        logger.log(
                            f"[RETRY] {smile_index:06d} attempt {attempt}/{args.max_retries} "
                            f"after {last_error}; sleeping {delay:.0f}s"
                        )
                        if attempt < args.max_retries:
                            await asyncio.sleep(delay)

                    except httpx.HTTPStatusError as e:
                        status = e.response.status_code
                        last_error = f"HTTP {status}: {e}"
                        request_payload["attempts"].append(
                            {
                                "attempt": attempt,
                                "status": "http_error",
                                "error": last_error,
                            }
                        )
                        write_json(req_log_path, request_payload)
                        if 400 <= status < 500:
                            logger.log(f"[HTTP {status}] {smile_index:06d} non-retryable: {e}")
                            break
                        delay = 2.0 * (2 ** (attempt - 1))
                        logger.log(
                            f"[RETRY] {smile_index:06d} attempt {attempt}/{args.max_retries} "
                            f"after {last_error}; sleeping {delay:.0f}s"
                        )
                        if attempt < args.max_retries:
                            await asyncio.sleep(delay)

                    except (asyncio.TimeoutError, httpx.RequestError, httpx.TimeoutException) as e:
                        last_error = f"{type(e).__name__}: {e}"
                        delay = 2.0 * (2 ** (attempt - 1))
                        request_payload["attempts"].append(
                            {
                                "attempt": attempt,
                                "status": "network_error",
                                "error": last_error,
                                "sleep_s": round(delay, 2),
                            }
                        )
                        write_json(req_log_path, request_payload)
                        logger.log(
                            f"[RETRY] {smile_index:06d} attempt {attempt}/{args.max_retries} "
                            f"after {last_error}; sleeping {delay:.0f}s"
                        )
                        if attempt < args.max_retries:
                            await asyncio.sleep(delay)

                    except (ValueError, KeyError, SyntaxError) as e:
                        last_error = f"Parse/validation error: {e}"
                        delay = 2.0 * (2 ** (attempt - 1))
                        request_payload["attempts"].append(
                            {
                                "attempt": attempt,
                                "status": "parse_error",
                                "error": last_error,
                                "sleep_s": round(delay, 2),
                            }
                        )
                        write_json(req_log_path, request_payload)
                        logger.log(
                            f"[RETRY] {smile_index:06d} attempt {attempt}/{args.max_retries} "
                            f"after {last_error}; sleeping {delay:.0f}s"
                        )
                        if attempt < args.max_retries:
                            await asyncio.sleep(delay)

                if last_error is not None:
                    async with counter_lock:
                        counter["errors"] += 1
                    request_payload["status"] = "failed"
                    request_payload["final_error"] = last_error
                    write_json(req_log_path, request_payload)
                    logger.log(f"[FAIL] {smile_index:06d} {smile['video_id']}: {last_error}")

            finally:
                queue.task_done()

    limits = httpx.Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=args.concurrency,
    )
    async with httpx.AsyncClient(limits=limits) as session:
        workers = [
            asyncio.create_task(worker(worker_id=i + 1, session=session))
            for i in range(args.concurrency)
        ]
        await queue.join()
        await asyncio.gather(*workers)

    output_path = combine_outputs(args, selected_indices if selected else None)
    logger.log(f"Combined output written to {output_path}")


def main() -> None:
    args = parse_args()
    log_path = args.output_dir / "logs" / "progress.log"
    logger = Logger(log_path)
    logger.log("=== Narrative feature extraction ===")
    logger.log(f"Model: {args.model}")
    logger.log(f"Temperature: {args.temperature}")
    logger.log(
        f"Analysis sampling: every {args.analysis_modulo} smiles, offset {args.analysis_offset}"
    )
    logger.log(f"Concurrency: {args.concurrency}")
    logger.log(
        f"Context window: {args.window_before:.1f}s before, {args.window_after:.1f}s after"
    )
    logger.log(f"Prompt file: {args.prompt_file}")
    try:
        asyncio.run(run_async(args, logger))
    finally:
        logger.close()


if __name__ == "__main__":
    main()
