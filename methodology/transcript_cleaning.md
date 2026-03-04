# Transcript Cleaning Pipeline

## Overview

The Fortunoff Video Archive provides XML transcripts with word-level timestamps
for each testimony tape segment. These are processed into structured JSON and
then refined with an LLM pass for speaker-role correction. A per-video temporal
offset is computed to align the archive transcript timestamps (which reference
the original tape) with the corresponding YouTube-hosted video files.

## Data Flow

```
test_data/transcripts/*.xml          (1) Raw archive XML
        │
        ▼
standardize_transcripts.py           (2) Parse & structure
        │
        ▼
VoiceOver/data/transcripts/*.json    (3) Structured JSON (one per segment)
        │
        ▼
llm_transcript_pass.py               (4) LLM speaker-role correction
        │
        ▼
VoiceOver/data/transcripts_llm/*.json (5) Final transcripts served by API
```

A separate alignment step produces per-video time offsets:

```
test_data/transcripts/*.xml  ─┐
                               ├─► compute_transcript_offsets.py ─► VoiceOver/data/transcript_offsets.json
test_data/videos/*.mp4       ─┘
```

The offsets are consumed by `standardize_transcripts.py` at step (2).

---

## Step 1 — Raw XML Format

**Location:** `test_data/transcripts/{IntCode}.{TapeNumber}.xml`
(~5,130 files)

Each file is a `<transcription>` element containing `<p>` paragraphs.
Each paragraph contains `<span>` elements with a millisecond attribute `m`:

```xml
<p>
  <span m="35260">INT 1:</span>
  <span m="35384">With</span>
  <span m="35758">name,</span>
  <span m="36290">date,</span>
  <span m="36750">location.</span>
</p>
```

The first span of a paragraph may begin with a speaker tag (`INT 1:`, `EJ:`,
etc.) which identifies who is speaking. Paragraphs without a speaker tag are
classified heuristically.

### Timestamp semantics

The `m` attribute represents **milliseconds from the start of the original
archive tape**, not from the start of the YouTube-hosted video. Because each
tape includes a variable-length leader (color bars, tone, slate) before the
digitised content begins, the `m` values are systematically higher than the
corresponding playback time in the YouTube MP4 by a per-video constant. This
offset is typically 5–20 seconds and varies across tapes (see Step 2a below).

---

## Step 2a — Temporal Offset Computation

**Script:** `VoiceOver/scripts/compute_transcript_offsets.py`
**Output:** `VoiceOver/data/transcript_offsets.json`

### Problem

The XML `m` values are in tape-time. The YouTube videos begin at some unknown
offset into the tape. This offset is constant per video but varies across
videos (empirically observed range: ~5–20 s). Manual spot-checks of four
videos found offsets of 9, 11, 11, and 13 seconds.

### Method

We exploit the fact that the offset is a single scalar unknown per video, while
the XML provides thousands of word-level timestamp anchors. A lightweight
automatic speech recognition (ASR) pass on the video audio produces an
independent set of word-level timestamps in video-time. Aligning the two word
sequences recovers the offset.

1. **ASR pass.** We run `faster-whisper` (CTranslate2 backend, `base` model)
   on the first 300 seconds of each video's audio, extracting word-level
   timestamps. Only the beginning of the video is needed because a small number
   of matched words suffices to estimate a single scalar.

2. **Word matching.** We walk the ASR word list and the XML word list in
   parallel. For each ASR word, we scan forward in the XML for a
   case-insensitive exact text match within a ±60 s window of the expected
   position. Each match produces a time-difference sample
   `δ = xml_ms − asr_ms`.

3. **Robust estimation.** We take the **median** of all δ samples as the
   offset estimate. The median is insensitive to outliers caused by ASR
   mis-recognitions, XML transcription errors, or timing jitter. As a quality
   check, we compute the median absolute deviation (MAD) and the number of
   matched words; videos with too few matches or too high MAD are flagged for
   manual review.

4. **Output.** A JSON file mapping each video ID to its offset in milliseconds:

   ```json
   {
     "8.1": 11240,
     "8.2": 10870,
     ...
   }
   ```

### Reproducing

```bash
CUDA_VISIBLE_DEVICES=2 python VoiceOver/scripts/compute_transcript_offsets.py
```

Options:
- `--limit N` — process only the first N videos (for testing)
- `--asr-seconds N` — how many seconds of audio to transcribe (default 300)
- `--model SIZE` — Whisper model size (default `base`)

---

## Step 2 — Standardization (XML → JSON)

**Script:** `VoiceOver/scripts/standardize_transcripts.py`

Parses each XML file into a JSON array of utterance objects:

```json
{
  "speaker": "interviewer",
  "tag": "INT 1",
  "text": "With name, date, location.",
  "start_ms": 24020,
  "end_ms": 26400,
  "words": [
    {"text": "With", "ms": 24144},
    {"text": "name,", "ms": 24518},
    ...
  ]
}
```

Key processing steps:

1. **Speaker identification.** The first span of each `<p>` is tested against a
   known set of interviewer tags (`INT`, `INT 1`, `INT 2`, `CREW`, etc.).
   Untagged paragraphs are classified by a weighted-vote heuristic that
   considers surrounding speaker context, question marks, cue phrases, and
   utterance length.

2. **Word grouping.** Consecutive spans with the same speaker are merged into a
   single utterance. Each word retains its individual `ms` timestamp.

3. **Temporal offset correction.** If `VoiceOver/data/transcript_offsets.json`
   exists, the per-video offset is subtracted from every `m` value before
   building utterances. This aligns all timestamps to video-time.

4. **`end_ms` computation.** Each utterance's `end_ms` is set to the next
   utterance's `start_ms`. The final utterance's `end_ms` is set to its last
   word's timestamp + 1000 ms.

### Reproducing

```bash
python VoiceOver/scripts/standardize_transcripts.py
```

---

## Step 3 — LLM Speaker-Role Correction

**Script:** `VoiceOver/scripts/llm_transcript_pass.py`

The heuristic speaker classification in Step 2 occasionally misattributes
utterances. An LLM pass reviews each transcript and returns a corrections-only
JSON payload identifying utterances whose speaker role should be flipped.

- **Model:** OpenRouter API (`openai/gpt-oss-120b` by default)
- **Approach:** corrections-only — the LLM never rewrites text or timestamps;
  it returns `{"corrections": [{"index": N, "correct_speaker": "...", "reason": "..."}]}`
- **Parallelism:** up to 50 concurrent API calls (configurable via `--concurrency`)
- **Resumable:** skips videos whose output file already exists

### Reproducing

```bash
export VOICES_OPENROUTER_KEY="..."
python VoiceOver/scripts/llm_transcript_pass.py --diff-report
```

Options:
- `--limit N` — process first N files only
- `--concurrency N` — max parallel API calls (default 50)
- `--model MODEL` — OpenRouter model identifier
- `--provider NAME` — restrict to a specific provider (e.g. `Cerebras`)
- `--no-resume` — reprocess even if output exists
- `--diff-report` — write a human-readable diff report to `_diff_report.txt`

---

## Output Schema

Each final JSON file (`transcripts_llm/{video_id}.json`) is an array of
utterance objects. The frontend loads these via
`GET /api/videos/{video_id}/transcript` and compares `start_ms`/`end_ms`/`words[].ms`
against `HTMLVideoElement.currentTime * 1000` for synchronized display.

| Field      | Type     | Description                              |
|------------|----------|------------------------------------------|
| `speaker`  | string   | `"interviewer"` or `"interviewee"`       |
| `tag`      | string   | Speaker initials (e.g. `"INT 1"`, `"EJ"`)|
| `text`     | string   | Full utterance text                      |
| `start_ms` | integer  | Utterance start time (video-time, ms)    |
| `end_ms`   | integer  | Utterance end time (video-time, ms)      |
| `words`    | array    | `[{"text": "...", "ms": N}, ...]`        |
| `type`     | string?  | `"non_verbal"` for bracketed markers     |
