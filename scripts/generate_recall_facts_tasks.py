#!/usr/bin/env python3
"""
Generate 200 human validation tasks from LLM-annotated data.

  100 × topic validation    – human confirms whether a specific LLM-assigned topic is appropriate
  100 × memory_type         – human independently labels internal vs. external (no LLM answer shown)

Both sets are sampled randomly (no stratification) to reflect the actual data distribution.
Tasks are output in two sequential blocks: topic tasks first (ids 0–99), then memory_type (ids 100–199).

Outputs: data/recall_facts_annotations/tasks.json

Run from the repo root:
    python scripts/generate_recall_facts_tasks.py
"""

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LLM_DIR = REPO_ROOT / "data" / "llm_annotated_recall_facts"
OUT_DIR = REPO_ROOT / "data" / "recall_facts_annotations"
OUT_FILE = OUT_DIR / "tasks.json"

N_TOPIC_TASKS = 100
N_MEMORY_TYPE_TASKS = 100
CONTEXT_WINDOW = 2    # sentences before/after to include as context
SEED = 42


def load_all_sentences() -> list[dict]:
    """Load every sentence from every annotated transcript."""
    sentences = []
    for path in sorted(LLM_DIR.glob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except Exception as e:
            print(f"  [SKIP] {path.name}: {e}")
            continue
        transcript_id = doc.get("transcript_id", path.stem)
        flat = doc.get("sentences", [])
        for i, sent in enumerate(flat):
            sentences.append({
                "transcript_id": transcript_id,
                "global_idx": i,
                "total_in_transcript": len(flat),
                "text": sent.get("text", "").strip(),
                "memory_type": sent.get("memory_type", "external"),
                "topics": sent.get("topics", []),
                "_flat": flat,
            })
    return sentences


def get_context(flat: list[dict], global_idx: int, window: int) -> tuple[list[str], list[str]]:
    before = [
        flat[j].get("text", "").strip()
        for j in range(max(0, global_idx - window), global_idx)
        if flat[j].get("text", "").strip()
    ]
    after = [
        flat[j].get("text", "").strip()
        for j in range(global_idx + 1, min(len(flat), global_idx + window + 1))
        if flat[j].get("text", "").strip()
    ]
    return before, after


def sample_topic_tasks(sentences: list[dict], rng: random.Random) -> list[dict]:
    """
    100 topic tasks: randomly sample from sentences that have at least one topic.
    Each task validates one randomly chosen topic from that sentence's assigned topics.
    Pure random sampling — no stratification by topic.
    """
    pool = [s for s in sentences if s["topics"] and len(s["text"]) > 20]
    rng.shuffle(pool)
    tasks = []
    for sent in pool:
        if len(tasks) >= N_TOPIC_TASKS:
            break
        topic = rng.choice(sent["topics"])
        tasks.append({**sent, "topic_to_validate": topic})
    return tasks


def sample_memory_type_tasks(sentences: list[dict], rng: random.Random) -> list[dict]:
    """
    100 memory_type tasks: randomly sample from all sentences with enough text.
    Annotator labels independently (LLM answer not shown).
    Pure random sampling — no stratification by memory_type.
    """
    pool = [s for s in sentences if len(s["text"]) > 20]
    rng.shuffle(pool)
    return pool[:N_MEMORY_TYPE_TASKS]


def build_task(task_id: int, kind: str, sent: dict) -> dict:
    flat = sent["_flat"]
    gidx = sent["global_idx"]
    before, after = get_context(flat, gidx, CONTEXT_WINDOW)
    base = {
        "id": task_id,
        "type": kind,
        "transcript_id": sent["transcript_id"],
        "sentence_text": sent["text"],
        "context_before": before,
        "context_after": after,
        "llm_memory_type": sent["memory_type"],
        "llm_topics": sent["topics"],
    }
    if kind == "topic":
        base["topic_to_validate"] = sent["topic_to_validate"]
    return base


def main():
    print(f"Loading annotated sentences from {LLM_DIR} …")
    sentences = load_all_sentences()
    if not sentences:
        print("ERROR: No annotated sentences found. Run annotate.py first.")
        sys.exit(1)
    print(f"  Loaded {len(sentences):,} sentences from "
          f"{len(set(s['transcript_id'] for s in sentences))} transcripts")

    rng = random.Random(SEED)

    topic_sents = sample_topic_tasks(sentences, rng)
    memory_sents = sample_memory_type_tasks(sentences, rng)

    if len(topic_sents) < N_TOPIC_TASKS:
        print(f"  WARNING: only {len(topic_sents)} topic tasks available (wanted {N_TOPIC_TASKS})")
    if len(memory_sents) < N_MEMORY_TYPE_TASKS:
        print(f"  WARNING: only {len(memory_sents)} memory_type tasks available (wanted {N_MEMORY_TYPE_TASKS})")

    # Topic block first (ids 0–99), then memory_type block (ids 100–199)
    tasks = []
    for i, s in enumerate(topic_sents[:N_TOPIC_TASKS]):
        tasks.append(build_task(i, "topic", s))
    for i, s in enumerate(memory_sents[:N_MEMORY_TYPE_TASKS]):
        tasks.append(build_task(N_TOPIC_TASKS + i, "memory_type", s))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_topic_tasks": N_TOPIC_TASKS,
        "n_memory_type_tasks": N_MEMORY_TYPE_TASKS,
        "total_tasks": len(tasks),
        "tasks": tasks,
    }
    OUT_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(tasks)} tasks → {OUT_FILE}")

    # Summary
    internal_count = sum(1 for s in memory_sents[:N_MEMORY_TYPE_TASKS] if s["memory_type"] == "internal")
    external_count = N_MEMORY_TYPE_TASKS - internal_count
    topic_counts: dict[str, int] = {}
    for t in tasks:
        if t["type"] == "topic":
            tv = t["topic_to_validate"]
            topic_counts[tv] = topic_counts.get(tv, 0) + 1

    print(f"\nTopic tasks: {len(topic_sents[:N_TOPIC_TASKS])}")
    print("  Topics covered (reflects actual data distribution):")
    for topic, n in sorted(topic_counts.items(), key=lambda x: -x[1]):
        print(f"    {n:3d}  {topic}")
    print(f"\nMemory-type tasks: {len(memory_sents[:N_MEMORY_TYPE_TASKS])}")
    print(f"  LLM says internal: {internal_count}, external: {external_count} "
          f"(not shown to annotator)")


if __name__ == "__main__":
    main()
