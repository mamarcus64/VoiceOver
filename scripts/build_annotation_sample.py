#!/usr/bin/env python3
"""
Build a stratified, interleaved ordering of video tapes for smile annotation.

Ordering guarantees:
  - At any prefix of N videos, gender ratios match the corpus proportions.
  - Within each gender, birth years are spread evenly across the range.
  - One video per subject until all subjects are exhausted ("rounds"), then second
    video per subject, etc.
  - Within each subject, tapes are ordered by smile count (most smiley first).

Output: data/annotation_sample.json
  A list of objects, each with:
    id, int_code, tape, gender, birth_year, num_smiles, youtube_url
"""

import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
META_DIR  = REPO_ROOT / "data" / "vha_metadata"
SMILE_DIR = REPO_ROOT / "data" / "smiling_segments"
MANIFEST  = REPO_ROOT / "data" / "manifest.json"
OUT_PATH  = REPO_ROOT / "data" / "annotation_sample.json"


# ── 1. Parse biographical data ────────────────────────────────────────────

bio = {}   # intcode (str) → {gender, birth_year}

for fname in os.listdir(META_DIR):
    if not fname.endswith(".xml"):
        continue
    intcode = fname.replace("intcode-", "").replace(".xml", "")
    try:
        root = ET.parse(META_DIR / fname).getroot()
        b = root.find("BiographicalInformation")
        if b is None:
            continue
        rec = {}
        for f in b.findall("format"):
            if f.get("modifier") == "Interviewee Gender":
                rec["gender"] = (f.text or "").strip()
        dob = b.find('.//created[@modifier="Interviewee Date of Birth"]')
        if dob is not None and dob.text:
            try:
                rec["birth_year"] = int(dob.text.split("/")[0])
            except ValueError:
                pass
        if rec:
            bio[intcode] = rec
    except ET.ParseError:
        pass

print(f"Bio records parsed: {len(bio)}")


# ── 2. Load per-tape smile counts ─────────────────────────────────────────

tape_smiles = {}   # tape_id (str) → int

for fname in os.listdir(SMILE_DIR):
    if not fname.endswith(".json"):
        continue
    tape_id = fname.replace(".json", "")
    with open(SMILE_DIR / fname) as fh:
        d = json.load(fh)
    tape_smiles[tape_id] = d.get("num_segments", len(d.get("segments", [])))

print(f"Tapes with smile data: {len(tape_smiles)}")


# ── 3. Load manifest for YouTube URLs ────────────────────────────────────

with open(MANIFEST) as fh:
    manifest_list = json.load(fh)

tape_url = {e["id"]: e["youtube_url"] for e in manifest_list}


# ── 4. Group tapes by subject; keep only tapes with smile data ────────────

# intcode → sorted list of (num_smiles DESC, tape_id)
subject_tapes = defaultdict(list)

for tape_id, n_smiles in tape_smiles.items():
    intcode = tape_id.split(".")[0]
    if intcode not in bio:
        continue   # no biographical metadata — skip
    subject_tapes[intcode].append((n_smiles, tape_id))

# Sort each subject's tapes: most smiles first
for intcode in subject_tapes:
    subject_tapes[intcode].sort(key=lambda x: -x[0])

print(f"Subjects with smile data and bio: {len(subject_tapes)}")


# ── 5. Build rounds (one tape per subject per round) ─────────────────────

max_rounds = max(len(v) for v in subject_tapes.values())
rounds = []   # list of lists of (intcode, tape_id, n_smiles)

for round_idx in range(max_rounds):
    round_entries = []
    for intcode, tapes in subject_tapes.items():
        if round_idx < len(tapes):
            n_smiles, tape_id = tapes[round_idx]
            round_entries.append({
                "intcode": intcode,
                "tape_id": tape_id,
                "num_smiles": n_smiles,
            })
    rounds.append(round_entries)

print(f"Rounds: {max_rounds}  (subjects per round: {[len(r) for r in rounds]})")


# ── 6. Interleave by gender × birth_year within each round ───────────────

def interleave_round(entries):
    """
    Given a list of {intcode, tape_id, num_smiles}, return them reordered so
    that gender proportions are preserved at every prefix, and within each
    gender group birth years are spread across the full range.

    Uses fractional/systematic sampling: assigns each entry a float position
    in [0, N) that keeps the two gender streams evenly interspersed.
    """
    female = [e for e in entries if bio.get(e["intcode"], {}).get("gender") == "Female"]
    male   = [e for e in entries if bio.get(e["intcode"], {}).get("gender") == "Male"]
    other  = [e for e in entries if bio.get(e["intcode"], {}).get("gender") not in ("Female", "Male")]

    # sort each gender group by birth_year (unknown → median imputed)
    known_years = [bio[e["intcode"]]["birth_year"]
                   for e in entries if "birth_year" in bio.get(e["intcode"], {})]
    median_year = sorted(known_years)[len(known_years) // 2] if known_years else 1920

    def sort_key(e):
        return bio.get(e["intcode"], {}).get("birth_year", median_year)

    female.sort(key=sort_key)
    male.sort(key=sort_key)
    other.sort(key=sort_key)

    n_total = len(female) + len(male) + len(other)
    if n_total == 0:
        return []

    def assign_positions(group, n_total):
        n = len(group)
        if n == 0:
            return []
        # place items at fractional positions so they cover [0, n_total) uniformly
        return [(item, (i + 0.5) * n_total / n) for i, item in enumerate(group)]

    positioned = (
        assign_positions(female, n_total) +
        assign_positions(male,   n_total) +
        assign_positions(other,  n_total)
    )
    positioned.sort(key=lambda x: x[1])
    return [item for item, _ in positioned]


# ── 7. Assemble final ordered list ────────────────────────────────────────

ordered = []
for round_entries in rounds:
    ordered.extend(interleave_round(round_entries))

print(f"Total entries in ordered list: {len(ordered)}")


# ── 8. Attach full metadata and write output ──────────────────────────────

output = []
for e in ordered:
    intcode  = e["intcode"]
    tape_id  = e["tape_id"]
    int_code_int, tape_int = tape_id.split(".", 1)

    b = bio.get(intcode, {})
    output.append({
        "id":          tape_id,
        "int_code":    int(int_code_int),
        "tape":        int(tape_int),
        "gender":      b.get("gender", "Unknown"),
        "birth_year":  b.get("birth_year"),
        "num_smiles":  e["num_smiles"],
        "youtube_url": tape_url.get(tape_id, ""),
    })

with open(OUT_PATH, "w") as fh:
    json.dump(output, fh, indent=2)

print(f"Written: {OUT_PATH}")

# ── 9. Quick sanity check on prefix balance ──────────────────────────────

import sys

df_output = output
total = len(df_output)
female_total = sum(1 for e in df_output if e["gender"] == "Female")
male_total   = sum(1 for e in df_output if e["gender"] == "Male")
f_frac = female_total / total
m_frac = male_total   / total

print(f"\nCorpus gender:  Female={female_total} ({f_frac:.1%})  Male={male_total} ({m_frac:.1%})")
print("\nPrefix balance check (gender Female%):")
for n in [50, 100, 200, 500, 978]:
    prefix = df_output[:n]
    pf = sum(1 for e in prefix if e["gender"] == "Female") / n
    print(f"  N={n:4d}  Female={pf:.1%}  (target {f_frac:.1%})")
