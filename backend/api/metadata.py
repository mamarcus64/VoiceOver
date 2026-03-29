import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from xml.etree import ElementTree as ET

DATA_DIR = Path(os.environ.get(
    "VOICEOVER_DATA_DIR",
    Path(__file__).resolve().parent.parent.parent / "data",
))

router = APIRouter()


def _parse_xml_root(int_code: int):
    path = DATA_DIR / "vha_metadata" / f"intcode-{int_code}.xml"
    if not path.is_file():
        return None
    try:
        tree = ET.parse(path)
        return tree.getroot()
    except Exception:
        return None


def _extract_bio(root) -> dict:
    bio = root.find("BiographicalInformation")
    if bio is None:
        return {}

    result: dict = {}

    name_el = bio.find("name")
    result["name"] = name_el.text if name_el is not None else None
    result["first_name"] = name_el.get("FirstName") if name_el is not None else None
    result["last_name"] = name_el.get("LastName") if name_el is not None else None

    fmt_gender = bio.find("format[@modifier='Interviewee Gender']")
    result["gender"] = fmt_gender.text if fmt_gender is not None else None

    fmt_label = bio.find("format[@modifier='ShortFormLabel']")
    result["label"] = fmt_label.text if fmt_label is not None else None

    date_el = bio.find("date/created[@modifier='Interviewee Date of Birth']")
    result["dob"] = date_el.text if date_el is not None else None

    interview = bio.find("interview")
    questionnaire: dict = {}
    interview_refs: dict = {}
    if interview is not None:
        for resp in interview.findall("response"):
            label = resp.get("questionlabel")
            if label:
                val = resp.text
                if label in questionnaire:
                    existing = questionnaire[label]
                    if isinstance(existing, list):
                        existing.append(val)
                    else:
                        questionnaire[label] = [existing, val]
                else:
                    questionnaire[label] = val
        for ref in interview.findall("references/reference"):
            mod = ref.get("modifier")
            if mod:
                interview_refs[mod] = ref.text

    result["questionnaire"] = questionnaire
    result["interview_refs"] = interview_refs

    relations = []
    for rel in bio.findall("relations/relation"):
        relations.append({
            "modifier": rel.get("modifier"),
            "name": rel.get("FullName"),
            "piq_id": rel.get("PIQPersonID"),
        })
    result["relations"] = relations

    return result


def _timecode_to_seconds(tc: str) -> float:
    """Parse HH:MM:SS:FF timecode to seconds (assume 30fps for frames)."""
    if not tc:
        return 0.0
    parts = tc.split(":")
    try:
        if len(parts) == 4:
            h, m, s, ff = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            return h * 3600 + m * 60 + s + ff / 30.0
        elif len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            return h * 3600 + m * 60 + s
    except (ValueError, IndexError):
        pass
    return 0.0


def _extract_segments(root) -> list:
    testimony = root.find("Testimony")
    if testimony is None:
        return []
    segments = []
    for seg in testimony.findall("segments/segment"):
        in_time = seg.get("InTime", "")
        out_time = seg.get("OutTime", "")
        segments.append({
            "number": int(seg.get("Number", 0)),
            "in_time": in_time,
            "out_time": out_time,
            "in_time_sec": _timecode_to_seconds(in_time),
            "out_time_sec": _timecode_to_seconds(out_time),
            "in_file": int(seg.get("InFile", 1)),
            "out_file": int(seg.get("OutFile", 1)),
            "tape_id": seg.get("Mpeg1FileID"),
            "people": [p.text for p in seg.findall("people/person") if p.text],
            "keywords": [k.text for k in seg.findall("keywords/keyword") if k.text],
        })
    return segments


@router.get("/metadata/subjects")
async def get_subject_list():
    """Return sorted list of all subject int_codes (fast directory scan, no XML parsing)."""
    metadata_dir = DATA_DIR / "vha_metadata"
    if not metadata_dir.is_dir():
        return []
    codes = []
    for xml_file in metadata_dir.glob("intcode-*.xml"):
        try:
            codes.append(int(xml_file.stem.replace("intcode-", "")))
        except ValueError:
            pass
    codes.sort()
    return codes


@router.get("/metadata/{int_code}/tapes")
async def get_tapes_for_subject(int_code: int):
    """List available tape numbers for a subject (scans transcripts_llm directory)."""
    transcripts_dir = DATA_DIR / "transcripts_llm"
    tapes = []
    for f in transcripts_dir.glob(f"{int_code}.*.json"):
        parts = f.stem.split(".", 1)
        if len(parts) == 2:
            try:
                tapes.append(int(parts[1]))
            except ValueError:
                pass
    tapes.sort()
    return {"int_code": int_code, "tapes": tapes}


@router.get("/metadata/group-stats")
async def get_group_stats():
    """Aggregate statistics across all VHA XML metadata files. May take a few seconds."""
    metadata_dir = DATA_DIR / "vha_metadata"
    if not metadata_dir.is_dir():
        raise HTTPException(status_code=404, detail="Metadata directory not found")

    xml_files = sorted(metadata_dir.glob("intcode-*.xml"))

    gender_counts: dict = {}
    label_counts: dict = {}
    country_counts: dict = {}
    city_counts: dict = {}
    keyword_freq: dict = {}
    interview_years: dict = {}
    interviews = []

    for xml_file in xml_files:
        int_code_str = xml_file.stem.replace("intcode-", "")
        try:
            int_code = int(int_code_str)
        except ValueError:
            continue
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except Exception:
            continue

        bio = root.find("BiographicalInformation")
        if bio is None:
            continue

        name_el = bio.find("name")
        name = name_el.text if name_el is not None else None

        fmt_gender = bio.find("format[@modifier='Interviewee Gender']")
        gender = fmt_gender.text if fmt_gender is not None else "Unknown"
        gender_counts[gender] = gender_counts.get(gender, 0) + 1

        fmt_label = bio.find("format[@modifier='ShortFormLabel']")
        label = fmt_label.text if fmt_label is not None else "Unknown"
        label_counts[label] = label_counts.get(label, 0) + 1

        interview = bio.find("interview")
        country = city = interview_date = interview_length = None
        if interview is not None:
            for resp in interview.findall("response"):
                ql = resp.get("questionlabel")
                if ql == "Country of Birth" and country is None:
                    country = resp.text
                elif ql == "City of Birth" and city is None:
                    city = resp.text
            for ref in interview.findall("references/reference"):
                mod = ref.get("modifier")
                if mod == "Date of Interview":
                    interview_date = ref.text
                elif mod == "Length of Interview":
                    interview_length = ref.text

        if country:
            country_counts[country] = country_counts.get(country, 0) + 1
        if city:
            city_counts[city] = city_counts.get(city, 0) + 1
        if interview_date:
            year = interview_date[:4]
            interview_years[year] = interview_years.get(year, 0) + 1

        testimony = root.find("Testimony")
        num_segments = 0
        if testimony is not None:
            for seg in testimony.findall("segments/segment"):
                num_segments += 1
                for kw in seg.findall("keywords/keyword"):
                    if kw.text:
                        keyword_freq[kw.text] = keyword_freq.get(kw.text, 0) + 1

        interviews.append({
            "int_code": int_code,
            "name": name,
            "gender": gender,
            "label": label,
            "country_of_birth": country,
            "city_of_birth": city,
            "interview_date": interview_date,
            "interview_length": interview_length,
            "num_segments": num_segments,
        })

    interviews.sort(key=lambda x: x["int_code"])

    return {
        "total": len(interviews),
        "gender_counts": gender_counts,
        "label_counts": label_counts,
        "country_counts": dict(
            sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:30]
        ),
        "city_counts": dict(
            sorted(city_counts.items(), key=lambda x: x[1], reverse=True)[:30]
        ),
        "keyword_freq": dict(
            sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:60]
        ),
        "interview_years": dict(sorted(interview_years.items())),
        "interviews": interviews,
    }


@router.get("/metadata/{int_code}")
async def get_metadata(int_code: int):
    """Return parsed biographical + testimony segment data for a single interview."""
    root = _parse_xml_root(int_code)
    if root is None:
        raise HTTPException(
            status_code=404,
            detail=f"Metadata not found for interview code {int_code}",
        )
    bio = _extract_bio(root)
    segments = _extract_segments(root)
    return {"int_code": int_code, **bio, "segments": segments}
