import ast
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chroma_store import reset_collection

DATA = ROOT / "data"

JOB_COLLECTION = "job_documents"
COURSE_COLLECTION = "course_documents"
COURSE_SKILL_COLLECTION = "course_skill_inventory"

JOB_SKILL_DATA = DATA / "job_skills.json"
COURSE_SKILL_DATA = DATA / "course_skills.json"


def safe_str(value: Any) -> str:
    """Safely convert None/NaN values to empty strings."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def parse_skill_list(value: Any) -> list:
    """
    Parses skills from multiple formats:
    - JSON list string: ["Python", "SQL"]
    - Python list string: ['Python', 'SQL']
    - Actual list: ["Python", "SQL"]
    - Comma-separated string: Python, SQL
    """
    if value is None:
        return []

    if isinstance(value, list):
        return value

    text = safe_str(value)
    if not text or text.lower() == "nan":
        return []

    # Valid JSON list
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    # Python-style list string with single quotes
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    # Fallback: comma separated
    return [part.strip() for part in text.split(",") if part.strip()]


def normalise_skill_item(item: Any) -> dict:
    """
    Converts a skill into a consistent format:
    {"skill": "..."}
    """
    if isinstance(item, str):
        skill = item.strip()
        return {"skill": skill} if skill else {}

    if isinstance(item, dict):
        skill = safe_str(item.get("skill") or item.get("name"))
        return {"skill": skill} if skill else {}

    return {}


def load_json_list(path: Path) -> list:
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # JSONL support
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    data = json.loads(text)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["courses", "jobs", "data", "items"]:
            if isinstance(data.get(key), list):
                return data[key]

    return []


def load_job_skills_lookup() -> dict:
    """
    Optional fallback if you also have data/job_skills.json.

    Expected format:
    [
        {"job_id": "0", "skills": [...]}
    ]
    """
    rows = load_json_list(JOB_SKILL_DATA)
    lookup = {}

    for item in rows:
        job_id = safe_str(item.get("job_id"))
        if job_id:
            lookup[job_id] = item.get("skills", [])

    return lookup


def load_course_skills_lookup() -> dict:
    """
    Loads data/course_skills.json.

    Expected format:
    [
        {
            "course_id": "0",
            "course_name": "...",
            "skills": [
                {"skill": "..."}
            ]
        }
    ]
    """
    rows = load_json_list(COURSE_SKILL_DATA)
    lookup = {}

    for item in rows:
        course_id = safe_str(item.get("course_id"))
        if course_id:
            lookup[course_id] = item.get("skills") or item.get("skills_taught") or []

    return lookup


def add_in_batches(collection, ids: list[str], documents: list[str], metadatas: list[dict], batch_size: int = 500):
    """
    Safer than adding everything at once.
    """
    if not ids:
        return

    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )


def ingest_jobs(batch_size: int = 500):
    jobs_path = DATA / "jd.csv"
    if not jobs_path.exists():
        raise FileNotFoundError(f"Missing {jobs_path}")

    jobs = pd.read_csv(jobs_path)
    collection = reset_collection(JOB_COLLECTION)

    job_skills_lookup = load_job_skills_lookup()

    ids = []
    documents = []
    metadatas = []

    for idx, row in jobs.iterrows():
        job_id = safe_str(row.get("job_id")) or str(idx)
        job_title = safe_str(row.get("job_title"))
        category = safe_str(row.get("category"))
        job_description = safe_str(row.get("job_description"))

        # Main fix: use robust parser instead of json.loads only.
        parsed_skills = parse_skill_list(row.get("job_skill_set"))

        # Optional fallback.
        if not parsed_skills:
            parsed_skills = job_skills_lookup.get(job_id, []) or job_skills_lookup.get(str(idx), [])

        extracted_skills = []
        for item in parsed_skills:
            skill_item = normalise_skill_item(item)
            if skill_item:
                extracted_skills.append(skill_item)

        skill_names = [item["skill"] for item in extracted_skills]

        # Use idx for unique Chroma ID. Store original job_id in metadata.
        ids.append(f"job_{idx}")

        documents.append(
            "\n".join(
                [
                    f"Job title: {job_title}",
                    f"Category: {category}",
                    f"Job description: {job_description}",
                    f"Extracted required skills: {', '.join(skill_names)}",
                ]
            )
        )

        metadatas.append(
            {
                "job_id": job_id,
                "row_index": str(idx),
                "job_title": job_title,
                "category": category,
                "job_description": job_description[:8000],
                "extracted_job_skills": json.dumps(extracted_skills, ensure_ascii=False),
            }
        )

    add_in_batches(collection, ids, documents, metadatas, batch_size=batch_size)
    print(f"Ingested {len(ids)} job documents into '{JOB_COLLECTION}'.")


def ingest_courses(batch_size: int = 500):
    courses_path = DATA / "skillsfuture_courses.xlsx"
    if not courses_path.exists():
        raise FileNotFoundError(f"Missing {courses_path}")

    if not COURSE_SKILL_DATA.exists():
        raise FileNotFoundError(
            f"Missing {COURSE_SKILL_DATA}. Save your extracted course skills as data/course_skills.json."
        )

    courses = pd.read_excel(courses_path)

    course_collection = reset_collection(COURSE_COLLECTION)
    skill_collection = reset_collection(COURSE_SKILL_COLLECTION)

    course_skills_lookup = load_course_skills_lookup()

    course_ids = []
    course_docs = []
    course_metas = []

    skill_ids = []
    skill_docs = []
    skill_metas = []

    for idx, row in courses.iterrows():
        # Your course_skills.json uses dataframe index as course_id: "0", "1", ...
        course_id = str(idx)

        course_name = safe_str(row.get("Course Name"))
        training_provider = safe_str(row.get("Training Provider"))
        course_overview = safe_str(row.get("Course overview"))
        what_will_be_taught = safe_str(row.get("What will be taught"))
        skills_you_pick_up = safe_str(row.get("Skills you will pick up"))
        entry_requirement = safe_str(row.get("Entry Requirement"))
        link = safe_str(row.get("Link"))

        raw_course_skills = course_skills_lookup.get(course_id, [])

        extracted_course_skills = []
        for item in raw_course_skills:
            skill_item = normalise_skill_item(item)
            if skill_item:
                extracted_course_skills.append(skill_item)

        skill_names = [item["skill"] for item in extracted_course_skills]

        # Collection 1: full course documents for course retrieval.
        course_ids.append(f"course_{course_id}")
        course_docs.append(
            "\n".join(
                [
                    f"Course name: {course_name}",
                    f"Training provider: {training_provider}",
                    f"Course overview: {course_overview}",
                    f"What will be taught: {what_will_be_taught}",
                    f"Skills you will pick up: {skills_you_pick_up}",
                    f"Entry requirement: {entry_requirement}",
                    f"Extracted skills taught: {', '.join(skill_names)}",
                ]
            )
        )
        course_metas.append(
            {
                "course_id": course_id,
                "course_name": course_name,
                "training_provider": training_provider,
                "course_overview": course_overview[:8000],
                "what_will_be_taught": what_will_be_taught[:8000],
                "skills": skills_you_pick_up,
                "entry_requirement": entry_requirement,
                "link": link,
                "extracted_course_skills": json.dumps(extracted_course_skills, ensure_ascii=False),
            }
        )

        # Collection 2: one row per extracted course skill for whole-catalogue gap detection.
        for skill_idx, skill_item in enumerate(extracted_course_skills):
            skill = skill_item.get("skill", "")
            if not skill:
                continue

            skill_ids.append(f"course_{course_id}_skill_{skill_idx}")

            # Add course context because your skill extraction is currently noisy.
            skill_docs.append(
                "\n".join(
                    [
                        f"Course name: {course_name}",
                        f"Training provider: {training_provider}",
                        f"Skill taught: {skill}",
                        f"Course overview: {course_overview[:500]}",
                        f"What will be taught: {what_will_be_taught[:500]}",
                    ]
                )
            )

            skill_metas.append(
                {
                    "course_id": course_id,
                    "course_name": course_name,
                    "training_provider": training_provider,
                    "skill": skill,
                    "link": link,
                }
            )

    add_in_batches(course_collection, course_ids, course_docs, course_metas, batch_size=batch_size)
    add_in_batches(skill_collection, skill_ids, skill_docs, skill_metas, batch_size=batch_size)

    print(f"Ingested {len(course_ids)} courses into '{COURSE_COLLECTION}'.")
    print(f"Ingested {len(skill_ids)} course skills into '{COURSE_SKILL_COLLECTION}'.")


def main():
    ingest_jobs()
    ingest_courses()
    print("ChromaDB ingestion complete.")


if __name__ == "__main__":
    main()
