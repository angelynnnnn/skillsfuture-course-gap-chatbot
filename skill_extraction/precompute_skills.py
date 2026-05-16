import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skill_extraction.skillner_extractor import extract_skills_with_skillner

DATA = ROOT / "data"
JOB_SKILL_DATA = DATA / "job_skills.json"
COURSE_SKILL_DATA = DATA / "course_skills.json"
DEFAULT_WORKERS = int(os.getenv("SKILL_PRECOMPUTE_WORKERS", "10"))


def _trim_skills(raw_skills: list[dict]) -> list[dict]:
    trimmed = []
    for item in raw_skills:
        name = str(item.get("skill", "")).strip()
        if not name:
            continue
        trimmed.append({
            "skill": name,
        })
    return trimmed


def _extract_job_skill_row(idx: int, row: pd.Series) -> dict:
    job_title = str(row.get("job_title", ""))
    job_desc = str(row.get("job_description", ""))
    job_context = f"{job_title}. {job_desc}".strip()
    start_time = time.perf_counter()
    skill_result = extract_skills_with_skillner(job_context)
    elapsed = time.perf_counter() - start_time
    return {
        "job_id": str(idx),
        "job_title": job_title,
        "skills": _trim_skills(skill_result),
        "elapsed": elapsed,
    }


def precompute_job_skills(workers: int = DEFAULT_WORKERS) -> list[dict]:
    jobs = pd.read_csv(DATA / "jd.csv")
    results = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_extract_job_skill_row, idx, row): idx for idx, row in jobs.iterrows()}
        for future in as_completed(future_map):
            idx = future_map[future]
            item = future.result()
            results[idx] = {
                "job_id": item["job_id"],
                "job_title": item["job_title"],
                "skills": item["skills"],
            }
            print(f"Job {idx + 1}/{len(jobs)} extracted in {item['elapsed']:.2f}s: {item['job_title']}")
    return results


def _extract_course_skill_row(idx: int, row: pd.Series) -> dict:
    course_name = str(row.get("Course Name", ""))
    course_taught = str(row.get("What will be taught", ""))
    course_text = " ".join([course_name, course_taught]).strip()
    start_time = time.perf_counter()
    skill_result = extract_skills_with_skillner(course_text)
    elapsed = time.perf_counter() - start_time
    return {
        "course_id": str(idx),
        "course_name": course_name,
        "skills": _trim_skills(skill_result),
        "elapsed": elapsed,
    }


def precompute_course_skills(workers: int = DEFAULT_WORKERS) -> list[dict]:
    courses = pd.read_excel(DATA / "skillsfuture_courses.xlsx")
    results = [None] * len(courses)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_extract_course_skill_row, idx, row): idx for idx, row in courses.iterrows()}
        for future in as_completed(future_map):
            idx = future_map[future]
            item = future.result()
            results[idx] = {
                "course_id": item["course_id"],
                "course_name": item["course_name"],
                "skills": item["skills"],
            }
            print(f"Course {idx + 1}/{len(courses)} extracted in {item['elapsed']:.2f}s: {item['course_name']}")
    return results


def main():
    # job_skills = precompute_job_skills()
    # JOB_SKILL_DATA.write_text(json.dumps(job_skills, ensure_ascii=True, indent=2), encoding="utf-8")
    course_skills = precompute_course_skills()
    COURSE_SKILL_DATA.write_text(json.dumps(course_skills, ensure_ascii=True, indent=2), encoding="utf-8")
    # print(f"Wrote job skills to {JOB_SKILL_DATA}")
    print(f"Wrote course skills to {COURSE_SKILL_DATA}")


if __name__ == "__main__":
    main()
