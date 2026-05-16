import json
import re
from src.llm_client import call_llm

def _extract_json_object(text: str) -> dict:
    if not text:
        return {}
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}

def build_job_skill_prompt(job_context: str) -> str:
    return f"""
Extract the required skills/capabilities for the job role from the retrieved job documents below.

Return JSON only in this exact schema:
{{
  "role_summary": "short summary of the likely role",
  "required_skills": [
    {{
      "skill": "specific skill or capability",
            "category": "technical/tool/domain/soft/credential/other"
    }}
  ]
}}

Rules:
- Extract only skills supported by the text.
- Prefer specific capabilities over broad buzzwords.
- Do not invent skills.
- Merge duplicates.

Retrieved job documents:
{job_context}
""".strip()

def build_course_skill_prompt(course_name: str, course_text: str) -> str:
    return f"""
Extract the skills/capabilities taught by this course.

Return JSON only in this exact schema:
{{
  "course_name": "{course_name}",
  "skills_taught": [
    {{
      "skill": "specific skill or capability taught",
      "category": "technical/tool/domain/soft/credential/other",
      "depth": "introductory/working/advanced"
    }}
  ]
}}

Rules:
- Extract only what the course appears to teach.
- Do not infer advanced coverage unless text supports it.
- Prefer specific skills over broad marketing phrases.
- Merge duplicates.

Course description:
{course_text}
""".strip()

def _normalise_job_skill(item: dict) -> dict:
    name = str(item.get("skill") or item.get("name") or "").strip()
    if not name:
        return {}
    return {"skill": name, "category": str(item.get("category") or "unknown")}

def _normalise_course_skill(item: dict) -> dict:
    name = str(item.get("skill") or item.get("name") or "").strip()
    if not name:
        return {}
    depth = str(item.get("depth") or "introductory").lower().strip()
    if depth not in {"introductory", "working", "advanced"}:
        depth = "introductory"
    return {"skill": name, "category": str(item.get("category") or "unknown"), "depth": depth}

def extract_job_skills_with_llm(job_context: str):
    prompt = build_job_skill_prompt(job_context)
    raw, err = call_llm(prompt, json_mode=True)
    if err:
        return None, err, prompt
    parsed = _extract_json_object(raw or "")
    skills = []
    for item in parsed.get("required_skills", []):
        skill = _normalise_job_skill(item)
        if skill:
            skills.append(skill)
    return {"role_summary": parsed.get("role_summary", ""), "required_skills": skills, "raw_response": raw}, None, prompt

def extract_course_skills_with_llm(course_name: str, course_text: str):
    prompt = build_course_skill_prompt(course_name, course_text)
    raw, err = call_llm(prompt, json_mode=True)
    if err:
        return None, err, prompt
    parsed = _extract_json_object(raw or "")
    skills = []
    for item in parsed.get("skills_taught", []):
        skill = _normalise_course_skill(item)
        if skill:
            skills.append(skill)
    return {"course_name": parsed.get("course_name", course_name), "skills_taught": skills, "raw_response": raw}, None, prompt
