import json
from src.llm_client import call_llm


def _compact_jobs(retrieved_jobs):
    return [
        {
            "job_title": j.get("job_title"),
            "similarity": j.get("retrieval_similarity"),
            "text": j.get("job_description") or j.get("text"),
            "extracted_job_skills": j.get("extracted_job_skills"),
        }
        for j in retrieved_jobs[:8]
    ]


def _compact_courses(course_results):
    return [
        {
            "course_id": c.get("course_id"),
            "course_name": c.get("course_name"),
            "provider": c.get("provider"),
            "label": c.get("label"),
            "final_score": c.get("final_score"),
            "retrieval_similarity": c.get("retrieval_similarity"),
            "skill_coverage": c.get("skill_coverage"),
            "matched_skills": c.get("matched_skills"),
            "url": c.get("url"),
        }
        for c in course_results[:10]
    ]


def _compact_gaps(gaps):
    return [
        {
            "skill": g.get("skill"),
            "importance": g.get("importance"),
            "source_count": g.get("source_count"),
            "coverage_status": g.get("coverage_status"),
        }
        for g in gaps[:15]
    ]


def build_final_answer_prompt(query, retrieved_jobs, job_skill_result, course_results, gaps):
    payload = {
        "user_query": query,
        "retrieved_job_documents": _compact_jobs(retrieved_jobs),
        "aggregated_job_required_skills": job_skill_result,
        "relevant_course_examples": _compact_courses(course_results),
        "course_supply_gaps": _compact_gaps(gaps),
    }

    return f"""
You are generating a final answer for a SkillsFuture policy-planning chatbot.

Your task:
Explain whether the current SkillsFuture course supply covers the skills required by the retrieved job descriptions.

Use ONLY the JSON data provided.
Do NOT give general career advice.
Do NOT recommend external websites.
Do NOT mention students, learners, ages, or educational backgrounds.
Do NOT add resources.
Do NOT add an introduction.
Do NOT add a conclusion.
Do NOT use headings other than the four headings below.

Follow this response template exactly. Use the same four headings, in the same order. Replace the bracketed placeholders with information from the provided JSON data only. Do not add an introduction, conclusion, extra resources, or personal career advice.
Generic response template:

### 1. Key skills required by the retrieved JDs
- **[Skill name]** — [high/medium/low] importance; [brief reason based on retrieved JD frequency or evidence].
- **[Skill name]** — [high/medium/low] importance; [brief reason based on retrieved JD frequency or evidence].
- **[Skill name]** — [high/medium/low] importance; [brief reason based on retrieved JD frequency or evidence].

### 2. Relevant available courses and whether they are strong/partial/weak matches
- **[Course name]** — [strong/partial/weak/relevant course example]; [brief reason based on matched skills or similarity evidence].
- **[Course name]** — [strong/partial/weak/relevant course example]; [brief reason based on matched skills or similarity evidence].
- **[Course name]** — [strong/partial/weak/relevant course example]; [brief reason based on matched skills or similarity evidence].

### 3. Skills weakly covered or not clearly covered
- **[Skill name]** — [weakly covered/not clearly covered]; [brief reason based on closest course-skill match or similarity evidence].
- **[Skill name]** — [weakly covered/not clearly covered]; [brief reason based on closest course-skill match or similarity evidence].
- **[Skill name]** — [weakly covered/not clearly covered]; [brief reason based on closest course-skill match or similarity evidence].

### 4. Recommended course areas SkillsFuture could consider adding or strengthening
- Strengthen courses on **[skill area]**, especially [specific module/topic based on the gap].
- Add or expand modules on **[skill area]** to address skills not clearly covered in the current course-skill inventory.


Example response format:
### 1. Key skills required by the retrieved JDs
- **SQL** — high importance; appears frequently across the retrieved job descriptions.
- **Python** — high importance; commonly required for data processing and automation tasks.
- **ETL pipelines** — high importance; needed for building and maintaining data workflows.
- **Cloud data platforms** — medium importance; appears in several retrieved job descriptions.
- **Workflow orchestration** — medium importance; required for managing scheduled data pipelines.

### 2. Relevant available courses and whether they are strong/partial/weak matches
- **Cloud Data Engineering** — relevant course example; appears aligned with ETL pipelines and cloud data platform skills.
- **Python and SQL Foundations** — relevant course example; appears aligned with Python and SQL foundation skills.
- **Big Data Processing with Spark** — relevant course example; appears aligned with distributed data processing skills.

### 3. Skills weakly covered or not clearly covered
- **Workflow orchestration** — weakly covered; the closest course-skill match is related but not a strong direct match.
- **Data quality monitoring** — not clearly covered; no strong matching course skill was found in the current course-skill inventory.
- **Data observability** — not clearly covered; current course matches appear weak based on semantic similarity.

### 4. Recommended course areas SkillsFuture could consider adding or strengthening
- Strengthen courses on **workflow orchestration**, including tools and practices for scheduling and monitoring data pipelines.
- Add or expand modules on **data quality monitoring** and **data observability** for production data systems.


NOTE: You are a SkillsFuture policy-planning chatbot assistant, not a career advisor. Focus your answer on the course-skill coverage analysis based on the provided JSON data. Do not give general career advice or recommend external resources.

Input JSON:
{json.dumps(payload, indent=2)}
""".strip()


def generate_final_answer(query, retrieved_jobs, job_skill_result, course_results, gaps):
    prompt = build_final_answer_prompt(query, retrieved_jobs, job_skill_result, course_results, gaps)
    answer, err = call_llm(prompt)
    return answer, err, prompt


def template_answer(query, job_skill_result, course_results, gaps):
    lines = []

    role_summary = job_skill_result.get("role_summary") or query

    lines.append("### Interpreted role")
    lines.append(f"The retrieved documents suggest the query is about **{role_summary}**.")
    lines.append("")

    lines.append("### Key required skills from retrieved JDs")
    for item in job_skill_result.get("required_skills", [])[:12]:
        count = item.get("source_count", 0)
        importance = item.get("importance", "medium")
        lines.append(f"- **{item['skill']}** — {importance}, found in {count} retrieved JD(s)")
    lines.append("")

    lines.append("### Relevant available courses")
    if course_results:
        for c in course_results[:8]:
            matched = ", ".join([m.get("job_skill", "") for m in c.get("matched_skills", [])])
            matched = matched or "no strong skill coverage detected"
            lines.append(
                f"- **{c.get('course_name')}** — **{c.get('label')} match**, "
                f"score {c.get('final_score')}/100. Covers: {matched}."
            )
    else:
        lines.append("- No relevant course examples were retrieved.")
    lines.append("")

    lines.append("### Course-supply gaps")
    if gaps:
        for g in gaps[:8]:
            status = g.get("coverage_status", "gap")
            best = g.get("best_similarity", 0)
            lines.append(
                f"- **{g.get('skill')}** — {status.replace('_', ' ')}, "
                f"best course-skill similarity {best}"
            )
    else:
        lines.append("- No major weakly covered or not-covered skill gaps were detected.")
    lines.append("")

    lines.append("### Suggested course improvements")
    if gaps:
        gap_names = ", ".join([g.get("skill", "") for g in gaps[:5]])
        lines.append(
            f"SkillsFuture could consider adding or strengthening courses/modules covering **{gap_names}**."
        )
    else:
        lines.append(
            "The current course-skill inventory appears to cover the main extracted skills, "
            "but course depth and learner outcomes should still be validated."
        )

    return "\n".join(lines)
