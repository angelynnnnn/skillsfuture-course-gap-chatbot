import json
import os
from collections import Counter

import pandas as pd
import streamlit as st

from src.chroma_store import get_or_create_collection, chroma_results_to_records
from src.ingest_chroma import ingest_jobs, ingest_courses
from src.semantic_gap import compare_job_to_course_skills, encode_job_skills
from src.scoring import final_score, label
from src.llm_answer import generate_final_answer, template_answer


JOB_COLLECTION = "job_documents"
COURSE_COLLECTION = "course_documents"
COURSE_SKILL_COLLECTION = "course_skill_inventory"


def ensure_chroma_data():
    """
    Ensures Chroma collections exist.
    If empty, rebuilds from your ingestion script.
    """
    job_collection = get_or_create_collection(JOB_COLLECTION)
    course_collection = get_or_create_collection(COURSE_COLLECTION)
    skill_collection = get_or_create_collection(COURSE_SKILL_COLLECTION)

    if job_collection.count() == 0:
        ingest_jobs()
        job_collection = get_or_create_collection(JOB_COLLECTION)

    if course_collection.count() == 0 or skill_collection.count() == 0:
        ingest_courses()
        course_collection = get_or_create_collection(COURSE_COLLECTION)
        skill_collection = get_or_create_collection(COURSE_SKILL_COLLECTION)

    return job_collection, course_collection, skill_collection


def query_collection(collection, query_text: str, k: int):
    results = collection.query(
        query_texts=[query_text],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return chroma_results_to_records(results)


def parse_json_list(raw):
    """
    Parses JSON metadata stored in Chroma.
    Returns [] if missing or invalid.
    """
    if not raw:
        return []

    if isinstance(raw, list):
        return raw

    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    return []


def normalise_skill_item(item):
    """
    Converts skill objects/strings into:
    {"skill": "...", "importance": "...", "source_count": ...}
    """
    if isinstance(item, str):
        skill = item.strip()
        return {"skill": skill, "importance": "medium"} if skill else {}

    if isinstance(item, dict):
        skill = str(item.get("skill") or item.get("name") or "").strip()
        if not skill:
            return {}

        return {
            "skill": skill,
            "importance": str(item.get("importance") or "medium"),
            "evidence": str(item.get("evidence") or ""),
        }

    return {}


def aggregate_job_skills(retrieved_jobs, max_skills=15):
    """
    Uses the pre-extracted skills stored in job_documents metadata.

    Since each retrieved JD may list overlapping skills, this aggregates them
    and adds a source_count. This becomes the role skill profile for the query.
    """
    counts = Counter()
    examples = {}

    for job in retrieved_jobs:
        raw = job.get("extracted_job_skills")
        items = parse_json_list(raw)

        for item in items:
            skill_item = normalise_skill_item(item)
            skill = skill_item.get("skill")
            if not skill:
                continue

            key = skill.lower().strip()
            counts[key] += 1

            if key not in examples:
                examples[key] = skill_item

    if not counts:
        return []

    # Sort by frequency across retrieved JDs.
    ranked = counts.most_common(max_skills)

    results = []
    total_docs = max(len(retrieved_jobs), 1)

    for key, count in ranked:
        item = dict(examples[key])
        item["source_count"] = count
        item["source_ratio"] = round(count / total_docs, 3)

        # Simple importance heuristic.
        if count >= max(2, total_docs * 0.6):
            item["importance"] = "high"
        elif count >= 2:
            item["importance"] = "medium"
        else:
            item["importance"] = "low"

        results.append(item)

    return results


def parse_course_skills(course_record):
    """
    Parses extracted_course_skills from course_documents metadata.
    """
    items = parse_json_list(course_record.get("extracted_course_skills"))
    cleaned = []

    for item in items:
        skill_item = normalise_skill_item(item)
        if skill_item:
            cleaned.append(skill_item)

    return cleaned


def skill_query_text(skill_item):
    """
    Text used to search course_skill_inventory.
    Keep it simple because your extracted skill dataset is noisy.
    """
    skill = str(skill_item.get("skill", "")).strip()
    evidence = str(skill_item.get("evidence", "")).strip()

    if evidence:
        return f"Required job skill: {skill}\nContext: {evidence}"

    return f"Required job skill: {skill}"


def assess_catalog_coverage(
    job_skills,
    skill_collection,
    top_k_per_skill=8,
    covered_threshold=0.68,
    weak_threshold=0.55,
):
    """
    Checks JD-required skills against the FULL course_skill_inventory collection.

    This is the important part for your use case:
    - Not just top-k retrieved courses
    - Searches all extracted course skills currently in Chroma
    """
    coverage_rows = []

    for item in job_skills:
        skill = item.get("skill", "")
        if not skill:
            continue

        results = skill_collection.query(
            query_texts=[skill_query_text(item)],
            n_results=top_k_per_skill,
            include=["documents", "metadatas", "distances"],
        )
        matches = chroma_results_to_records(results)

        best_score = matches[0]["retrieval_similarity"] if matches else 0.0

        if best_score >= covered_threshold:
            status = "covered"
        elif best_score >= weak_threshold:
            status = "weakly_covered"
        else:
            status = "not_covered"

        coverage_rows.append(
            {
                "skill": skill,
                "importance": item.get("importance", "medium"),
                "source_count": item.get("source_count", 0),
                "source_ratio": item.get("source_ratio", 0),
                "coverage_status": status,
                "best_similarity": round(best_score, 3),
                "top_matches": [
                    {
                        "course_id": m.get("course_id"),
                        "course_name": m.get("course_name"),
                        "training_provider": m.get("training_provider"),
                        "matched_course_skill": m.get("skill"),
                        "similarity": m.get("retrieval_similarity"),
                        "link": m.get("link"),
                    }
                    for m in matches
                ],
            }
        )

    return coverage_rows


def coverage_to_dataframe(coverage_rows):
    rows = []
    for item in coverage_rows:
        top = item["top_matches"][0] if item.get("top_matches") else {}
        rows.append(
            {
                "skill": item.get("skill"),
                "importance": item.get("importance"),
                "source_count": item.get("source_count"),
                "coverage_status": item.get("coverage_status"),
                "best_similarity": item.get("best_similarity"),
                "best_matching_course": top.get("course_name"),
                "matched_course_skill": top.get("matched_course_skill"),
            }
        )
    return pd.DataFrame(rows)


def retrieve_course_examples(course_collection, job_context, k_courses):
    """
    Retrieves course documents to show as examples.
    This is separate from gap detection.
    Gap detection uses course_skill_inventory.
    """
    return query_collection(course_collection, job_context, k_courses)


def score_course_examples(course_examples, job_skills, skill_match_threshold):
    """
    Scores top retrieved course examples using extracted skills stored in course_documents.
    This is for ranking/display, not for full catalogue gap detection.
    """
    job_vecs = encode_job_skills(job_skills)

    course_results = []

    for course in course_examples:
        course_skills = parse_course_skills(course)

        comparison = compare_job_to_course_skills(
            job_skills=job_skills,
            course_skills=course_skills,
            threshold=skill_match_threshold,
            job_vecs=job_vecs,
        )

        score = final_score(
            skill_coverage=comparison["weighted_coverage"],
            retrieval_similarity=float(course.get("retrieval_similarity", 0)),
        )

        course_results.append(
            {
                "course_id": course.get("course_id") or course.get("id"),
                "course_name": course.get("course_name"),
                "provider": course.get("training_provider"),
                "label": label(score),
                "final_score": score,
                "retrieval_similarity": course.get("retrieval_similarity"),
                "skill_coverage": comparison["weighted_coverage"],
                "matched_skills": comparison["matched_skills"],
                "missing_skills": comparison["missing_skills"],
                "extracted_course_skills": course_skills,
                "url": course.get("link"),
            }
        )

    return sorted(course_results, key=lambda x: x["final_score"], reverse=True)


def build_job_context(retrieved_jobs):
    """
    Combines retrieved JDs for course retrieval.
    """
    parts = []

    for job in retrieved_jobs:
        title = job.get("job_title", "")
        desc = job.get("job_description") or job.get("text", "")
        skills = parse_json_list(job.get("extracted_job_skills"))
        skill_names = [normalise_skill_item(s).get("skill", "") for s in skills]
        skill_names = [s for s in skill_names if s]

        parts.append(
            "\n".join(
                [
                    f"Job title: {title}",
                    f"Job description: {desc}",
                    f"Required skills: {', '.join(skill_names)}",
                ]
            )
        )

    return "\n\n".join(parts)


def build_job_skill_result(query, job_skills):
    return {
        "role_summary": query,
        "required_skills": job_skills,
    }


def main():
    st.set_page_config(page_title="SkillsFuture Course Gap Chatbot", page_icon="🎓", layout="wide")
    st.title("🎓 SkillsFuture Course Gap Chatbot")

    st.write(
        "Ask about a job role. The chatbot retrieves related JDs, uses pre-extracted JD skills, "
        "checks those skills against the full course-skill inventory, and suggests course gaps."
    )

    with st.sidebar:
        st.header("Settings")
        st.caption(f"Active LLM provider: {os.getenv('LLM_PROVIDER', 'none')}")

        k_jobs = st.slider("Top-k job documents", 3, 30, 10)
        k_courses = st.slider("Top-k course examples to show", 3, 20, 8)
        max_skills = st.slider("Max JD skills to check", 5, 30, 15)

        with st.expander("Advanced settings"):
            covered_threshold = st.slider("Covered threshold", 0.50, 0.90, 0.68, 0.01)
            weak_threshold = st.slider("Weak coverage threshold", 0.30, 0.80, 0.55, 0.01)
            skill_match_threshold = st.slider("Course skill match threshold", 0.50, 0.90, 0.68, 0.01)
            use_llm_final_answer = st.checkbox("Use LLM for final answer if configured", value=True)

        # if st.button("Rebuild ChromaDB index"):
        #     ingest_jobs()
        #     ingest_courses()
        #     st.success("Rebuilt ChromaDB collections.")

        if st.button("Clear chat"):
            st.session_state.pop("messages", None)
            st.session_state.pop("last_run", None)

    job_collection, course_collection, skill_collection = ensure_chroma_data()

    st.caption(
        f"ChromaDB status: "
        f"{job_collection.count()} job docs, "
        f"{course_collection.count()} course docs, "
        f"{skill_collection.count()} course-skill rows."
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Ask me a role, for example: **Find course gaps for Data Engineer**.",
            }
        ]

    if "last_run" not in st.session_state:
        st.session_state["last_run"] = None

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about a job role, e.g. Find course gaps for Data Engineer")

    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status = st.status("Running course-gap pipeline...", expanded=True)

            status.write("Retrieving related job documents...")
            retrieved_jobs = query_collection(job_collection, prompt, k_jobs)
            status.write(f"Retrieved {len(retrieved_jobs)} job documents.")

            status.write("Aggregating pre-extracted JD skills...")
            job_skills = aggregate_job_skills(retrieved_jobs, max_skills=max_skills)

            if not job_skills:
                answer = (
                    "I retrieved related job documents, but I could not find pre-extracted skills "
                    "in the `extracted_job_skills` metadata. Check that `jd.csv` has `job_skill_set` "
                    "and rerun `python src/ingest_chroma.py`."
                )
                st.markdown(answer)
                st.session_state["messages"].append({"role": "assistant", "content": answer})
                status.update(label="Stopped", state="error")
                return

            status.write(f"Aggregated {len(job_skills)} JD-required skills.")

            status.write("Checking each JD skill against the full course-skill inventory...")
            coverage_rows = assess_catalog_coverage(
                job_skills=job_skills,
                skill_collection=skill_collection,
                top_k_per_skill=8,
                covered_threshold=covered_threshold,
                weak_threshold=weak_threshold,
            )

            gaps = [
                row for row in coverage_rows
                if row["coverage_status"] in ("weakly_covered", "not_covered")
            ]

            status.write("Retrieving relevant course examples...")
            job_context = build_job_context(retrieved_jobs)
            course_examples = retrieve_course_examples(course_collection, job_context, k_courses)

            status.write("Scoring retrieved course examples...")
            course_results = score_course_examples(
                course_examples=course_examples,
                job_skills=job_skills,
                skill_match_threshold=skill_match_threshold,
            )

            job_skill_result = build_job_skill_result(prompt, job_skills)

            status.write("Generating final response...")
            answer_prompt = None

            if use_llm_final_answer:
                answer, answer_err, answer_prompt = generate_final_answer(
                    prompt,
                    retrieved_jobs,
                    job_skill_result,
                    course_results,
                    gaps,
                )
                if answer_err:
                    st.info(f"Using template answer because final LLM answer failed: {answer_err}")
                    answer = template_answer(prompt, job_skill_result, course_results, gaps)
            else:
                answer = template_answer(prompt, job_skill_result, course_results, gaps)

            st.markdown(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer})

            st.session_state["last_run"] = {
                "retrieved_jobs": retrieved_jobs,
                "job_skills": job_skills,
                "coverage_rows": coverage_rows,
                "gaps": gaps,
                "course_results": course_results,
                "job_context": job_context,
                "answer_prompt": answer_prompt,
            }

            status.update(label="Pipeline complete", state="complete", expanded=False)

    if st.session_state.get("last_run"):
        last_run = st.session_state["last_run"]

        with st.expander("Latest run details", expanded=False):
            st.markdown("#### Retrieved job documents")
            st.dataframe(pd.DataFrame(last_run["retrieved_jobs"]), use_container_width=True)

            st.markdown("#### Aggregated JD-required skills")
            st.dataframe(pd.DataFrame(last_run["job_skills"]), use_container_width=True)

            st.markdown("#### Course-supply coverage by JD skill")
            st.dataframe(coverage_to_dataframe(last_run["coverage_rows"]), use_container_width=True)

            st.markdown("#### Skills flagged as gaps")
            st.dataframe(coverage_to_dataframe(last_run["gaps"]), use_container_width=True)

            st.markdown("#### Relevant course examples")
            st.dataframe(pd.DataFrame(last_run["course_results"]), use_container_width=True)

            with st.expander("Debug: combined job context used to retrieve course examples"):
                st.write(last_run["job_context"])

            if last_run.get("answer_prompt"):
                with st.expander("Debug: final answer prompt"):
                    st.code(last_run["answer_prompt"], language="text")


if __name__ == "__main__":
    main()
