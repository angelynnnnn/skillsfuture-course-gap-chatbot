import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def skill_text(skill_item: dict) -> str:
    return " ".join([str(skill_item.get("skill", "")), str(skill_item.get("category", ""))]).strip()

def encode_job_skills(job_skills: list[dict]):
    if not job_skills:
        return None
    model = get_model()
    return model.encode([skill_text(s) for s in job_skills], normalize_embeddings=True)

def compare_job_to_course_skills(job_skills: list[dict], course_skills: list[dict], threshold: float = 0.68, job_vecs=None) -> dict:
    if not job_skills:
        return {"matched_skills": [], "missing_skills": [], "weighted_coverage": 0.0}
    weight = 1.0 / max(len(job_skills), 1)
    if not course_skills:
        return {"matched_skills": [], "missing_skills": [item["skill"] for item in job_skills], "weighted_coverage": 0.0}

    if job_vecs is None:
        job_vecs = encode_job_skills(job_skills)
    model = get_model()
    course_vecs = model.encode([skill_text(s) for s in course_skills], normalize_embeddings=True)
    sims = cosine_similarity(job_vecs, course_vecs)

    matched = []
    missing = []
    coverage = 0.0
    for i, job_skill in enumerate(job_skills):
        best_idx = int(np.argmax(sims[i]))
        best_sim = float(sims[i][best_idx])
        best_course_skill = course_skills[best_idx]
        job_name = job_skill["skill"]
        if best_sim >= threshold:
            contribution = weight
            coverage += contribution
            matched.append({"job_skill": job_name, "course_skill": best_course_skill["skill"], "similarity": round(best_sim, 3), "coverage_contribution": round(contribution * 100, 2)})
        else:
            missing.append({"job_skill": job_name, "best_course_skill": best_course_skill["skill"], "best_similarity": round(best_sim, 3)})
    return {"matched_skills": matched, "missing_skills": missing, "weighted_coverage": round(coverage * 100, 2)}

def detect_gaps_across_courses(job_skills: list[dict], course_comparisons: list[dict], min_courses_covering: int = 2) -> list[dict]:
    coverage_counts = {item["skill"]: 0 for item in job_skills}
    for comp in course_comparisons:
        for match in comp.get("matched_skills", []):
            name = match["job_skill"]
            if name in coverage_counts:
                coverage_counts[name] += 1
    gaps = []
    for item in job_skills:
        name = item["skill"]
        count = coverage_counts.get(name, 0)
        if count < min_courses_covering:
            gaps.append({"skill": name, "covered_by_courses": count})
    return gaps
