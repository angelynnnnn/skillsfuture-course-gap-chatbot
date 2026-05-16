def final_score(skill_coverage, retrieval_similarity):
    sim_score = max(0, min(float(retrieval_similarity) * 100, 100))
    return round(0.80 * skill_coverage + 0.20 * sim_score, 2)

def label(score):
    if score >= 80:
        return "Strong"
    if score >= 50:
        return "Partial"
    return "Weak"
