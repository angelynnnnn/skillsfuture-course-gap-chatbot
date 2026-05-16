import spacy
from spacy.matcher import PhraseMatcher
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor

_extractor = None


def get_skillner_extractor():
    global _extractor
    if _extractor is None:
        nlp = spacy.load("en_core_web_lg")
        _extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
    return _extractor


def extract_skills_with_skillner(text: str) -> list[dict]:
    extractor = get_skillner_extractor()
    annotations = extractor.annotate(text or "")
    results = annotations.get("results", {})

    skills = []
    for key in ("full_matches", "ngram_scored"):
        for item in results.get(key, []):
            name = str(item.get("doc_node_value", "")).strip()
            if not name:
                continue
            skills.append({
                "skill": name,
            })

    deduped = []
    seen = set()
    for skill in skills:
        key = skill["skill"].lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(skill)
    return deduped
