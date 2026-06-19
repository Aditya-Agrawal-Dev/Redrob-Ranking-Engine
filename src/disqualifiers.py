"""
Disqualifier detection — implements job_description.docx's explicit
"Things we explicitly do NOT want" and "disqualifiers we actually apply"
sections. Returns a multiplicative penalty in (0, 1], not a hard exclusion,
because the JD itself hedges most of these with "we will probably not move
forward" rather than an absolute rule, and a single weak match (e.g. one
consulting stint years ago) shouldn't zero out an otherwise strong career.
"""
import re
from . import jd_config as cfg


def _text_blob(candidate: dict) -> str:
    profile = candidate.get("profile", {}) or {}
    parts = [profile.get("headline", ""), profile.get("summary", "")]
    for c in candidate.get("career_history", []) or []:
        parts.append(c.get("title", ""))
        parts.append(c.get("description", ""))
    return " ".join(parts).lower()


def _career_industries(candidate: dict) -> list[str]:
    industries = [(candidate.get("profile", {}) or {}).get("current_industry", "")]
    for c in candidate.get("career_history", []) or []:
        industries.append(c.get("industry", ""))
    return [i.lower() for i in industries if i]


def _career_companies(candidate: dict) -> list[str]:
    companies = [(candidate.get("profile", {}) or {}).get("current_company", "")]
    for c in candidate.get("career_history", []) or []:
        companies.append(c.get("company", ""))
    return [c.lower() for c in companies if c]


def disqualifier_penalty(candidate: dict) -> tuple[float, list[str]]:
    """Returns (multiplier in (0,1], list of reasons applied)."""
    blob = _text_blob(candidate)
    penalty = 1.0
    reasons = []

    # --- Pure research/academia, no production deployment evidence ---
    industries = _career_industries(candidate)
    if industries and all(any(r in ind for r in cfg.RESEARCH_ONLY_INDUSTRIES) for ind in industries):
        if not any(t in blob for t in cfg.PRODUCTION_EVIDENCE_TERMS):
            penalty *= 0.25
            reasons.append("pure_research_no_production_evidence")

    # --- Consulting-only career (no product company experience at all) ---
    companies = _career_companies(candidate)
    if companies and all(any(cc in comp for cc in cfg.CONSULTING_COMPANIES) for comp in companies):
        penalty *= 0.35
        reasons.append("consulting_only_career")

    # --- CV/speech/robotics-heavy with no NLP/IR exposure ---
    cv_hits = sum(1 for t in cfg.CV_SPEECH_ROBOTICS_TERMS if t in blob)
    nlp_hits = sum(1 for t in cfg.NLP_IR_TERMS if t in blob)
    if cv_hits >= 2 and nlp_hits == 0:
        penalty *= 0.4
        reasons.append("cv_speech_robotics_no_nlp_ir")

    # --- Title-chasing: 3+ short stints (<18mo) with escalating seniority ---
    career = sorted(
        (c for c in candidate.get("career_history", []) or [] if c.get("start_date")),
        key=lambda c: c.get("start_date", ""),
    )
    short_escalating = 0
    seniority_words = ["junior", "senior", "staff", "principal", "lead"]
    prev_level = -1
    for c in career:
        dur = c.get("duration_months", 999) or 999
        title = (c.get("title") or "").lower()
        level = max((i for i, w in enumerate(seniority_words) if w in title), default=-1)
        if dur < 18 and level > prev_level and level >= 1:
            short_escalating += 1
        if level >= 0:
            prev_level = level
    if short_escalating >= 3:
        penalty *= 0.6
        reasons.append("title_chasing_pattern")

    # --- Senior, not coding recently (architect/lead/manager titles, no
    #     coding evidence in most recent role) ---
    if career:
        latest = career[-1]
        latest_title = (latest.get("title") or "").lower()
        latest_desc = (latest.get("description") or "").lower()
        is_noncoding_senior = any(re.search(p, latest_title) for p in cfg.NON_CODING_SENIOR_TITLES)
        dur = latest.get("duration_months", 0) or 0
        coding_evidence = any(t in latest_desc for t in cfg.PRODUCTION_RETRIEVAL_TERMS) or "code" in latest_desc or "wrote" in latest_desc
        if is_noncoding_senior and dur >= 18 and not coding_evidence:
            penalty *= 0.5
            reasons.append("senior_non_coding_role_18mo_plus")

    return penalty, reasons
