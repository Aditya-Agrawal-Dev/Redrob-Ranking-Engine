"""
Per-candidate feature extraction (excluding text similarity, which is
computed in bulk via TF-IDF in scoring.py since it needs the full corpus).
"""
import re
import math
from datetime import date
from . import jd_config as cfg


def _all_titles(candidate: dict) -> list[str]:
    profile = candidate.get("profile", {}) or {}
    titles = [profile.get("current_title", "")]
    titles += [c.get("title", "") for c in candidate.get("career_history", []) or []]
    return [t.lower() for t in titles if t]


def title_relevance(candidate: dict) -> float:
    """1.0 if current title matches, 0.6 if only a past title matches, else 0."""
    profile = candidate.get("profile", {}) or {}
    current = (profile.get("current_title") or "").lower()
    past_titles = [c.get("title", "").lower() for c in candidate.get("career_history", []) or []]

    current_match = any(re.search(p, current) for p in cfg.RELEVANT_TITLE_PATTERNS)
    if current_match:
        return 1.0
    past_match = any(
        re.search(p, t) for t in past_titles for p in cfg.RELEVANT_TITLE_PATTERNS
    )
    return 0.6 if past_match else 0.0


def _career_description_blob(candidate: dict) -> str:
    return " ".join(
        (c.get("description") or "") for c in candidate.get("career_history", []) or []
    ).lower()


def production_evidence(candidate: dict) -> tuple[float, list[str]]:
    """Fraction of production-retrieval terms present in career descriptions
    (NOT skills list), capped/scaled. Returns (score 0-1, matched terms)."""
    blob = _career_description_blob(candidate)
    hits = [t for t in cfg.PRODUCTION_RETRIEVAL_TERMS if t in blob]
    # Scale: 4+ distinct hits = full credit. Diminishing returns beyond that.
    score = min(1.0, len(hits) / 4)
    return score, hits


def eval_evidence(candidate: dict) -> tuple[float, list[str]]:
    blob = _career_description_blob(candidate)
    hits = [t for t in cfg.EVAL_FRAMEWORK_TERMS if t in blob]
    score = min(1.0, len(hits) / 2)
    return score, hits


def experience_fit(candidate: dict) -> float:
    yoe = (candidate.get("profile", {}) or {}).get("years_of_experience")
    if not isinstance(yoe, (int, float)):
        return 0.3
    lo, hi = cfg.EXPERIENCE_BAND_CENTER
    if lo <= yoe <= hi:
        return 1.0
    if yoe < lo:
        # Soft falloff below band down to soft_min
        span = max(lo - cfg.EXPERIENCE_SOFT_MIN, 0.1)
        return max(0.0, 1.0 - (lo - yoe) / span * 0.8)
    # yoe > hi: soft falloff above band (overqualified risk, JD doesn't
    # forbid it but ideal candidate is 6-8 yrs)
    span = max(cfg.EXPERIENCE_SOFT_MAX - hi, 0.1)
    return max(0.2, 1.0 - (yoe - hi) / span * 0.6)


def location_fit(candidate: dict) -> float:
    profile = candidate.get("profile", {}) or {}
    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    willing = (candidate.get("redrob_signals", {}) or {}).get("willing_to_relocate", False)

    if any(loc in location for loc in cfg.PRIMARY_LOCATIONS):
        return 1.0
    if any(loc in location for loc in cfg.TIER1_INDIA_LOCATIONS):
        return 0.85
    if country == "india":
        return 0.65 if willing else 0.45
    # Outside India: JD says case-by-case, no visa sponsorship
    return 0.35 if willing else 0.15


def _days_since(date_str: str) -> float:
    try:
        d = date.fromisoformat(date_str[:10])
        return (date.today() - d).days
    except (ValueError, TypeError, AttributeError):
        return 9999


def behavioral_modifier(candidate: dict) -> tuple[float, dict]:
    """Multiplicative modifier in roughly [0.45, 1.15]. Combines recruiter
    response rate, recency of activity, interview completion, profile
    completeness, open-to-work flag, and notice period."""
    sig = candidate.get("redrob_signals", {}) or {}

    response_rate = sig.get("recruiter_response_rate", 0) or 0
    interview_completion = sig.get("interview_completion_rate", 0) or 0
    completeness = (sig.get("profile_completeness_score", 0) or 0) / 100
    open_to_work = bool(sig.get("open_to_work_flag", False))
    notice_days = sig.get("notice_period_days", 60) or 60
    days_inactive = _days_since(sig.get("last_active_date", ""))

    # Recency: full credit if active in last 30 days, decaying to floor by 180+
    recency_score = max(0.0, 1.0 - max(0, days_inactive - 30) / 150)

    # Notice period: ideal <=30 days, linear penalty up to 90
    notice_score = max(0.3, 1.0 - max(0, notice_days - cfg.NOTICE_PERIOD_IDEAL_DAYS) / 90)

    raw = (
        0.30 * response_rate
        + 0.20 * recency_score
        + 0.15 * interview_completion
        + 0.15 * completeness
        + 0.10 * (1.0 if open_to_work else 0.3)
        + 0.10 * notice_score
    )
    # Map raw (0-1) to modifier range [0.45, 1.15]
    modifier = 0.45 + raw * 0.70
    detail = {
        "response_rate": response_rate,
        "recency_score": round(recency_score, 2),
        "interview_completion": interview_completion,
        "open_to_work": open_to_work,
        "notice_days": notice_days,
        "days_inactive": days_inactive,
    }
    return modifier, detail
