"""
Honeypot detection.

Per submission_spec.docx section 7: ~80 honeypot candidates have "subtly
impossible profiles" — e.g. 8 years of experience at a company founded 3
years ago (we don't have company-founding dates, so we can't check that
directly), or "expert" proficiency in 10 skills with 0 years used (we CAN
check that). We detect the checkable anomaly classes and flag candidates
with multiple co-occurring anomalies, since any single anomaly could be
noisy/legitimate (e.g. a fast learner who's genuinely expert quickly).

Design choice: this returns a continuous suspicion score (0+) rather than a
hard boolean, and scoring.py applies a steep penalty above a flag-count
threshold. This avoids accidentally nuking a real candidate on one noisy
field while still being aggressive against clear stacking of anomalies.
"""
from datetime import date


def _parse_date(s):
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except (ValueError, TypeError):
        return None


def honeypot_flags(candidate: dict) -> list[str]:
    """Return a list of anomaly flag names. Empty list = no anomalies found."""
    flags = []

    profile = candidate.get("profile", {}) or {}
    career = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    yoe = profile.get("years_of_experience")

    # --- Flag 1: skills claimed "expert"/"advanced" with near-zero duration ---
    overconfident_skills = [
        s for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and isinstance(s.get("duration_months"), (int, float))
        and s.get("duration_months", 999) < 3
    ]
    if len(overconfident_skills) >= 2:
        flags.append(f"expert_skill_zero_duration:{len(overconfident_skills)}")

    # --- Flag 2: implausibly many "expert" skills with low average tenure ---
    expert_skills = [s for s in skills if s.get("proficiency") == "expert"]
    if len(expert_skills) >= 6:
        durations = [s.get("duration_months", 0) for s in expert_skills
                     if isinstance(s.get("duration_months"), (int, float))]
        if durations and (sum(durations) / len(durations)) < 6:
            flags.append(f"mass_expert_low_tenure:{len(expert_skills)}")

    # --- Flag 3: career_history total duration grossly exceeds stated YOE ---
    if isinstance(yoe, (int, float)) and yoe > 0 and career:
        total_months = sum(
            c.get("duration_months", 0) for c in career
            if isinstance(c.get("duration_months"), (int, float))
        )
        if total_months > (yoe * 12) * 1.35:
            flags.append(f"career_months_exceed_yoe:{total_months}vs{yoe*12:.0f}")

    # --- Flag 4: malformed date ranges (end before start, or current role
    #     with a non-null end_date far in the past) ---
    for c in career:
        sd = _parse_date(c.get("start_date"))
        ed = _parse_date(c.get("end_date"))
        if sd and ed and ed < sd:
            flags.append(f"end_before_start:{c.get('company')}")
        if c.get("is_current") and ed is not None:
            flags.append(f"current_role_has_end_date:{c.get('company')}")

    # --- Flag 5: overlapping full-time roles that, summed, imply >24
    #     months of simultaneous full-time work in some window ---
    intervals = []
    for c in career:
        sd = _parse_date(c.get("start_date"))
        ed = _parse_date(c.get("end_date")) or date.today()
        if sd:
            intervals.append((sd, ed))
    intervals.sort()
    overlap_months = 0
    for i in range(len(intervals) - 1):
        cur_end = intervals[i][1]
        nxt_start = intervals[i + 1][0]
        if nxt_start < cur_end:
            overlap_days = (cur_end - nxt_start).days
            overlap_months += overlap_days / 30
    if overlap_months > 18:
        flags.append(f"overlapping_roles:{overlap_months:.0f}mo")

    # --- Flag 6: years_of_experience inconsistent with earliest career start
    #     (e.g. claims 8 yrs but earliest job started 2 years ago) ---
    if isinstance(yoe, (int, float)) and intervals:
        earliest_start = min(s for s, _ in intervals)
        years_since_earliest = (date.today() - earliest_start).days / 365.25
        if yoe > years_since_earliest + 1.5:
            flags.append(f"yoe_exceeds_career_span:{yoe}vs{years_since_earliest:.1f}")

    return flags


# Severity weight per flag prefix. A single high-severity anomaly (e.g. a
# career timeline that's arithmetically impossible) is, on its own, as
# strong a signal as two independent low-severity anomalies co-occurring.
_SEVERITY = {
    "expert_skill_zero_duration": 2,
    "mass_expert_low_tenure": 1,
    "career_months_exceed_yoe": 2,
    "end_before_start": 2,
    "current_role_has_end_date": 1,
    "overlapping_roles": 2,
    "yoe_exceeds_career_span": 2,
}


def honeypot_severity(candidate: dict) -> tuple[int, list[str]]:
    flags = honeypot_flags(candidate)
    score = sum(_SEVERITY.get(f.split(":")[0], 1) for f in flags)
    return score, flags


def is_honeypot_suspect(candidate: dict, threshold: int = 2) -> bool:
    score, _ = honeypot_severity(candidate)
    return score >= threshold
