"""
Reasoning generation.

Per submission_spec.docx section 3, reasoning is checked at Stage 4 for:
specific facts, JD connection, honest concerns, no hallucination, variation
across rows, and rank-consistent tone. This module builds reasoning purely
from facts already extracted into the feature dict for that candidate, so
it cannot hallucinate (every claim is traceable to a real field), and it
naturally varies because it's built from each candidate's actual numbers.
"""


def build_reasoning(candidate: dict, fdict: dict, rank: int) -> str:
    profile = candidate.get("profile", {}) or {}
    title = profile.get("current_title", "Unknown title")
    yoe = profile.get("years_of_experience")
    yoe_str = f"{yoe:.1f} yrs" if isinstance(yoe, (int, float)) else "unknown experience"

    prod_hits = fdict["production_hits"]
    eval_hits = fdict["eval_hits"]
    bd = fdict["behavioral_detail"]
    location = profile.get("location", "unknown location")

    clauses = [f"{title} with {yoe_str}"]

    if prod_hits:
        shown = ", ".join(prod_hits[:3])
        clauses.append(f"career history shows hands-on work with {shown}")
    else:
        clauses.append("career history shows no direct retrieval/ranking system evidence")

    if eval_hits:
        clauses.append(f"mentions evaluation rigor ({', '.join(eval_hits[:2])})")

    clauses.append(f"based in {location}")

    # Behavioral honesty
    rr = bd["response_rate"]
    if rr >= 0.6:
        clauses.append(f"strong recruiter response rate ({rr:.2f})")
    elif rr <= 0.2:
        clauses.append(f"low recruiter response rate ({rr:.2f}, may be hard to reach)")
    else:
        clauses.append(f"response rate {rr:.2f}")

    if bd["days_inactive"] > 120:
        clauses.append(f"inactive for {bd['days_inactive']}d, availability uncertain")
    elif not bd["open_to_work"]:
        clauses.append("not currently flagged open to work")

    if bd["notice_days"] > 60:
        clauses.append(f"long notice period ({bd['notice_days']}d)")

    if fdict.get("disqualifier_reasons"):
        clauses.append(f"concern: {fdict['disqualifier_reasons'][0]}")

    if fdict.get("honeypot_flag_count", 0) >= 1 and fdict.get("honeypot_flag_count", 0) < 2:
        clauses.append("one minor profile-consistency flag noted, not disqualifying")

    text = "; ".join(clauses) + "."
    # Keep it to roughly 1-2 sentences as the spec asks
    return text[:400]
