"""
Unit tests on synthetic profiles designed to exercise the exact trap
classes the challenge describes: keyword stuffers, hidden strong
candidates, behavioral twins, and honeypots.

Run: python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scoring import score_candidates
from src.honeypot import honeypot_flags, is_honeypot_suspect


def _base_candidate(cid, **overrides):
    c = {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": "Test Person",
            "headline": "Test headline",
            "summary": "Test summary",
            "location": "Pune, Maharashtra",
            "country": "India",
            "years_of_experience": 6.0,
            "current_title": "Software Engineer",
            "current_company": "TestCo",
            "current_company_size": "201-500",
            "current_industry": "Software",
        },
        "career_history": [{
            "company": "TestCo", "title": "Software Engineer",
            "start_date": "2020-01-01", "end_date": None,
            "duration_months": 60, "is_current": True,
            "industry": "Software", "company_size": "201-500",
            "description": "Built software.",
        }],
        "education": [],
        "skills": [],
        "redrob_signals": {
            "profile_completeness_score": 80, "signup_date": "2025-01-01",
            "last_active_date": "2026-06-01", "open_to_work_flag": True,
            "profile_views_received_30d": 10, "applications_submitted_30d": 1,
            "recruiter_response_rate": 0.5, "avg_response_time_hours": 24,
            "skill_assessment_scores": {}, "connection_count": 100,
            "endorsements_received": 10, "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 20, "max": 30},
            "preferred_work_mode": "hybrid", "willing_to_relocate": True,
            "github_activity_score": 50, "search_appearance_30d": 50,
            "saved_by_recruiters_30d": 5, "interview_completion_rate": 0.8,
            "offer_acceptance_rate": 0.5, "verified_email": True,
            "verified_phone": True, "linkedin_connected": True,
        },
    }
    for k, v in overrides.items():
        c[k] = v
    return c


def test_keyword_stuffer_ranks_below_genuine_fit():
    stuffer = _base_candidate("CAND_0000001", profile={
        **_base_candidate("x")["profile"],
        "current_title": "Marketing Manager",
    }, skills=[
        {"name": s, "proficiency": "expert", "endorsements": 5, "duration_months": 24}
        for s in ["FAISS", "Pinecone", "Embeddings", "NDCG", "RAG", "Vector Search"]
    ])
    genuine = _base_candidate("CAND_0000002", profile={
        **_base_candidate("x")["profile"],
        "current_title": "Senior ML Engineer",
        "years_of_experience": 7.0,
    }, career_history=[{
        "company": "TechCo", "title": "Senior ML Engineer",
        "start_date": "2019-01-01", "end_date": None,
        "duration_months": 84, "is_current": True,
        "industry": "Software", "company_size": "1001-5000",
        "description": (
            "Built a production embedding-based retrieval system using BGE "
            "and FAISS serving real users at scale. Designed evaluation "
            "framework using NDCG and ran A/B tests for ranking quality."
        ),
    }])
    results = score_candidates([stuffer, genuine])
    by_id = {r["candidate_id"]: r["score"] for r in results}
    assert by_id["CAND_0000002"] > by_id["CAND_0000001"], (
        "Genuine production ML engineer must outrank a keyword-stuffed "
        "Marketing Manager."
    )


def test_hidden_strong_candidate_not_penalized_for_missing_buzzwords():
    plain = _base_candidate("CAND_0000003", profile={
        **_base_candidate("x")["profile"],
        "current_title": "Backend Engineer",
        "years_of_experience": 6.5,
    }, career_history=[{
        "company": "ShopCo", "title": "Backend Engineer",
        "start_date": "2019-01-01", "end_date": None,
        "duration_months": 78, "is_current": True,
        "industry": "E-commerce", "company_size": "1001-5000",
        "description": (
            "Built the recommendation system that powers our homepage "
            "product feed for 2M daily users, including the ranking model "
            "and the candidate retrieval index."
        ),
    }])
    irrelevant = _base_candidate("CAND_0000004", profile={
        **_base_candidate("x")["profile"],
        "current_title": "Accountant",
        "years_of_experience": 6.5,
    })
    results = score_candidates([plain, irrelevant])
    by_id = {r["candidate_id"]: r["score"] for r in results}
    assert by_id["CAND_0000003"] > by_id["CAND_0000004"]


def test_behavioral_twin_inactive_ranks_lower():
    active = _base_candidate("CAND_0000005")
    inactive = _base_candidate("CAND_0000006", redrob_signals={
        **_base_candidate("x")["redrob_signals"],
        "last_active_date": "2025-01-01",
        "recruiter_response_rate": 0.02,
        "open_to_work_flag": False,
    })
    results = score_candidates([active, inactive])
    by_id = {r["candidate_id"]: r["score"] for r in results}
    assert by_id["CAND_0000005"] > by_id["CAND_0000006"]


def test_honeypot_flagged_expert_zero_duration():
    honeypot = _base_candidate("CAND_0000007", skills=[
        {"name": "FAISS", "proficiency": "expert", "endorsements": 5, "duration_months": 0},
        {"name": "NDCG", "proficiency": "expert", "endorsements": 5, "duration_months": 1},
    ])
    flags = honeypot_flags(honeypot)
    assert is_honeypot_suspect(honeypot)
    assert any("expert_skill_zero_duration" in f for f in flags)


def test_honeypot_career_months_exceed_yoe():
    honeypot = _base_candidate("CAND_0000008", profile={
        **_base_candidate("x")["profile"], "years_of_experience": 2.0,
    }, career_history=[{
        "company": "A", "title": "Eng", "start_date": "2015-01-01",
        "end_date": "2024-01-01", "duration_months": 108, "is_current": False,
        "industry": "Software", "company_size": "1-10", "description": "x",
    }])
    assert is_honeypot_suspect(honeypot)


def test_clean_candidate_not_flagged_as_honeypot():
    clean = _base_candidate("CAND_0000009")
    assert not is_honeypot_suspect(clean)
    assert honeypot_flags(clean) == []


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
