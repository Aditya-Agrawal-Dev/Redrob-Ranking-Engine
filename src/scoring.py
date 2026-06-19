"""
Top-level scoring pipeline. Computes TF-IDF similarity in bulk (vectorized,
fast across 100K candidates), then combines with per-candidate rule-based
features into a final score.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import jd_config as cfg
from . import features as feat
from . import honeypot as hp
from . import disqualifiers as dq
from . import reasoning as rs


def _candidate_doc(candidate: dict) -> str:
    """Text used for TF-IDF similarity: summary + headline + all career
    descriptions + all titles. Deliberately excludes the skills array so
    keyword-stuffed skill lists don't drive this signal."""
    profile = candidate.get("profile", {}) or {}
    parts = [profile.get("headline", ""), profile.get("summary", "")]
    for c in candidate.get("career_history", []) or []:
        parts.append(c.get("title", ""))
        parts.append(c.get("description", ""))
    return " ".join(p for p in parts if p)


def score_candidates(candidates: list[dict]) -> list[dict]:
    """Returns list of dicts with candidate_id, score, reasoning, plus
    diagnostic fields, sorted descending by score."""

    docs = [_candidate_doc(c) for c in candidates]
    vectorizer = TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), stop_words="english", min_df=2
    )
    corpus = docs + [cfg.JD_TEXT]
    tfidf = vectorizer.fit_transform(corpus)
    jd_vec = tfidf[-1]
    cand_vecs = tfidf[:-1]
    sims = cosine_similarity(cand_vecs, jd_vec).flatten()  # 0..1 already (tfidf is non-negative)

    results = []
    for cand, text_sim in zip(candidates, sims):
        t_rel = feat.title_relevance(cand)
        prod_score, prod_hits = feat.production_evidence(cand)
        eval_score, eval_hits = feat.eval_evidence(cand)
        exp_fit = feat.experience_fit(cand)
        loc_fit = feat.location_fit(cand)

        base = (
            cfg.WEIGHTS["text_similarity"] * text_sim
            + cfg.WEIGHTS["title_relevance"] * t_rel
            + cfg.WEIGHTS["production_evidence"] * prod_score
            + cfg.WEIGHTS["eval_evidence"] * eval_score
            + cfg.WEIGHTS["experience_fit"] * exp_fit
            + cfg.WEIGHTS["location_fit"] * loc_fit
        )

        behav_mult, behav_detail = feat.behavioral_modifier(cand)
        dq_mult, dq_reasons = dq.disqualifier_penalty(cand)
        hp_severity, hp_flags = hp.honeypot_severity(cand)
        if hp_severity >= 2:
            hp_mult = 0.05  # near-zero: strong honeypot suspect
        elif hp_severity == 1:
            hp_mult = 0.85  # mild caution, not disqualifying
        else:
            hp_mult = 1.0

        final = base * behav_mult * dq_mult * hp_mult

        fdict = {
            "text_sim": round(float(text_sim), 4),
            "title_relevance": t_rel,
            "production_hits": prod_hits,
            "eval_hits": eval_hits,
            "experience_fit": round(exp_fit, 3),
            "location_fit": round(loc_fit, 3),
            "behavioral_detail": behav_detail,
            "disqualifier_reasons": dq_reasons,
            "honeypot_flag_count": len(hp_flags),
            "honeypot_flags": hp_flags,
        }

        results.append({
            "candidate_id": cand["candidate_id"],
            "score": final,
            "fdict": fdict,
            "candidate": cand,
        })

    results.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    return results


def build_top100_rows(results: list[dict]) -> list[dict]:
    top = results[:100]
    max_score = top[0]["score"] if top else 1.0

    scored = []
    for r in top:
        norm_score = round(r["score"] / max_score, 4) if max_score > 0 else 0.0
        scored.append((norm_score, r))

    # Re-sort after rounding so any ties created by rounding still satisfy
    # the spec's tie-break rule: equal scores -> candidate_id ascending.
    scored.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))

    rows = []
    for i, (norm_score, r) in enumerate(scored, start=1):
        reasoning = rs.build_reasoning(r["candidate"], r["fdict"], i)
        rows.append({
            "candidate_id": r["candidate_id"],
            "rank": i,
            "score": norm_score,
            "reasoning": reasoning,
        })

    # Enforce strictly non-increasing score by rank (guard against any
    # remaining float noise).
    for i in range(1, len(rows)):
        if rows[i]["score"] > rows[i - 1]["score"]:
            rows[i]["score"] = rows[i - 1]["score"]
    return rows
