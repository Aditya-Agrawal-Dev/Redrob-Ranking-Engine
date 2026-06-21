# Redrob Ranking Engine

Ranking system for the Redrob Intelligent Candidate Discovery & Ranking
Challenge. Takes a Senior AI Engineer job description and 100,000
candidate profiles, outputs the top 100 as a CSV.

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

CPU only, no GPU, no network calls during ranking, ~90-150s for the full
dataset.

## Problem

The challenge dataset is built to break keyword matching. It has
candidates with irrelevant titles but AI-sounding skill tags, candidates
with real production ML experience who never mention specific buzzwords,
near-duplicate profiles where only one is actually reachable, and ~80
honeypots with internally inconsistent data (e.g. "expert" proficiency in
a skill listed with 0 months of use). More than 10% honeypots in the top
100 results in disqualification.

To handle this, the scoring pipeline never reads the `skills` array
directly. All technical scoring comes from career-history text and job
titles, since those are harder to fabricate than a tag list.

## Architecture

```
rank.py
src/
  jd_config.py     keyword lists + component weights
  features.py      per-candidate feature extraction
  honeypot.py       anomaly detection
  disqualifiers.py  JD-based hard filters
  scoring.py         combines everything, builds the output CSV
  reasoning.py        generates reasoning text from extracted features
tests/
  test_scoring.py
```

`jd_config.py` holds every keyword list and weight used by the system, so
the mapping from JD requirements to scoring logic stays in one place
instead of being scattered across the codebase.

## Scoring

```python
base_fit = (
    0.34 * text_similarity
  + 0.20 * title_relevance
  + 0.23 * production_evidence
  + 0.10 * eval_evidence
  + 0.10 * experience_fit
  + 0.03 * location_fit
)

final_score = base_fit * behavioral_modifier * disqualifier_penalty * honeypot_penalty
```

- **text_similarity** — TF-IDF cosine similarity between the JD and each
  candidate's summary + career descriptions.
- **title_relevance** — regex match against current and past job titles
  (ML/AI/search/ranking/recommendation roles), not skills.
- **production_evidence** — keyword hits inside career-history
  descriptions, covering vector DBs/retrieval (FAISS, Pinecone,
  embeddings, BM25...), LLM fine-tuning (LoRA, QLoRA, PEFT), and
  learning-to-rank (XGBoost, ranking model). Score is `min(1.0, hits/4)`.
- **eval_evidence** — NDCG/MRR/MAP/A-B-testing mentions in descriptions.
- **experience_fit** — soft curve around the JD's 5-9 year band.
- **location_fit** — Pune/Noida highest, other Tier-1 cities next,
  relocation-adjusted otherwise. Weighted low (0.03) since the JD treats
  location as a soft preference, not a requirement.

## Behavioral modifier

Applied as a multiplier (0.45x-1.15x) on top of base_fit:

```python
raw = (
    0.30 * recruiter_response_rate
  + 0.20 * recency_score
  + 0.15 * interview_completion_rate
  + 0.15 * profile_completeness_score
  + 0.10 * (1.0 if open_to_work else 0.3)
  + 0.10 * notice_score
)
modifier = 0.45 + raw * 0.70
```

Multiplicative so a weak-fit candidate can't compensate with high
availability, but a strong-fit candidate with poor availability is
down-weighted instead of zeroed out.

## Honeypot detection

`honeypot.py` checks for six anomaly types: expert skills with near-zero
duration, too many expert skills with low average tenure, career-history
months exceeding stated years of experience, broken date ranges,
overlapping roles, and stated experience exceeding the time since the
earliest listed job.

A single flag applies a 0.85x discount only. The maximum penalty (0.05x)
requires two or more flags at once. This matters because `career_history`
only requires 1-10 entries per the schema — a candidate with a long
career and a short job list can trigger a flag like
`yoe_exceeds_career_span` without actually being dishonest. Requiring two
independent flags avoids false-positives from this while still catching
profiles that stack multiple inconsistencies.

## Disqualifiers

`disqualifiers.py` applies penalty multipliers for patterns the JD
explicitly rules out: pure research with no production experience,
consulting-only careers, CV/speech backgrounds with no NLP/IR exposure,
title-chasing (short tenures with escalating seniority), and senior
titles with no recent coding evidence.

## Runtime

| Stage | Time |
|---|---|
| Load 100K candidates | ~16-23s |
| Score all candidates | ~65-126s |
| Write output CSV | <1s |
| **Total** | **~90-150s on the full dataset** |

Well within the 5-minute / 16GB / CPU-only constraint.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py submission.csv
python -m pytest tests/ -v
```

## Results

On the full 100K candidate pool:

- 100 rows output, passes `validate_submission.py`
- 0/100 honeypot suspects (2+ flags) in the top 100; 1 candidate with a
  single mild flag, discounted but not excluded
- 0/100 keyword-stuffed titles in the top 100
- score range: 1.00 (rank 1) to 0.76 (rank 100)
- top candidate: Lead AI Engineer, 6.7 yrs experience, strong overlap on
  text similarity and production evidence, full eval-rigor coverage
- all 7 unit tests pass, covering keyword-stuffer ranking, hidden-strong
  candidate fairness, behavioral-twin separation, and honeypot gating
- final submission passes `validate_submission.py` without errors
