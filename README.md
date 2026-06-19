# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

A rule-based + TF-IDF hybrid ranker for the Redrob "Senior AI Engineer"
candidate pool. No GPU, no network calls, no hosted LLM inference during
ranking — fully reproducible on a CPU-only 16GB machine in well under the
5-minute budget (~60-100s for the full 100K-candidate pool on this
environment).

## Why this approach

The challenge explicitly traps keyword matching: a "Marketing Manager"
with 9 AI skills listed should rank low, and a "Backend Engineer" who
built a real recommendation system but never wrote "RAG" should rank high.
A pure embedding-similarity system over the `skills` array gets this
backwards. So this ranker deliberately:

- **Never scores against the `skills` array directly.** Production-ML
  evidence and eval-rigor evidence are extracted from `career_history`
  *descriptions* only — free text of what someone actually did, which is
  much harder to game than a list of tags.
- **Scores title relevance against actual job titles held**, not skills.
- **Uses TF-IDF cosine similarity** (not skills) between the JD and each
  candidate's headline + summary + career descriptions, so plain-language
  "hidden strong" candidates aren't penalized for not using buzzwords.
- **Applies a separate honeypot detector** that looks for arithmetically
  impossible profiles (skills marked "expert" with near-zero duration,
  career-history months that exceed stated years of experience, overlapping
  employment, malformed date ranges) and a separate disqualifier detector
  for the JD's explicit "don't want" list (pure research, consulting-only
  careers, CV/speech-only without NLP/IR, title-chasing, senior-but-not-coding).
- **Applies a behavioral modifier** (response rate, recency of activity,
  interview completion, notice period) multiplicatively on top of fit, so
  "behavioral twin" pairs are correctly separated.

## Repository structure

```
rank.py                  CLI entrypoint
src/
  jd_config.py            All JD-derived keyword lists & weights (auditable, no magic numbers in logic)
  features.py              Per-candidate feature extraction (title relevance, production/eval evidence, experience fit, location fit, behavioral modifier)
  honeypot.py               Severity-weighted honeypot anomaly detector
  disqualifiers.py          JD-explicit disqualifier penalty logic
  scoring.py                Orchestrator: TF-IDF + features → final score, tie-break-safe top-100 builder
  reasoning.py               Builds reasoning strings from already-extracted facts (no hallucination risk: nothing in the reasoning isn't already in fdict)
tests/
  test_scoring.py            Unit tests on synthetic keyword-stuffer / hidden-strong / behavioral-twin / honeypot cases
requirements.txt
submission_metadata.yaml
```

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the submission CSV

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Runtime on a 16GB CPU-only machine: ~60-100 seconds for the full 100,000-row
pool. No pre-computation step is required — TF-IDF is fit fresh on each run
(fast enough that caching wasn't necessary).

## Validate before submitting

```bash
python validate_submission.py submission.csv
```

## Run tests

```bash
python -m pytest tests/ -v
```

## Scoring methodology (summary)

```
base_fit = 0.30·text_similarity + 0.20·title_relevance + 0.20·production_evidence
         + 0.10·eval_evidence + 0.10·experience_fit + 0.10·location_fit

final_score = base_fit × behavioral_modifier × disqualifier_penalty × honeypot_penalty
```

- `text_similarity`: TF-IDF cosine similarity (1-2 grams) between JD text and
  candidate's headline+summary+career descriptions.
- `title_relevance`: regex match of current/past titles against an ML/AI/search/
  ranking/recommendation role pattern set.
- `production_evidence` / `eval_evidence`: keyword hits for retrieval/vector-DB/
  ranking terms and NDCG/MRR/MAP/A-B-testing terms, found only in career
  descriptions (not skills).
- `experience_fit`: soft curve centered on the JD's 5-9 year band.
- `location_fit`: Pune/Noida highest, other Tier-1 India cities next,
  relocation-willingness-adjusted for the rest.
- `behavioral_modifier` (0.45-1.15×): recruiter response rate, activity
  recency, interview completion, profile completeness, open-to-work flag,
  notice period.
- `disqualifier_penalty` (≤1.0×): JD's explicit "do not want" patterns
  (pure research, consulting-only, CV/speech-only, title-chasing, senior
  non-coding roles).
- `honeypot_penalty`: severity-weighted anomaly score; ≥2 severity points
  forces score to near-zero, 1 point applies a mild caution discount.

Verified on the released 100,000-candidate pool: **0% honeypot rate** and
**0 keyword-stuffer-titled candidates** in the produced top-100.

## Compute environment this was tested on

CPU-only, no GPU, no network calls during the ranking step. See
`submission_metadata.yaml` for the full declared environment — fill in your
own machine's specifics before submitting.

## AI tools used

This codebase was built with Claude as a pair-programming/architecture
collaborator. See `submission_metadata.yaml` for the full declaration.
Fill in the honest summary of your own usage before submitting — don't
just copy this file's text.
