"""
JD-derived configuration for the Redrob Senior AI Engineer ranking challenge.

All keyword lists and weights live here, separate from scoring logic, so the
mapping from "what the JD says" to "what the code checks" is auditable.
Every list below traces to a specific sentence in job_description.docx —
see the inline comments.
"""

# ---------------------------------------------------------------------------
# Roles considered genuinely relevant to "owns ranking/retrieval/matching
# systems" — used against current_title AND every career_history title.
# This is the anti-keyword-stuffer signal: it ignores the skills array
# entirely and only looks at what someone was actually employed to do.
# ---------------------------------------------------------------------------
RELEVANT_TITLE_PATTERNS = [
    r"\bai engineer\b", r"\bml engineer\b", r"\bmachine learning engineer\b",
    r"\bapplied scientist\b", r"\bresearch engineer\b", r"\bnlp engineer\b",
    r"\bdata scientist\b", r"\bsearch engineer\b", r"\branking engineer\b",
    r"\bretrieval engineer\b", r"\brecommendation\b", r"\binformation retrieval\b",
    r"\bsearch relevance\b", r"\bsenior software engineer\b.*\b(search|ml|ai|nlp)\b",
    r"\bbackend engineer\b.*\b(ml|search|recommendation|ranking)\b",
]

# Title patterns that suggest the person has moved away from hands-on coding
# (JD disqualifier: "hasn't written production code in 18 months").
NON_CODING_SENIOR_TITLES = [
    r"\bengineering manager\b", r"\btech(nical)? lead\b", r"\barchitect\b",
    r"\bdirector\b", r"\bvp\b", r"\bhead of\b",
]

# ---------------------------------------------------------------------------
# Production ML/retrieval evidence — checked ONLY against career_history
# descriptions (free text of what they actually did), never against the
# skills array, because the skills array is the documented trap surface.
# ---------------------------------------------------------------------------
PRODUCTION_RETRIEVAL_TERMS = [
    "embedding", "sentence-transformer", "sentence transformers", "bge", "e5 ",
    "vector database", "vector db", "vector search", "pinecone", "weaviate",
    "qdrant", "milvus", "opensearch", "elasticsearch", "faiss", "bm25",
    "hybrid search", "hybrid retrieval", "dense retrieval", "semantic search",
    "retrieval", "re-ranking", "reranking", "ranking system", "ranking model",
    "recommendation system", "recommender", "search relevance", "candidate matching",
    "index refresh", "embedding drift", "ann index", "approximate nearest neighbor",
]

EVAL_FRAMEWORK_TERMS = [
    "ndcg", "mrr", "map@", "mean average precision", "precision@", "recall@",
    "a/b test", "ab test", "offline evaluation", "online evaluation",
    "offline-to-online", "evaluation framework", "click-through", "ctr ",
]

LLM_FINETUNE_TERMS = ["lora", "qlora", "peft", "fine-tun"]
LEARNING_TO_RANK_TERMS = ["learning-to-rank", "learning to rank", "xgboost", "ltr model"]

# Signals that someone's "AI experience" is shallow LangChain-wrapper work
# rather than real systems experience (JD disqualifier).
SHALLOW_LLM_WRAPPER_TERMS = ["langchain tutorial", "called openai api", "openai api wrapper"]

# ---------------------------------------------------------------------------
# Disqualifier-relevant terms
# ---------------------------------------------------------------------------
CV_SPEECH_ROBOTICS_TERMS = [
    "computer vision", "image classification", "object detection", "image segmentation",
    "speech recognition", "robotics", "autonomous", "lidar", "slam ",
]
NLP_IR_TERMS = [
    "nlp", "natural language", "information retrieval", "search", "retrieval",
    "ranking", "recommendation", "text classification", "named entity",
]

CONSULTING_COMPANIES = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant", "capgemini",
]

RESEARCH_ONLY_INDUSTRIES = ["research", "academia", "academic", "university"]
PRODUCTION_EVIDENCE_TERMS = [
    "shipped", "deployed", "production", "real users", "scale", "live system",
    "launched", "rolled out",
]

# ---------------------------------------------------------------------------
# Location — JD: "Pune/Noida-preferred... Hyderabad, Pune, Mumbai, Delhi NCR
# welcome to apply... outside India: case-by-case, no visa sponsorship"
# ---------------------------------------------------------------------------
PRIMARY_LOCATIONS = ["pune", "noida"]
TIER1_INDIA_LOCATIONS = [
    "pune", "noida", "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram",
    "bengaluru", "bangalore", "delhi ncr",
]

# ---------------------------------------------------------------------------
# JD full text used for TF-IDF similarity (trimmed to substantive content,
# excluding boilerplate headers/legal language).
# ---------------------------------------------------------------------------
JD_TEXT = """
Senior AI Engineer, Founding Team, Redrob AI. Series A AI-native talent
intelligence platform. Own the intelligence layer of Redrob's product:
ranking, retrieval, and matching systems that decide what recruiters see
when they search for candidates and what candidates see when they search
for roles. Audit existing BM25 and rule-based scoring. Ship a v2 ranking
system using embeddings, hybrid retrieval, and LLM-based re-ranking.
Set up evaluation infrastructure: offline benchmarks, online A/B testing,
recruiter-feedback loops. Drive long-term architecture of candidate-JD
matching at scale. Production experience with embeddings-based retrieval
systems: sentence-transformers, OpenAI embeddings, BGE, E5, deployed to
real users, handling embedding drift, index refresh, retrieval-quality
regression in production. Production experience with vector databases or
hybrid search infrastructure: Pinecone, Weaviate, Qdrant, Milvus,
OpenSearch, Elasticsearch, FAISS. Strong Python, code quality. Hands-on
experience designing evaluation frameworks for ranking systems: NDCG, MRR,
MAP, offline-to-online correlation, A/B test interpretation. LLM
fine-tuning LoRA QLoRA PEFT. Learning-to-rank models XGBoost or neural.
HR-tech recruiting tech marketplace products. Distributed systems
large-scale inference optimization. Open-source contributions AI ML.
Scrappy product-engineering attitude, ship a working ranker in a week,
tilt toward shipper not researcher. Mentor engineers, grow team from 4 to
12. Work closely with recruiter-experience product manager.
"""

# ---------------------------------------------------------------------------
# Component weights for the base fit score (must sum to 1.0)
# ---------------------------------------------------------------------------
WEIGHTS = {
    "text_similarity": 0.30,
    "title_relevance": 0.20,
    "production_evidence": 0.20,
    "eval_evidence": 0.10,
    "experience_fit": 0.10,
    "location_fit": 0.10,
}

# Experience band per JD ("5-9 years... not a hard requirement")
EXPERIENCE_BAND_CENTER = (5, 9)
EXPERIENCE_SOFT_MIN = 3   # below this, fit drops sharply
EXPERIENCE_SOFT_MAX = 13  # above this, fit drops (overqualified/title-track risk)

# Notice period preference: JD wants sub-30-day, can buy out up to 30
NOTICE_PERIOD_IDEAL_DAYS = 30
