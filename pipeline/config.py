"""
Central configuration for the pipeline evaluation protocol.

Pipeline: split (train/val/test) -> LLM action-sentence labeling (val+test) ->
spaCy VerbPhrase extraction (report-local G_r) -> index train into a dedicated
Chroma + BM25 index -> two-stage hybrid HP search on val (embedder x alpha,
then k) -> freeze HP -> BERTScore/MoverScore and text diversity on test.

ALL tunable knobs live here.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_ROOT = Path(__file__).resolve().parent

DATA_CSV = REPO_ROOT / "data" / "breach_report_202509.csv"
BAD_VERBS_CSV = REPO_ROOT / "data" / "bad_verbs.csv"

DATA_DIR = PIPELINE_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
LABELS_DIR = DATA_DIR / "labels"
CLAUDE_DIR = LABELS_DIR / "claude"
MISTRAL_DIR = LABELS_DIR / "mistral"
LLAMA_DIR = LABELS_DIR / "llama"
VPS_DIR = DATA_DIR / "vps"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = PIPELINE_ROOT / "results"
VECTORSTORE_DIR = PIPELINE_ROOT / "vectorstore"

for _d in (SPLITS_DIR, CLAUDE_DIR, MISTRAL_DIR, LLAMA_DIR, VPS_DIR, CACHE_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

REPORTS_CACHE = SPLITS_DIR / "reports.json"
TRAIN_IDS_JSON = SPLITS_DIR / "train_ids.json"
VAL_IDS_JSON = SPLITS_DIR / "val_ids.json"
TEST_IDS_JSON = SPLITS_DIR / "test_ids.json"
SPLIT_SUMMARY_JSON = SPLITS_DIR / "split_summary.json"
REPORT_SENTENCES_JSON = LABELS_DIR / "report_sentences.json"
AGREED_ACTIONS_JSON = LABELS_DIR / "agreed_action_sentences.json"
REPORT_VPS_JSON = VPS_DIR / "report_vps.json"

# --------------------------------------------------------------------------- #
# CSV schema (same schema as the old-paper CSV)
# --------------------------------------------------------------------------- #
TEXT_FIELD = "Web Description"
DATE_FIELD = "Breach Submission Date"
DATE_FORMAT = "%m/%d/%Y"
TYPE_FIELD = "Type of Breach"
METADATA_FIELDS = [
    "Name of Covered Entity",
    "State",
    "Covered Entity Type",
    "Individuals Affected",
    "Breach Submission Date",
    "Type of Breach",
    "Location of Breached Information",
    "Business Associate Present",
]

# --------------------------------------------------------------------------- #
# Split: stratify by Type of Breach, sort old->new within each stratum
# --------------------------------------------------------------------------- #
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1  # remainder

# Strata smaller than this go entirely to train (too small to split reliably).
MIN_GROUP_SIZE_TO_SPLIT = 10

# --------------------------------------------------------------------------- #
# Sentence splitting (EMBER report rules)
# --------------------------------------------------------------------------- #
PRESPLIT_BREAK_BEFORE = [" As a result of", " Upon"]
NORMALIZE_CURLY_QUOTES = True
APPLY_MERGE_RULES = True

# --------------------------------------------------------------------------- #
# Indexing (train split only)
# --------------------------------------------------------------------------- #
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_DISTANCE_SPACE = "cosine"
COLLECTION_NAME = "ember_train_minilm"
# EMBER "Good" report filter (5-10 sentences, no duplicate marker). Set False
# to index every train sentence regardless of report length.
INDEX_APPLY_GOOD_FILTER = False
MIN_SENTENCES = 5
MAX_SENTENCES = 10
DUPLICATE_MARKER = "duplicat"
MIN_CHUNK_WORDS = 0

RETRIEVE_N_CANDIDATES = 500

# --------------------------------------------------------------------------- #
# Hyperparameter search on validation: embedder x hybrid alpha at k=10 (Stage 1),
# then k on the chosen setting (Stage 2).
# Hybrid: score = α * cosine_minmax + (1-α) * BM25_minmax  (per-query min-max)
# Cache keys: `{embedder}_{alpha}_{topk}_{breach_id}` e.g. minilm_1.0_10_952
# --------------------------------------------------------------------------- #
EMBEDDERS = {
    "minilm": {
        "model": "all-MiniLM-L6-v2",
        "collection": COLLECTION_NAME,
        "kind": "sentence_transformer",
    },
    "openai3large": {
        "model": "text-embedding-3-large",
        "collection": "ember_train_openai3large",
        "kind": "openai",
    },
}
SWEEP_EMBEDDERS = ["minilm", "openai3large"]
SWEEP_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]  # α on cosine; 1=dense-only, 0=BM25-only
SWEEP_STAGE1_TOPK = 10
SWEEP_STAGE2_TOPKS = [5, 10, 15, 20]
# Setting chosen in Stage 1 (hybrid MiniLM, alpha=0.25); Stage 2 sweeps k on it
# and the test run uses k=10.
STAGE2_EMBEDDER = "minilm"
STAGE2_ALPHA = 0.25
FROZEN_EMBEDDER = "minilm"
FROZEN_ALPHA = 0.25
FROZEN_TOPK = 10
BM25_DIR = DATA_DIR / "bm25"
BM25_INDEX_PATH = BM25_DIR / "train_sentences.pkl"
OPENAI_TRAIN_EMB_CACHE = CACHE_DIR / "openai3large_train_embeddings.npz"
OPENAI_VAL_QUERY_EMB_CACHE = CACHE_DIR / "openai3large_val_query_embeddings.npz"
HP_STAGE1_INVENTORY_JSON = RESULTS_DIR / "hp_stage1_inventory.json"
HP_STAGE1_SWEEP_JSON = RESULTS_DIR / "val_sweep_hybrid_stage1.json"
HP_STAGE2_INVENTORY_JSON = RESULTS_DIR / "hp_stage2_inventory.json"
HP_STAGE2_SWEEP_JSON = RESULTS_DIR / "val_sweep_hybrid_stage2.json"
HP_FROZEN_TEST_INVENTORY_JSON = RESULTS_DIR / "hp_frozen_test_inventory.json"
TEST_HYBRID_JSON = RESULTS_DIR / "test_hybrid_minilm_a025_k10.json"
SENTENCE_GR_JSON = VPS_DIR / "report_g_r_sentences.json"

BM25_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Generation LLM (RAG answer generation). Set OPENAI_API_KEY in the
# environment or in .env (see .env.example).
# --------------------------------------------------------------------------- #
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

GEN_LLM_MODEL = os.environ.get("EMBER_LLM_MODEL", "gpt-4o")
GEN_LLM_TEMPERATURE = 0.0
GEN_LLM_MAX_TOKENS = 600
SUGGESTED_ACTION_COUNT = 3  # median verb-phrase count of G_r
INCLUDE_FULL_REPORT_IN_CONTEXT = True

SYSTEM_PROMPT = (
    "You are a HIPAA compliance expert. You have been given similar breach "
    "reports from the HHS OCR portal. Based on these real past breach reports, "
    "suggest specific, actionable steps that the covered entity should take.\n"
    "Focus on corrective actions that are:\n"
    "1. Causally relevant to the specific breach described\n"
    "2. Helpful toward HIPAA compliance\n"
    f"Provide exactly {SUGGESTED_ACTION_COUNT} numbered action items — no more, "
    "no fewer. Each item should be one concrete corrective measure (verb phrase "
    "style, as in past OCR reports). Order from most to least important.\n"
    "Only suggest actions that appear in or are directly supported by the "
    "retrieved reports below. Do not invent, infer, or add generic HIPAA advice "
    "that is not grounded in the provided context."
)

USER_PROMPT_TEMPLATE = (
    "## New breach to advise on\n"
    "{query}\n\n"
    "## Similar past breach reports (retrieved from HHS OCR corpus)\n"
    "{context}\n\n"
    "Suggest corrective actions for the new breach using ONLY actions that "
    "similar covered entities took in the retrieved reports above. Do not make "
    "up actions that are not stated or clearly implied in that context.\n"
    f"Output ONLY a numbered list of exactly {SUGGESTED_ACTION_COUNT} actions "
    "(lines 1. through "
    f"{SUGGESTED_ACTION_COUNT}.) — no introduction, no sub-bullets, no extra text."
)



# --------------------------------------------------------------------------- #
# Text diversity metrics (Shaib et al. 2024, arXiv:2403.00553)
# --------------------------------------------------------------------------- #
# "Document" for every corpus-level metric = one report's generated actions
# (or G_r phrases) concatenated into a single string.
DIVERSITY_NGD_MAX_N = 4          # N-Gram Diversity Score: sum over n=1..4
DIVERSITY_SELF_BLEU_MAX_N = 4    # pairwise BLEU order for Self-BLEU / homogenization
# Default from Shaib et al. (2024). See diversity_utils.py.
DIVERSITY_SELF_REP_N = 4
# Cap on # ordered pairs for corpus-level Self-BLEU (O(n^2) pairs); None = no cap.
DIVERSITY_SELF_BLEU_MAX_PAIRS = None

# --------------------------------------------------------------------------- #
# Action-sentence labeling LLMs
# --------------------------------------------------------------------------- #
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
MISTRAL_MODEL = "mistral-large-latest"  # Mistral Large 3
# OpenRouter slug for Llama 3.3 70B Instruct.
LLAMA_MODEL = "meta-llama/llama-3.3-70b-instruct"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

LABEL_LLM_MAX_TOKENS = 512
LABEL_LLM_TEMPERATURE = 0.0

# Agreement rule to build report-local G_r sentence set from the 3 labelers.
# "intersection" -> all 3 must flag the sentence. "majority" -> >=2 of 3.
AGREEMENT_RULE = "intersection"

ACTION_LABEL_SYSTEM_PROMPT = (
    "You identify which numbered lines in a HIPAA breach report are "
    "corrective-action sentences.\n"
    "A corrective-action sentence describes something a covered entity, "
    "business associate, employee, or OCR actually did (or is doing) in "
    "response to the breach (e.g. encrypt devices, retrain staff, revise "
    "policies, notify individuals, sanction employees, hire a vendor, "
    "install monitoring).\n"
    "Do NOT mark sentences that only describe the breach, its cause, who "
    "was affected, background, or speculation.\n"
    "Return ONLY a JSON array of the matching line numbers (ordered). Example: "
    "[2,4,5]. Return [] if none. No other text. Do not rank, order by "
    "importance, or invent a fixed count of actions."
)

ACTION_LABEL_USER_TEMPLATE = (
    "Numbered lines:\n{numbered_sentences}\n\n"
    "Identify the line numbers that are corrective-action sentences. "
    "JSON array of integers only."
)

# --------------------------------------------------------------------------- #
# spaCy VerbPhrase extraction
# --------------------------------------------------------------------------- #
SPACY_MODEL = "en_core_web_lg"
KEPT_VERB_DEPS = ["ROOT", "advcl", "conj", "relcl", "ccomp", "xcomp", "pcomp", "acl"]
# Also accept mistagged remediation verbs (e.g. "revised"/"retrained" as amod).
ALLOW_AMOD_VERB_TAGS = ("VBN", "VBD", "VBG")
# Lemmas to keep even though they appear in the shared Ember bad_verbs.csv
# (HIPAA remediation staples that the stop list would otherwise block).
BAD_VERBS_ALLOWLIST = ["notify", "provide"]
MIN_VP_WORDS = 3  # shorter phrases are parser fragments
