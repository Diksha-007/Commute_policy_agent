"""All tunable settings in one place.

Values come from environment variables (or a .env file), falling back to the
defaults below. Field names are matched case-insensitively, so the Groq key is
read from `groq_key` / `GROQ_KEY`.
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Paths ---
    data_dir: Path = PROJECT_ROOT / "data"
    policies_dir: Path = PROJECT_ROOT / "policies"

    # --- Auth ---
    # Dev default for docker compose only (>= 32 bytes, as HS256 recommends).
    jwt_secret: str = "dev-only-secret-change-me-0123456789abcdef"
    jwt_issuer: str = "mis-demo-issuer"
    jwt_algorithm: str = "HS256"
    jwt_ttl_minutes: int = 60

    # Demo page + /demo/login (sign in as any company, no password). OFF unless DEMO_MODE=true; see app/demo.py.
    demo_mode: bool = False

    # --- LLM (Groq) ---
    groq_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # Optional separate key for the evaluation judge (eval/judge.py); empty = use groq_key.
    groq_judge_key: str = ""
    # Models measured on Groq (Nov 2026; the Llama 3.x models are no longer served):
    #   qwen3.8-27b, reasoning none : ~0.4-0.75 s   -> HyDE (short passage, speed matters)
    #   gpt-oss-120b, reasoning low : ~0.6-1.2 s    -> answer (strongest instruction following)
    hyde_model: str = "qwen/qwen3.8-27b"
    hyde_reasoning_effort: str = "none"
    # The evaluation judge (eval/judge.py): a different model family than the answerer, so it does not grade itself.
    judge_model: str = "qwen/qwen3.8-27b"
    judge_reasoning_effort: str = "none"
    answer_model: str = "openai/gpt-oss-120b"
    answer_reasoning_effort: str = "low"
    # HyDE (an LLM call writes a policy-style passage that is searched alongside the question) is ON by project choice.
    # Measured effect on retrieval: none we can detect (experiments E3, E4 in docs/experiments.md); cost: one more LLM call and
    # about 0.2-0.7 s. It never fails a question: on any HyDE error the search simply uses the question alone.
    # Set USE_HYDE=false to turn it off.
    use_hyde: bool = True

    # --- Limits from the brief ---
    max_llm_calls: int = 2       # per question
    top_k: int = 3               # max chunks CITED per answer
    # Chunks retrieved and shown to the model. Experiment E2: retrieving 5 finds +0.10 more needed facts than
    # retrieving 3 (7 questions better, 0 worse); only 3 can be cited.
    retrieve_k: int = 5

    # --- Latency budget (seconds) ---
    # These are ceilings for SLOW hardware and a slow first request, not typical times. On the machine the evaluation was
    # run on, questions take about 1.0 to 2.4 s (p50 1.5 s, p95 2.4 s with HyDE on), so the brief's p95 <= 3 s is met there.
    # They were tighter (2.7 s total, 0.5 s search, 1.0 s HyDE, 2.2 s answer) and a slower machine then got "try again" on
    # its first questions (search over 0.5 s, HyDE over 1.0 s). The trade: a slow machine now gets a slow answer, up to
    # total_budget_s, instead of an error; its p95 can exceed 3 s.
    total_budget_s: float = 5.0        # hard cap for the whole question: HyDE + search + answer share it
    hyde_timeout_s: float = 1.5        # HyDE is optional: past this the search uses the question alone
    retrieval_timeout_s: float = 1.5   # embed the question + search the collection
    answer_timeout_s: float = 3.0      # the answer call (also bounded by what is left of total_budget_s)
    min_answer_budget_s: float = 1.0   # do not start an answer call with less than this left: it could not finish

    # --- Retrieval ---
    candidates_per_query: int = 6   # pre-fusion candidates per query
    rrf_k: int = 60
    # MMR re-ranking (optional): choose the top_k chunks that are relevant AND different
    # from each other, so near-duplicate chunks do not fill the 3 citation slots.
    # score = lambda * relevance - (1 - lambda) * max similarity to chunks already chosen.
    use_mmr: bool = False
    mmr_lambda: float = 0.7
    # MMR needs more candidates than it keeps: a chunk that duplicates a better one can only
    # be replaced by something useful if the useful chunk is in the pool (measured: pool 6 -> 10
    # recovers sections plain top-3 and pool-6 MMR both miss).
    mmr_pool: int = 10
    # Below this, reply "not covered" without calling the LLM. Chosen from
    # eval/results/threshold_sweep.md (bge-base, 61 cases): the lowest answerable case scores 0.573, so 0.50
    # leaves a 0.073 safety margin (a wrongly gated answerable question is worse than one extra LLM call)
    # and still catches the clearly off-topic questions. (MiniLM needed 0.30; its scores sat lower.)
    min_similarity: float = 0.50

    # --- Chunking ---
    semantic_breakpoint_percentile: float = 85.0
    min_chunk_chars: int = 150
    max_chunk_chars: int = 900
    table_rows_per_chunk: int = 4

    # --- Ingestion ---
    gc_grace_seconds: float = 30.0

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "registry.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
