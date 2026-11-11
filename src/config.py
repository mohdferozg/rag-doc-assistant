"""Central configuration. All env vars and tunable constants live here."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present (falls back silently if not)
load_dotenv()

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# --- LLM ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001")

# --- Chunking ---
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))

# --- Retrieval ---
TOP_K = int(os.environ.get("TOP_K", "5"))


def require_api_key() -> str:
    """Fail fast if the API key is missing."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Set it via: $env:GEMINI_API_KEY='...'  (PowerShell) "
            "or add it to a .env file at the repo root."
        )
    return GEMINI_API_KEY