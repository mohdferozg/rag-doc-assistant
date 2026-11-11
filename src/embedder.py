"""Embed text using Gemini's text-embedding-004.
Adds retry logic because Google's free tier has per-minute quotas.
"""
import time
from typing import List

from google import genai
from google.genai.errors import APIError

from src.config import EMBEDDING_MODEL, require_api_key

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=require_api_key())
    return _client


def embed_texts(texts: List[str], max_retries: int = 3) -> List[List[float]]:
    """Batch-embed a list of texts. Returns one vector per input."""
    if not texts:
        return []

    client = _get_client()
    last_err = None

    for attempt in range(max_retries):
        try:
            resp = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
            )
            return [e.values for e in resp.embeddings]
        except APIError as e:
            last_err = e
            # Only retry on rate limit (429) or server errors (5xx)
            status = getattr(e, "code", None) or getattr(e, "status_code", None)
            if status == 429 or (isinstance(status, int) and 500 <= status < 600):
                wait = 2 ** attempt
                print(f"  [embed] rate-limited, retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Embedding failed after {max_retries} retries: {last_err}")


def embed_query(text: str) -> List[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]