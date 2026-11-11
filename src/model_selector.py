"""Dynamically pick an available Gemini model that supports generateContent."""
from typing import Optional

from google import genai

from src.config import require_api_key

# Preference order: newest Flash first, fallbacks after
PREFERRED_MODELS = [
    "gemini-3.5-flash-lite",  # or gemini-3.1-flash-lite
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.8-flash",       # ← try after
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
]

_cached_model: Optional[str] = None


def pick_model(client: genai.Client) -> str:
    """Return the first preferred model that exists and supports generateContent."""
    global _cached_model
    if _cached_model:
        return _cached_model

    available = set()
    for m in client.models.list():
        if "generateContent" in (m.supported_actions or []):
            # m.name is like "models/gemini-3.8-flash"
            available.add(m.name.replace("models/", ""))

    for candidate in PREFERRED_MODELS:
        if candidate in available:
            _cached_model = candidate
            print(f"  [model] selected: {candidate}")
            return candidate

    raise RuntimeError(
        f"No preferred model available. Found: {sorted(available)}"
    )