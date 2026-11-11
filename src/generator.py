"""Grounded generation: answer using ONLY the retrieved context.
Enforces the guardrail that's easiest to demo in review: 'I don't know'
when context doesn't cover the question.
"""
import time
from typing import List, TypedDict

from google import genai
from google.genai.errors import ServerError

from src.config import require_api_key
from src.model_selector import pick_model

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=require_api_key())
    return _client


SYSTEM_PROMPT = """You are a documentation assistant answering questions about a specific document.

RULES:
1. Answer ONLY using the provided CONTEXT. Do not use outside knowledge.
2. If the context does not contain the answer, respond exactly: "I don't know based on the provided document."
3. Cite the page number(s) you used, in the format [p.N].
4. Be concise. Prefer 1-3 sentences unless the question requires more.
5. Do not speculate. If a detail is not in the context, do not invent it."""


class Answer(TypedDict):
    text: str
    sources: List[int]  # page numbers cited


def build_context(chunks: List[dict]) -> str:
    """Turn retrieved chunks into a numbered context block."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] (page {c['page_number']})\n{c['text']}")
    return "\n\n".join(parts)


def generate_answer(question: str, chunks: List[dict]) -> Answer:
    if not chunks:
        return {"text": "I don't know based on the provided document.", "sources": []}

    context = build_context(chunks)
    prompt = f"""CONTEXT:
{context}

QUESTION: {question}

Answer using only the context above."""

    client = _get_client()
    
    max_retries = 3
    resp = None
    model_name = None

    for attempt in range(max_retries):
        try:
            model_name = pick_model(client)
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"system_instruction": SYSTEM_PROMPT},
            )
            break  # Success, exit the loop
        except ServerError as e:
            if "503" in str(e) and attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  [gen] 503 on {model_name}, retrying in {wait}s...")
                time.sleep(wait)
                # Clear the model cache so the next attempt picks a different model
                import src.model_selector as ms
                ms._cached_model = None
                continue
            raise  # If it's not a 503 or we ran out of retries, raise the error

    answer_text = (resp.text or "").strip() if resp else "Generation failed."
    sources = sorted({c["page_number"] for c in chunks})
    return {"text": answer_text, "sources": sources}