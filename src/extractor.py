"""PDF text extraction. Uses pypdf; keeps page numbers for citations."""
from pathlib import Path
from typing import List, TypedDict

from pypdf import PdfReader


class Page(TypedDict):
    page_number: int  # 1-indexed
    text: str


def extract_pdf(path: str | Path) -> List[Page]:
    """Return a list of Page dicts (1-indexed page numbers, cleaned text)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(str(path))
    pages: List[Page] = []

    for i, raw_page in enumerate(reader.pages, start=1):
        text = (raw_page.extract_text() or "").strip()
        if text:  # skip empty pages (covers, blank pages)
            pages.append({"page_number": i, "text": text})

    return pages