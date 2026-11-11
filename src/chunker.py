"""Recursive-ish character chunker with overlap.
Keeps page provenance so answers can cite the page."""
from typing import List, TypedDict

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.extractor import Page


class Chunk(TypedDict):
    chunk_id: str
    page_number: int
    text: str


def chunk_pages(
    pages: List[Page],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """Split each page into overlapping chunks.
    chunk_size and overlap are measured in *characters* here
    (a deliberate simplification — see README).
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[Chunk] = []
    stride = chunk_size - overlap

    for page in pages:
        text = page["text"]
        page_num = page["page_number"]

        if len(text) <= chunk_size:
            chunks.append({
                "chunk_id": f"p{page_num}_c0",
                "page_number": page_num,
                "text": text,
            })
            continue

        idx = 0
        for c_idx, start in enumerate(range(0, len(text), stride)):
            end = start + chunk_size
            piece = text[start:end].strip()
            if not piece:
                break
            chunks.append({
                "chunk_id": f"p{page_num}_c{c_idx}",
                "page_number": page_num,
                "text": piece,
            })
            if end >= len(text):
                break

    return chunks


if __name__ == "__main__":
    import sys
    from src.extractor import extract_pdf

    if len(sys.argv) < 2:
        print("Usage: python -m src.chunker <path-to-pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    pages = extract_pdf(pdf_path)
    chunks = chunk_pages(pages)

    print(f"Extracted {len(pages)} non-empty pages")
    print(f"Produced   {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print("-" * 60)
    print(f"Sample chunk (first):")
    print(f"  id: {chunks[0]['chunk_id']}")
    print(f"  page: {chunks[0]['page_number']}")
    print(f"  text: {chunks[0]['text'][:300]}...")