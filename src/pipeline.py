"""End-to-end RAG pipeline: PDF → chunks → embeddings → vector store → retrieve → generate.

Exposed as a CLI for Block 2 verification and as a library for Block 3 API.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from src.chunker import chunk_pages
from src.config import TOP_K
from src.embedder import embed_query, embed_texts
from src.extractor import extract_pdf
from src.generator import generate_answer
from src.store import RetrievedChunk, VectorStore


@dataclass
class RAGPipeline:
    store: VectorStore = field(default_factory=VectorStore)
    source_path: Optional[str] = None

    def ingest(self, pdf_path: str) -> int:
        """Load a PDF into the store. Returns chunk count."""
        pages = extract_pdf(pdf_path)
        chunks = chunk_pages(pages)
        print(f"  [ingest] {len(pages)} pages → {len(chunks)} chunks")

        # Embed in batches of 20 to stay well within free-tier rate limits
        embeddings: List[List[float]] = []
        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            batch = [c["text"] for c in chunks[i:i + batch_size]]
            embeddings.extend(embed_texts(batch))
            print(f"  [ingest] embedded {min(i + batch_size, len(chunks))}/{len(chunks)}")

        self.store.add(chunks, embeddings)
        self.source_path = pdf_path
        return len(chunks)

    def ask(self, question: str, top_k: int = TOP_K) -> dict:
        q_emb = embed_query(question)
        retrieved = self.store.search(q_emb, top_k=top_k)
        answer = generate_answer(question, retrieved)
        return {
            "question": question,
            "answer": answer["text"],
            "sources": answer["sources"],
            "retrieved": retrieved,
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m src.pipeline <path-to-pdf> \"<question>\"")
        sys.exit(1)

    pdf_path = sys.argv[1]
    question = sys.argv[2]

    print(f"Ingesting: {pdf_path}")
    pipe = RAGPipeline()
    pipe.ingest(pdf_path)

    print(f"\nQuestion: {question}")
    result = pipe.ask(question)

    print(f"\nAnswer: {result['answer']}")
    print(f"Cited pages: {result['sources']}")
    print(f"\nTop retrieved chunks:")
    for c in result["retrieved"]:
        print(f"  [{c['chunk_id']}] page {c['page_number']} (score {c['score']:.3f})")