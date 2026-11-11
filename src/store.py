"""In-memory vector store. Brute-force cosine similarity using numpy.

Deliberately simple: for ~72 chunks, brute force beats a vector DB on
every axis (latency, setup, complexity). See README for the
production trade-off discussion.
"""
from typing import List, TypedDict

import numpy as np

from src.chunker import Chunk


class StoredChunk(TypedDict):
    chunk_id: str
    page_number: int
    text: str
    embedding: List[float]


class RetrievedChunk(TypedDict):
    chunk_id: str
    page_number: int
    text: str
    score: float


class VectorStore:
    def __init__(self) -> None:
        self._chunks: List[StoredChunk] = []
        self._matrix: np.ndarray | None = None

    def add(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be same length")
        for c, e in zip(chunks, embeddings):
            self._chunks.append({
                "chunk_id": c["chunk_id"],
                "page_number": c["page_number"],
                "text": c["text"],
                "embedding": e,
            })
        self._rebuild_matrix()

    def _rebuild_matrix(self) -> None:
        if not self._chunks:
            self._matrix = None
            return
        mat = np.array([c["embedding"] for c in self._chunks], dtype=np.float32)
        # Normalize rows so cosine = dot product
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = mat / norms

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[RetrievedChunk]:
        if self._matrix is None or not self._chunks:
            return []
        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm
        scores = self._matrix @ q
        k = min(top_k, len(self._chunks))
        top_idx = np.argsort(scores)[::-1][:k]
        return [
            {
                "chunk_id": self._chunks[i]["chunk_id"],
                "page_number": self._chunks[i]["page_number"],
                "text": self._chunks[i]["text"],
                "score": float(scores[i]),
            }
            for i in top_idx
        ]

    def __len__(self) -> int:
        return len(self._chunks)