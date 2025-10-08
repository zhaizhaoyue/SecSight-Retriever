"""Lightweight dense retriever for the numeric AIE pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .segmentation import DocumentSegment

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Single retrieval hit."""

    segment: DocumentSegment
    score: float
    rank: int
    metadata: Dict[str, Optional[float]]


class DocumentRetriever:
    """Simple dense retriever using SentenceTransformer embeddings."""

    def __init__(self, config: Optional[Dict[str, any]] = None):
        config = config or {}
        self.model_name = config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.device = config.get("device", "cpu")
        self.top_k = int(config.get("top_k", 5))
        self.normalize = bool(config.get("normalize", True))
        self.batch_size = int(config.get("batch_size", 32))
        self.min_score = float(config.get("min_score", -1.0))

        logger.info("Loading sentence-transformer model: %s", self.model_name)
        self.encoder = SentenceTransformer(self.model_name, device=self.device)

        self._segments: List[DocumentSegment] = []
        self._embeddings: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def build_index(self, segments: Sequence[DocumentSegment]) -> None:
        if not segments:
            raise ValueError("No segments provided for indexing")
        self._segments = list(segments)
        texts = [seg.content for seg in self._segments]
        logger.debug("Encoding %d segments for retrieval", len(texts))
        embeddings = self.encoder.encode(
            texts, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False
        )
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            embeddings = embeddings / norms
        self._embeddings = embeddings.astype(np.float32)
        logger.info("Index built with %d segments", len(self._segments))

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        if not query:
            raise ValueError("Query must be a non-empty string")
        if self._embeddings is None or not len(self._segments):
            raise RuntimeError("Index has not been built. Call build_index first.")

        top_k = top_k or self.top_k
        if top_k <= 0:
            return []

        logger.debug("Encoding query for retrieval")
        query_emb = self.encoder.encode([query], convert_to_numpy=True)[0]
        if self.normalize:
            query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-12)

        scores = np.dot(self._embeddings, query_emb.astype(np.float32))
        if not isinstance(scores, np.ndarray):
            scores = np.asarray(scores)

        top_indices = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        sorted_indices = top_indices[np.argsort(-scores[top_indices])]

        results: List[RetrievalResult] = []
        for rank, idx in enumerate(sorted_indices, start=1):
            score = float(scores[idx])
            if score < self.min_score:
                continue
            results.append(
                RetrievalResult(
                    segment=self._segments[idx],
                    score=score,
                    rank=rank,
                    metadata={"score": score},
                )
            )

        logger.info("Retrieved %d segments for query", len(results))
        return results

    # ------------------------------------------------------------------
    def clear(self) -> None:
        self._segments = []
        self._embeddings = None


__all__ = ["DocumentRetriever", "RetrievalResult"]
