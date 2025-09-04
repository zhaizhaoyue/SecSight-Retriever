from __future__ import annotations
from typing import List, Dict, Any, Optional

from .metadata import MappingStore
from .bm25 import BM25Retriever
from .filters import filter_hits
from .reranker import RuleReranker

# We assume you already have DenseRetriever with .search(query, top_k, ticker, year, form)
# returning a list of hits in the same schema: {chunk_id, score_dense, meta, title, content, heading, snippet}


def _by_id(hits: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {h["chunk_id"]: h for h in hits}


def _merge_hits(dense_hits: List[Dict[str, Any]], sparse_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    d_map = _by_id(dense_hits)
    all_ids = set(d_map.keys())
    for h in sparse_hits:
        all_ids.add(h["chunk_id"])
    merged: List[Dict[str, Any]] = []
    for cid in all_ids:
        dd = d_map.get(cid)
        ss = next((h for h in sparse_hits if h["chunk_id"] == cid), None)
        base = dd or ss
        merged.append({
            "chunk_id": cid,
            "meta": base.get("meta", {}),
            "title": base.get("title"),
            "content": base.get("content"),
            "heading": base.get("heading"),
            "snippet": base.get("snippet"),
            "score_dense": (dd or {}).get("score_dense", 0.0),
            "score_sparse": (ss or {}).get("score_sparse", 0.0),
        })
    return merged


class HybridRetriever:
    def __init__(self, dense_retriever, bm25_retriever: BM25Retriever, mappings: MappingStore):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.reranker = RuleReranker(mappings)

    def search(self, query: str, top_k: int = 8, ticker=None, year=None, form=None) -> List[Dict[str, Any]]:
        dense_hits = self.dense.search(query, top_k=top_k*3, ticker=ticker, year=year, form=form)
        sparse_hits = self.bm25.search(query, top_k=top_k*3, ticker=ticker, year=year, form=form)
        merged = _merge_hits(dense_hits, sparse_hits)
        # optional meta filtering (in case dense/sparse produced extras)
        merged = filter_hits(merged, ticker=ticker, year=year, form=form)
        reranked = self.reranker.rerank(query, merged)
        return reranked[:top_k]
