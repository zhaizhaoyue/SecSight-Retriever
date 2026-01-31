
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.retrieval.retriever.bm25_text import BM25TextConfig
from src.retrieval.retriever.dense import DenseRetriever
from src.retrieval.retriever.hybrid import HybridRetrieverRRF, CrossEncoderReranker
from src.retrieval.retriever.answer_api import LLMClient, answer_with_llm


@dataclass
class QueryRequest:
    query: str
    index_dir: Path
    content_dir: Optional[Path] = None
    bm25_meta: Optional[Path] = None
    dense_model: str = "BAAI/bge-base-en-v1.5"
    dense_device: str = "cpu"
    topk: int = 8
    bm25_topk: int = 200
    dense_topk: int = 200
    ce_candidates: int = 256
    rrf_k: float = 60.0
    rrf_w_bm25: float = 2.0
    rrf_w_dense: float = 2.0
    ce_weight: float = 0.5
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_device: Optional[str] = None
    ticker: Optional[str] = None
    form: Optional[str] = None
    year: Optional[int] = None
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    max_context_tokens: int = 2400
    strict_filters: bool = True


@dataclass
class QueryResult:
    query: str
    records: List[Dict[str, Any]]
    answer: Optional[Dict[str, Any]]


def _build_hybrid(req: QueryRequest) -> HybridRetrieverRRF:
    bm25_cfg = BM25TextConfig(
        index_dir=str(req.index_dir),
        meta=str(req.bm25_meta) if req.bm25_meta else None,
        content_dir=str(req.content_dir) if req.content_dir else None,
        content_path=str(req.content_dir) if req.content_dir and req.content_dir.is_file() else None,
    )

    dense = DenseRetriever(
        index_dir=str(req.index_dir),
        model=req.dense_model,
        device=req.dense_device,
    )

    reranker: Optional[CrossEncoderReranker] = None
    if req.ce_weight > 1e-6:
        reranker = CrossEncoderReranker(
            model_name=req.rerank_model,
            device=req.rerank_device,
        )

    return HybridRetrieverRRF(
        bm25_cfg=bm25_cfg,
        dense=dense,
        reranker=reranker,
        k=req.rrf_k,
        w_bm25=req.rrf_w_bm25,
        w_dense=req.rrf_w_dense,
        ce_weight=req.ce_weight,
    )


def run_query(req: QueryRequest) -> QueryResult:
    index_dir = req.index_dir
    if not index_dir.exists():
        raise FileNotFoundError(f"Index directory not found: {index_dir}")

    content_dir = req.content_dir
    if content_dir is not None and not content_dir.exists():
        raise FileNotFoundError(f"Chunk directory not found: {content_dir}")

    hybrid = _build_hybrid(req)

    filters: Dict[str, Any] = {}
    if req.ticker:
        filters["ticker"] = req.ticker
    if req.form:
        filters["form"] = req.form
    if req.year:
        filters["year"] = req.year

    records = hybrid.search(
        req.query,
        topk=req.topk,
        content_dir=str(content_dir) if content_dir else None,
        bm25_topk=req.bm25_topk,
        dense_topk=req.dense_topk,
        ce_candidates=req.ce_candidates,
        strict_filters=req.strict_filters,
        **filters,
    )

    answer: Optional[Dict[str, Any]] = None
    if req.llm_base_url and req.llm_model and req.llm_api_key:
        llm = LLMClient(
            base_url=req.llm_base_url,
            model=req.llm_model,
            api_key=req.llm_api_key,
        )
        answer = answer_with_llm(
            req.query,
            records,
            llm,
            max_ctx_tokens=req.max_context_tokens,
        )

    return QueryResult(query=req.query, records=records, answer=answer)
