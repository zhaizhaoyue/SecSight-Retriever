from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import math
import re

class FaissRetriever:
    """
    纯向量检索器：加载 hnsw.index + meta.jsonl
    支持按 ticker/year/form 等元数据过滤；支持返回 top_k 候选（含 meta）
    """
    def __init__(
        self,
        index_dir: str | Path,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cuda",
    ):
        self.index_dir = Path(index_dir)
        self.index = faiss.read_index(str(self.index_dir / "hnsw.index"))
        self.metas: List[Dict[str, Any]] = []
        with (self.index_dir / "meta.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                self.metas.append(json.loads(line))
        self.enc = SentenceTransformer(model_name, device=device)
        self.dim = self.enc.get_sentence_embedding_dimension()

    def _encode_q(self, q: str) -> np.ndarray:
        v = self.enc.encode([q], normalize_embeddings=True, show_progress_bar=False)
        return v.astype("float32")

    def _passes_filter(self, m: Dict[str, Any],
                       ticker: Optional[str],
                       year: Optional[int],
                       form: Optional[str]) -> bool:
        if ticker and str(m.get("ticker", "")).upper() != ticker.upper():
            return False
        if year and int(m.get("year", -1)) != int(year):
            return False
        if form and str(m.get("form", "")).upper() != form.upper():
            return False
        return True

    def search(
        self,
        query: str,
        top_k: int = 50,
        ticker: Optional[str] = None,
        year: Optional[int] = None,
        form: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """返回 [{'text':..., 'score':..., 'meta':{...}}]"""
        qv = self._encode_q(query)
        D, I = self.index.search(qv, top_k * 4)  # 多取一些再做过滤
        cand = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            m = self.metas[idx]
            if not self._passes_filter(m, ticker, year, form):
                continue
            # HNSWFlat + 归一化 → 距离/相似度方向取决于index；统一做一个“越大越好”的分数
            score = float(dist)  # 如需统一为相似度，可用 sim = -dist 或直接用 dist
            cand.append({
                "text": m.get("text_preview") or "",  # 有些人把预览放meta里；若没有可不返回
                "score": score,
                "meta": m,
                "faiss_id": int(idx),
            })
            if len(cand) >= top_k:  # 过滤后凑够 top_k 就停
                break
        return cand


# 可选：pip install rank-bm25


class HybridRetriever(FaissRetriever):
    """
    dense + BM25；RRF 融合
    如果 meta.jsonl 没有 text，会跳过空文档；若全空则回退到 dense-only
    """
    def __init__(self, index_dir: str | Path, corpus_texts: Optional[List[str]] = None, **kwargs):
        super().__init__(index_dir, **kwargs)

        # 1) 收集文本
        if corpus_texts is None:
            texts_raw = []
            for m in self.metas:
                t = m.get("text") or m.get("raw_text") or m.get("text_preview") or ""
                texts_raw.append(t)
        else:
            texts_raw = corpus_texts

        # 2) 过滤空文本并建立映射：bm25_id -> meta_id
        self._bm25_meta_ids: List[int] = []
        bm25_docs: List[List[str]] = []
        for meta_id, t in enumerate(texts_raw):
            toks = self._tokenize(t)
            if not toks:
                continue
            self._bm25_meta_ids.append(meta_id)
            bm25_docs.append(toks)

        # 3) 构建 BM25 或标记为空
        if bm25_docs:
            self._bm25 = BM25Okapi(bm25_docs)
        else:
            self._bm25 = None  # 标记：没有可用文本

    @staticmethod
    def _tokenize(s: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9$%\.]+", s.lower())

    def search_hybrid(
        self,
        query: str,
        top_k: int = 50,
        ticker: Optional[str] = None,
        year: Optional[int] = None,
        form: Optional[str] = None,
        rr_k_dense: int = 100,
        rr_k_bm25: int = 100,
    ) -> List[Dict[str, Any]]:
        # 1) dense
        dense = super().search(query, top_k=rr_k_dense, ticker=ticker, year=year, form=form)

        # 2) 若无 BM25 语料，直接返回 dense
        if self._bm25 is None:
            return dense[:top_k]

        # BM25 分数（在 bm25_docs 空间）
        scores_bm25 = self._bm25.get_scores(self._tokenize(query))
        top_idx_local = np.argsort(scores_bm25)[::-1][:rr_k_bm25]

        bm25_cand = []
        for local_i in top_idx_local:
            meta_i = self._bm25_meta_ids[local_i]  # 映射回 meta 索引
            m = self.metas[meta_i]
            if not self._passes_filter(m, ticker, year, form):
                continue
            bm25_cand.append({
                "text": m.get("text") or m.get("raw_text") or m.get("text_preview") or "",
                "score": float(scores_bm25[local_i]),
                "meta": m,
                "faiss_id": int(meta_i),
            })

        # 3) RRF
        c = 60.0
        rank_dense = {item["faiss_id"]: r for r, item in enumerate(dense, start=1)}
        rank_bm25 = {item["faiss_id"]: r for r, item in enumerate(bm25_cand, start=1)}
        ids = set(rank_dense) | set(rank_bm25)

        fused = []
        for i in ids:
            s = 0.0
            if i in rank_dense: s += 1.0 / (c + rank_dense[i])
            if i in rank_bm25: s += 1.0 / (c + rank_bm25[i])
            m = self.metas[i]
            fused.append({
                "text": m.get("text") or m.get("raw_text") or m.get("text_preview") or "",
                "score": float(s),
                "meta": m,
                "faiss_id": int(i),
            })

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:top_k]

