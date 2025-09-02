from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import math
import re

import faiss
import numpy as np

# —— BM25 为可选依赖（未安装则回退为 dense-only）
try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:
    BM25Okapi = None  # noqa: N816

# —— SentenceTransformer（设备自动回退）
from sentence_transformers import SentenceTransformer


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

        # 1) 读取索引与 meta
        index_path = self.index_dir / "hnsw.index"
        meta_path = self.index_dir / "meta.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_path}")

        self.index = faiss.read_index(str(index_path))
        self.metric_type = getattr(self.index, "metric_type", faiss.METRIC_L2)

        self.metas: List[Dict[str, Any]] = []
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                self.metas.append(json.loads(line))

        # 2) 加载编码器（设备自动回退）
        try:
            self.enc = SentenceTransformer(model_name, device=device)
        except Exception:
            # 例如无 CUDA 环境，自动退回 CPU
            self.enc = SentenceTransformer(model_name, device="cpu")

        self.dim = self.enc.get_sentence_embedding_dimension()

    def _encode_q(self, q: str) -> np.ndarray:
        v = self.enc.encode([q], normalize_embeddings=True, show_progress_bar=False)
        return v.astype("float32")

    def _passes_filter(
        self,
        m: Dict[str, Any],
        ticker: Optional[str],
        year: Optional[int],
        form: Optional[str],
    ) -> bool:
        if ticker and str(m.get("ticker", "")).upper() != ticker.upper():
            return False
        if year is not None:
            # 元数据里常见字段名是 fy（财政年）。保留你原有的严格过滤。
            fy = m.get("fy")
            if fy is None or int(fy) != int(year):
                return False
        if form and str(m.get("form", "")).upper() != form.upper():
            return False
        return True

    def _faiss_dist_to_score(self, dist: float) -> float:
        """
        将 FAISS 返回的距离转换为“越大越好”的相似度分数：
        - Inner Product：dist 已是相似度（越大越好）
        - L2：距离越小越相似 → 取负号作为分数
        """
        if self.metric_type == faiss.METRIC_INNER_PRODUCT:
            return float(dist)
        # 其他情况一律按距离越小越好 → 取负
        return -float(dist)

    def search(
        self,
        query: str,
        top_k: int = 50,
        ticker: Optional[str] = None,
        year: Optional[int] = None,
        form: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """返回 [{'text':..., 'score':..., 'meta':{...}, 'faiss_id': int}]"""
        qv = self._encode_q(query)
        # 多取一些再做过滤
        D, I = self.index.search(qv, max(top_k * 4, top_k))
        cand: List[Dict[str, Any]] = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            m = self.metas[idx]
            if not self._passes_filter(m, ticker, year, form):
                continue

            score = self._faiss_dist_to_score(float(dist))
            cand.append(
                {
                    "text": m.get("text_preview") or m.get("text") or m.get("raw_text") or "",
                    "score": score,
                    "meta": m,
                    "faiss_id": int(idx),
                }
            )
            if len(cand) >= top_k:
                break
        return cand


class HybridRetriever(FaissRetriever):
    """
    dense + BM25；RRF 融合
    如果没有可用文本或未安装 rank_bm25，则自动回退 dense-only
    """
    def __init__(
        self,
        index_dir: str | Path,
        corpus_texts: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(index_dir, **kwargs)

        # 1) 收集文本
        if corpus_texts is None:
            texts_raw: List[str] = []
            for m in self.metas:
                t = m.get("text") or m.get("raw_text") or m.get("text_preview") or ""
                texts_raw.append(t or "")
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
        if bm25_docs and BM25Okapi is not None:
            self._bm25 = BM25Okapi(bm25_docs)
        else:
            self._bm25 = None  # 没有可用文本或未安装 rank_bm25

    # ----------------- 公共方法 -----------------

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
        """
        返回统一的 hits：[{chunk_id, score, snippet, meta, faiss_id}, ...]
        - 先 dense（faiss）
        - 若有 BM25 语料，则做 RRF 融合；否则直接 dense-only
        """
        # 1) dense
        dense_raw = super().search(query, top_k=rr_k_dense, ticker=ticker, year=year, form=form)
        dense = [self._ensure_item_schema(d) for d in dense_raw]

        # 2) 若无 BM25 → 直接返回 dense 前 top_k
        if self._bm25 is None:
            return dense[:top_k]

        # 3) BM25 候选（在 bm25_docs 空间）
        q_tokens = self._tokenize(query)
        scores_bm25 = self._bm25.get_scores(q_tokens)  # np.ndarray
        top_idx_local = np.argsort(scores_bm25)[::-1][:rr_k_bm25]

        bm25_cand: List[Dict[str, Any]] = []
        for local_i in top_idx_local:
            meta_i = self._bm25_meta_ids[local_i]  # 映射回 meta 索引
            m = self.metas[meta_i]
            if not self._passes_filter(m, ticker, year, form):
                continue
            bm25_cand.append(self._to_item(meta_i, float(scores_bm25[local_i])))

        # 4) RRF 融合
        c = 60.0

        def _key(item: Dict[str, Any], fallback_rank: int):
            # 尽量拿到可对齐的稳定 ID
            return (
                item.get("faiss_id")
                or item.get("meta", {}).get("_id")
                or item.get("chunk_id")
                or fallback_rank
            )

        rank_dense: Dict[Any, int] = {}
        for r, item in enumerate(dense, start=1):
            rank_dense[_key(item, r)] = r

        rank_bm25: Dict[Any, int] = {}
        for r, item in enumerate(bm25_cand, start=1):
            rank_bm25[_key(item, r)] = r

        ids = set(rank_dense.keys()) | set(rank_bm25.keys())

        fused: List[Dict[str, Any]] = []
        for k in ids:
            rrf = 0.0
            if k in rank_dense:
                rrf += 1.0 / (c + rank_dense[k])
            if k in rank_bm25:
                rrf += 1.0 / (c + rank_bm25[k])

            # 把 key 映射回 metas 下标
            meta_idx = self._resolve_meta_index(k)
            fused.append(self._to_item(meta_idx, rrf))

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:top_k]

    # ----------------- 内部工具 -----------------

    def _resolve_meta_index(self, any_id: Any) -> int:
        """
        将“_id 或 faiss_id 或 meta 下标”解析为 metas 的下标。
        若 faiss_id 就是 meta 下标，则直接返回。
        """
        try:
            any_id_int = int(any_id)
            if 0 <= any_id_int < len(self.metas):
                return any_id_int
        except Exception:
            pass
        for idx, m in enumerate(self.metas):
            if m.get("_id") == any_id:
                return idx
        # 最差回退
        return 0

    def _tokenize(self, s: str) -> List[str]:
        """
        英文/数字：按词切分；中文：粗粒度切成字符 + bigram。
        """
        s = (s or "").lower()
        toks_en = re.findall(r"[a-z0-9$%\.]+", s)
        zh = re.findall(r"[\u4e00-\u9fff]", s)
        zh_bi = [a + b for a, b in zip(zh, zh[1:])]
        return toks_en + zh + zh_bi

    def _to_item(self, meta_idx: int, score: float) -> Dict[str, Any]:
        m = self.metas[meta_idx]
        text = m.get("text") or m.get("raw_text") or m.get("text_preview") or ""
        return self._ensure_item_schema(
            {
                "chunk_id": m.get("chunk_id")
                or f"{m.get('accno','NA')}::{m.get('file_type','NA')}::idx::{meta_idx}",
                "score": float(score),
                "snippet": text[:600],
                "meta": m,
                "faiss_id": meta_idx,
            }
        )

    def _ensure_item_schema(self, item: Dict[str, Any]) -> Dict[str, Any]:
        out = {
            "chunk_id": item.get("chunk_id")
            or item.get("id")
            or item.get("meta", {}).get("chunk_id"),
            "score": float(item.get("score", 0.0)),
            "snippet": item.get("snippet") or item.get("text") or "",
            "meta": item.get("meta") or {},
            "faiss_id": item.get("faiss_id"),
        }
        # 补全常用 meta 字段（保持现有键名，不强设默认值）
        m = out["meta"]
        for k in ["_id", "ticker", "form", "fy", "fq", "page_no", "source_path", "accno", "file_type"]:
            if k not in m:
                m[k] = m.get(k)
        return out


# ------------------------------
# 对外暴露的函数式 wrapper
# ------------------------------
_INDEX_DIR = "data/index/faiss_bge_base_en"
_retriever_singleton: Optional[HybridRetriever] = None

def get_retriever() -> HybridRetriever:
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = HybridRetriever(index_dir=_INDEX_DIR)
    return _retriever_singleton

def hybrid_search(query: str, filters: Dict[str, Any], topk: int = 8) -> List[Dict[str, Any]]:
    r = get_retriever()
    return r.search_hybrid(
        query=query,
        top_k=topk,
        ticker=filters.get("ticker"),
        year=filters.get("year"),
        form=str(filters.get("form") or "").upper() if filters.get("form") else None,
        )