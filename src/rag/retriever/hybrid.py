# src/rag/retriever/hybrid.py
# -*- coding: utf-8 -*-
"""
Hybrid retriever: BM25F + Dense (FAISS) with z-score fusion.

Example:
python -m src.rag.retriever.hybrid `
  --q "How did foreign exchange rates impact Tesla’s financial results in 2023?" `
  --ticker TSLA --form 10-K --year 2023 `
  --topk 8 --alpha 0.5 `
  --index-dir data/index --faiss data/index/text_index.faiss --meta data/index/meta.jsonl `
  --content-dir data/chunked `
  --model BAAI/bge-base-en-v1.5 --normalize

"""

from __future__ import annotations
import argparse, json, math, re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---- import BM25F components from your file ----
# 若你的包结构允许直接导入，保留如下 import；
# 如果不允许，也可以把所需函数复制到本文件（下方已提供同名备份实现）。
try:
    from src.rag.retriever.bm25 import (
        BM25FConfig, BM25FRetriever, best_snippet as bm25_best_snippet
    )
    _HAS_BM25_IMPORT = True
except Exception:
    _HAS_BM25_IMPORT = False

# ----------------- local helpers (与 dense / bm25f 脚本保持一致) -----------------
def load_metas(meta_path: Path) -> List[Dict[str, Any]]:
    metas: List[Dict[str, Any]] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            if isinstance(raw.get("meta"), dict):
                flat = dict(raw)
                inner = flat.pop("meta") or {}
                for k, v in inner.items():
                    flat.setdefault(k, v)
            else:
                flat = raw
            flat.setdefault("chunk_id", flat.get("chunk_id") or flat.get("id"))
            flat.setdefault("file_type", flat.get("file_type") or "text")
            flat.setdefault("title", flat.get("title") or "")
            metas.append(flat)
    return metas

def _iter_text_chunk_files(base: Path) -> List[Path]:
    if base.is_file():
        # 仅当传入的就是 text_chunked.jsonl 才接受
        return [base] if base.name.lower() == "text_chunked.jsonl" else []
    # 递归匹配所有 text_chunked.jsonl
    return [p for p in base.rglob("text_chunked.jsonl")]

def fetch_contents(content_path: Optional[Path], chunk_ids: Set[str]) -> Dict[str, str]:
    if not content_path:
        return {}
    found: Dict[str, str] = {}
    files = _iter_text_chunk_files(content_path)   # ← 这里改为只扫 text_chunked.jsonl
    targets = set([cid for cid in chunk_ids if cid])
    if not targets or not files:
        return {}

    for fp in files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    cid = obj.get("chunk_id") or obj.get("id") or obj.get("chunkId")
                    if cid in targets:
                        txt = (obj.get("content") or obj.get("text") or obj.get("raw_text")
                               or obj.get("page_text") or obj.get("body") or "")
                        if txt:
                            found[cid] = txt
                            targets.remove(cid)
                            if not targets:  # 找齐提前返回
                                return found
        except Exception:
            continue
    return found

_TOKEN_RE = re.compile(r"[A-Za-z0-9%$€¥£\.-]+", re.UNICODE)

def best_snippet(raw: str, q_terms: Set[str], win_chars: int = 700) -> str:
    # 复用 bm25f 的方法（若可导入），否则本地简版
    if _HAS_BM25_IMPORT:
        return bm25_best_snippet(raw, q_terms, win_chars=win_chars)
    txt = (raw or "").replace("\n", " ").strip()
    if len(txt) <= win_chars:
        return txt
    words = txt.split()
    qset = {t.lower().strip(".,;:()") for t in q_terms if t}
    positions = [i for i, w in enumerate(words) if w.lower().strip(".,;:()") in qset]
    if not positions:
        return txt[:win_chars]
    best_score, best_l = -1, 0
    for p in positions:
        L = max(0, p - 20); R = min(len(words), p + 20)
        score = sum(1 for w in words[L:R] if w.lower().strip(".,;:()") in qset)
        if score > best_score:
            best_score, best_l = score, L
    snippet = " ".join(words[best_l:best_l + 80])
    return snippet[:win_chars]

def to_score(metric_type: int, dist: float) -> float:
    if metric_type == faiss.METRIC_INNER_PRODUCT:
        return float(dist)
    return 1.0 / (1.0 + float(dist))

def zscore_normalize(values: List[float]) -> List[float]:
    """
    对一个检索通道的候选分数做 z-score 标准化。
    - 若样本过少或方差接近 0，则回退为居中缩放：x' = x - mean。
    """
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return []
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    if std < 1e-9:
        return [float(v - mean) for v in arr]
    return [float((v - mean) / std) for v in arr]

# ----------------- Dense 通道：只用于 query 编码 + FAISS 搜索 -----------------
@dataclass
class DenseConfig:
    index_dir: str = "data/index"
    faiss: Optional[str] = None
    meta: Optional[str] = None
    model: str = "BAAI/bge-base-en-v1.5"
    device: str = "cuda"
    normalize: bool = False

class DenseSearcher:
    def __init__(self, cfg: DenseConfig):
        self.cfg = cfg
        index_dir = Path(cfg.index_dir)
        faiss_path = Path(cfg.faiss) if cfg.faiss else (index_dir / "text_index.faiss")
        meta_path  = Path(cfg.meta)  if cfg.meta  else (index_dir / "meta.jsonl")
        if not faiss_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Missing files. faiss={faiss_path} meta={meta_path}")

        self.index = faiss.read_index(str(faiss_path))
        self.metas = load_metas(meta_path)
        if self.index.ntotal != len(self.metas):
            raise ValueError(f"index.ntotal={self.index.ntotal} != meta lines={len(self.metas)}")
        self.metric_type = getattr(self.index, "metric_type", faiss.METRIC_INNER_PRODUCT)
        try:
            self.enc = SentenceTransformer(cfg.model, device=cfg.device)
        except Exception:
            self.enc = SentenceTransformer(cfg.model, device="cpu")

    def search(self, query: str, topk: int,
               ticker: Optional[str], form: Optional[str], year: Optional[int]) -> List[Dict[str, Any]]:
        qvec = self.enc.encode([query], normalize_embeddings=self.cfg.normalize, show_progress_bar=False)
        qvec = qvec.astype("float32")
        k_search = max(topk * 100, 2000)
        D, I = self.index.search(qvec, k_search)
        tick = ticker.upper() if ticker else None
        frm  = form.upper() if form else None
        yr   = year

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            m = self.metas[idx]
            if tick and str(m.get("ticker","")).upper() != tick:
                continue
            if frm and str(m.get("form","")).upper() != frm:
                continue
            if yr is not None:
                try:
                    if int(m.get("fy")) != int(yr):
                        continue
                except Exception:
                    continue
            results.append({
                "chunk_id": m.get("chunk_id"),
                "score": to_score(self.metric_type, float(dist)),
                "meta": m,
                "faiss_id": int(idx),
            })
            if len(results) >= topk:
                break
        return results

# ----------------- Hybrid 检索器 -----------------
@dataclass
class HybridConfig:
    # common
    index_dir: str = "data/index"
    meta: Optional[str] = None
    # content
    content_path: Optional[str] = None
    content_dir: Optional[str] = None
    # fusion
    alpha: float = 0.5  # 权重给 BM25F，(1 - alpha) 给 Dense
    # bm25f params (可按需透传)
    k1: float = 1.5
    b_title: float = 0.20
    b_body: float = 0.75
    w_title: float = 2.5
    w_body: float = 1.0
    min_token_len: int = 2
    # dense params
    model: str = "BAAI/bge-base-en-v1.5"
    device: str = "cuda"
    normalize: bool = False
    # candidate pool sizes
    bm25_pool: int = 64
    dense_pool: int = 64

class HybridRetriever:
    def __init__(self, cfg: HybridConfig, faiss_path: Optional[str] = None):
        self.cfg = cfg

        # ---- Dense searcher ----
        self.dense = DenseSearcher(DenseConfig(
            index_dir=cfg.index_dir,
            faiss=faiss_path,
            meta=cfg.meta,
            model=cfg.model,
            device=cfg.device,
            normalize=cfg.normalize
        ))

        # ---- BM25F retriever ----
        if _HAS_BM25_IMPORT:
            bm_cfg = BM25FConfig(
                index_dir=cfg.index_dir, meta=cfg.meta,
                content_path=cfg.content_path, content_dir=cfg.content_dir,
                k1=cfg.k1, b_title=cfg.b_title, b_body=cfg.b_body,
                w_title=cfg.w_title, w_body=cfg.w_body,
                min_token_len=cfg.min_token_len
            )
            self.bm25 = BM25FRetriever(bm_cfg)
        else:
            # 如果无法 import，你需要把 BM25FRetriever 复制进来或确保可导入
            raise ImportError("Cannot import BM25FRetriever. Ensure src.rag.retriever.bm25f is importable.")

        # 内容源
        if cfg.content_path:
            self.content_root = Path(cfg.content_path)
        elif cfg.content_dir:
            self.content_root = Path(cfg.content_dir)
        else:
            self.content_root = None

    def _collect_candidates(
        self, query: str, ticker: Optional[str], form: Optional[str], year: Optional[int]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        bm_hits = self.bm25.search(query, topk=self.cfg.bm25_pool, ticker=ticker, form=form, year=year)
        dn_hits = self.dense.search(query, topk=self.cfg.dense_pool, ticker=ticker, form=form, year=year)
        return bm_hits, dn_hits

    def _zscore_fuse(
        self,
        bm_hits: List[Dict[str, Any]],
        dn_hits: List[Dict[str, Any]],
        alpha: float
    ) -> List[Tuple[str, float]]:
        """
        输入两路命中列表（各自包含 chunk_id + 原始 score）。
        输出：[(chunk_id, fused_score), ...]，按 fused_score 降序。
        """
        # 取各自的分数并 z-score
        bm_scores = [h["score"] for h in bm_hits]
        dn_scores = [h["score"] for h in dn_hits]
        bm_z = zscore_normalize(bm_scores)
        dn_z = zscore_normalize(dn_scores)

        # 建立映射
        bm_map: Dict[str, float] = {}
        for h, z in zip(bm_hits, bm_z):
            cid = h.get("chunk_id")
            if cid:
                bm_map[cid] = z

        dn_map: Dict[str, float] = {}
        for h, z in zip(dn_hits, dn_z):
            cid = h.get("chunk_id")
            if cid:
                dn_map[cid] = z

        # 候选并集
        all_ids: Set[str] = set(bm_map.keys()) | set(dn_map.keys())

        fused: List[Tuple[str, float]] = []
        for cid in all_ids:
            z_bm = bm_map.get(cid, None)
            z_dn = dn_map.get(cid, None)

            # 缺失的一路赋予一个较低的惩罚值（-3），避免被完全抹除
            if z_bm is None: z_bm = -3.0
            if z_dn is None: z_dn = -3.0

            fused_score = alpha * z_bm + (1.0 - alpha) * z_dn
            fused.append((cid, float(fused_score)))

        fused.sort(key=lambda x: x[1], reverse=True)
        return fused

    def search(self, query: str, topk: int,
               ticker: Optional[str] = None, form: Optional[str] = None, year: Optional[int] = None) -> List[Dict[str, Any]]:
        bm_hits, dn_hits = self._collect_candidates(query, ticker, form, year)

        if not bm_hits and not dn_hits:
            return []

        fused = self._zscore_fuse(bm_hits, dn_hits, self.cfg.alpha)
        fused = fused[:max(1, topk)]

        # 组装最终输出（优先从 bm25 命中条目拿 meta/snippet，否则从 dense）
        cid2bm = {h["chunk_id"]: h for h in bm_hits if h.get("chunk_id")}
        cid2dn = {h["chunk_id"]: h for h in dn_hits if h.get("chunk_id")}

        # 收集需要的正文
        need_ids = {cid for cid, _ in fused}
        id2txt: Dict[str, str] = {}
        if self.content_root:
            id2txt = fetch_contents(self.content_root, need_ids)

        # query terms for snippet
        q_tokens = {t.lower() for t in _TOKEN_RE.findall(query)}
        out: List[Dict[str, Any]] = []
        for rank, (cid, fused_score) in enumerate(fused, 1):
            src_hit = cid2bm.get(cid) or cid2dn.get(cid)
            meta = (cid2bm.get(cid) or cid2dn.get(cid) or {}).get("meta", {})
            raw = id2txt.get(cid, "")
            snippet = raw and best_snippet(raw, q_tokens, win_chars=500) or (meta.get("title") or "")
            out.append({
                "rank": rank,
                "score": float(fused_score),
                "chunk_id": cid,
                "meta": meta,
                "snippet": snippet,
                "source": "hybrid(zscore)",
                "bm25_score": float(cid2bm[cid]["score"]) if cid in cid2bm else None,
                "dense_score": float(cid2dn[cid]["score"]) if cid in cid2dn else None,
            })
        return out

# ----------------- CLI -----------------
def _cli():
    ap = argparse.ArgumentParser(description="Hybrid retriever: BM25F + Dense (z-score fusion)")
    # common index/meta/content
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--content-path", default=None)
    ap.add_argument("--content-dir", default=None)
    # query & filters
    ap.add_argument("--q", "--query", dest="query", required=True)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--ticker"); ap.add_argument("--form"); ap.add_argument("--year", type=int)
    # fusion
    ap.add_argument("--alpha", type=float, default=0.5, help="weight for BM25F after z-score")
    ap.add_argument("--bm25-pool", type=int, default=64)
    ap.add_argument("--dense-pool", type=int, default=64)
    # bm25f params
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b-title", type=float, default=0.20)
    ap.add_argument("--b-body", type=float, default=0.75)
    ap.add_argument("--w-title", type=float, default=2.5)
    ap.add_argument("--w-body", type=float, default=1.0)
    ap.add_argument("--min-token-len", type=int, default=2)
    # dense params
    ap.add_argument("--faiss", default=None, help="path to text_index.faiss")
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--normalize", action="store_true")

    args = ap.parse_args()

    cfg = HybridConfig(
        index_dir=args.index_dir, meta=args.meta,
        content_path=args.content_path, content_dir=args.content_dir,
        alpha=args.alpha,
        k1=args.k1, b_title=args.b_title, b_body=args.b_body,
        w_title=args.w_title, w_body=args.w_body, min_token_len=args.min_token_len,
        model=args.model, device=args.device, normalize=args.normalize,
        bm25_pool=max(1, args.bm25_pool), dense_pool=max(1, args.dense_pool),
    )

    retr = HybridRetriever(cfg, faiss_path=args.faiss)
    hits = retr.search(args.query, topk=args.topk, ticker=args.ticker, form=args.form, year=args.year)

    if not hits:
        print("[INFO] No hits.")
        return
    print(f"Query: {args.query}\n" + "="*80)
    for r in hits:
        m = r["meta"] or {}
        heading = " ".join(str(m.get(k,"")) for k in ("heading","title","section"))
        print(f"[{r['rank']:02d}] fused={r['score']:.6f} "
              f"(bm25={r['bm25_score'] if r['bm25_score'] is not None else 'NA'}, "
              f"dense={r['dense_score'] if r['dense_score'] is not None else 'NA'}) "
              f"| {m.get('ticker')} {m.get('fy')} {m.get('form')} "
              f"| chunk={m.get('chunk_index')} | id={r['chunk_id']}")
        print(f"     heading: {heading[:120]}")
        print(f"     {r['snippet']}")
        print("-"*80)

if __name__ == "__main__":
    _cli()
