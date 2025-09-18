# src/rag/retriever_dense.py
"""
Pure dense (bi-encoder) retriever with optional Cross-Encoder rerank.

Usage:
python -m src.rag.retriever.dense `
  --q "What risks or opportunities did Coca-Cola outline in relation to emerging markets in 2022" `
  --ticker KO --form 10-K --year 2022 `
  --topk 8 --normalize `
  --content-dir data/chunked `
  --rerank --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --fusion linear --alpha 0.75 --pretopk-mult 5 --batch-size 16 --max-length 512
  
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# cross-encoder
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------- I/O helpers ----------------
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

def to_score(metric_type: int, dist: float) -> float:
    if metric_type == faiss.METRIC_INNER_PRODUCT:
        return float(dist)
    return 1.0 / (1.0 + float(dist))

def _iter_jsonl_files(base: Path) -> List[Path]:
    if base.is_file():
        return [base]
    return [p for p in base.rglob("*.jsonl")]

def fetch_contents(content_path: Optional[Path], chunk_ids: Set[str]) -> Dict[str, str]:
    if not content_path:
        return {}
    found: Dict[str, str] = {}
    files = _iter_jsonl_files(content_path)
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
                        txt = obj.get("content") or obj.get("text") or obj.get("raw_text") or obj.get("page_text") or obj.get("body") or ""
                        if txt:
                            found[cid] = txt
                            targets.remove(cid)
                            if not targets:
                                return found
        except Exception:
            continue
    return found

# ---------------- Cross-Encoder reranker ----------------
class CrossEncoderReranker:
    def __init__(self, model_name: str, device: Optional[str] = None, max_length: int = 512, batch_size: int = 16):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)

    def score(self, query: str, docs: List[str]) -> List[float]:
        scores: List[float] = []
        for i in range(0, len(docs), self.batch_size):
            batch_pairs = [(query, d) for d in docs[i:i+self.batch_size]]
            enc = self.tokenizer(
                batch_pairs, padding=True, truncation=True,
                return_tensors="pt", max_length=self.max_length
            )
            with torch.no_grad():
                outputs = self.model(**{k: v.to(self.device) for k, v in enc.items()})
                logits = outputs.logits
                if logits.shape[1] == 1:  # regression
                    s = logits.squeeze(-1).detach().cpu().tolist()
                else:  # classification
                    s = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
                scores.extend(s)
        return scores

# ---------------- utils for fusion ----------------
def minmax(x: List[float]) -> List[float]:
    if not x: return []
    lo, hi = float(min(x)), float(max(x))
    if hi - lo < 1e-12:
        return [0.5 for _ in x]
    return [(v - lo) / (hi - lo) for v in x]

def zscore(x: List[float]) -> List[float]:
    if not x: return []
    arr = np.asarray(x, dtype=np.float64)
    mu, std = float(arr.mean()), float(arr.std() + 1e-12)
    return [float((v - mu) / std) for v in arr]

def fuse_scores(bm: Optional[List[float]], de: List[float], ce: Optional[List[float]],
                mode: str = "ce", alpha: float = 1, norm: str = "minmax") -> List[float]:
    def _norm(v):
        if v is None: return None
        return minmax(v) if norm == "minmax" else zscore(v)
    de_n = _norm(de)
    ce_n = _norm(ce) if ce is not None else None
    if mode == "dense" or ce_n is None:
        return de_n
    if mode == "ce":
        return ce_n
    a = float(alpha)
    return [a * ce_n[i] + (1.0 - a) * de_n[i] for i in range(len(de_n))]

# ======================= NEW: DenseRetriever (最小改动抽类) =======================
class DenseRetriever:
    """
    可在程序内被 import 使用的检索器。保留与原 CLI 相同的行为与排序逻辑。
    """
    def __init__(
        self,
        index_dir: str = "data/index",
        faiss_path: Optional[str] = None,
        meta_path: Optional[str] = None,
        model: str = "BAAI/bge-base-en-v1.5",
        device: str = "cuda",
        normalize: bool = False,
        pretopk_mult: int = 5,
        batch_size: int = 16,
        max_length: int = 512,
        rerank_model_default: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        fusion: str = "ce",
        alpha: float = 0.6,
    ):
        # 路径
        self.index_dir = Path(index_dir)
        self.faiss_path = Path(faiss_path) if faiss_path else (self.index_dir / "text_index.faiss")
        self.meta_path  = Path(meta_path)  if meta_path  else (self.index_dir / "meta.jsonl")
        if not self.faiss_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError(f"Missing files. faiss={self.faiss_path} meta={self.meta_path}")

        # 载入索引与元数据
        self.index = faiss.read_index(str(self.faiss_path))
        self.metas = load_metas(self.meta_path)
        if self.index.ntotal != len(self.metas):
            raise ValueError(f"index.ntotal={self.index.ntotal} != meta lines={len(self.metas)} (one-to-one required)")
        self.metric_type = getattr(self.index, "metric_type", faiss.METRIC_INNER_PRODUCT)

        # 可选 IDMap
        def _try_idmap_labels(index):
            try:
                arr = faiss.vector_to_array(getattr(index, "id_map"))
                return arr.astype(np.int64)
            except Exception:
                return None
        self.labels = _try_idmap_labels(self.index)
        self.label2meta = {int(lbl): self.metas[i] for i, lbl in enumerate(self.labels)} if self.labels is not None else None

        # 编码器
        try:
            self.enc = SentenceTransformer(model, device=device)
        except Exception:
            self.enc = SentenceTransformer(model, device="cpu")

        # 运行参数（保持与原脚本一致）
        self.device = device
        self.normalize = bool(normalize)
        self.pretopk_mult = int(pretopk_mult)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.rerank_model_default = rerank_model_default
        self.fusion = fusion
        self.alpha = float(alpha)

    def search(
        self,
        query: str,
        topk: int = 8,
        *,
        ticker: Optional[str] = None,
        form: Optional[str] = None,
        year: Optional[int] = None,
        content_path: Optional[str] = None,
        content_dir: Optional[str] = None,
        rerank: bool = False,
        rerank_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        返回结构化结果列表（已按最终分数降序）：
        [{
           "rank": int,
           "final_score": float,
           "dense_score": float,
           "ce_score": Optional[float],
           "id": str,
           "meta": dict,
           "snippet": str,
         }, ...]
        """
        # 3) encode query
        qvec = self.enc.encode([query], normalize_embeddings=self.normalize, show_progress_bar=False)
        qvec = qvec.astype("float32")

        # 4) vector search
        k_search = max(topk * max(self.pretopk_mult, 1) * 5, 2000)
        D, I = self.index.search(qvec, k_search)

        # 5) filter by meta
        tick = ticker.upper() if ticker else None
        frm  = form.upper() if form else None

        candidates: List[Tuple[float, Dict[str, Any], int]] = []  # (dense_score, meta, faiss_id)
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            m = self.label2meta.get(int(idx)) if self.label2meta is not None else (self.metas[int(idx)] if 0 <= idx < len(self.metas) else None)
            if m is None:
                continue
            if tick and str(m.get("ticker","")).upper() != tick:
                continue
            if frm  and str(m.get("form","")).upper() != frm:
                continue
            if year is not None:
                try:
                    if int(m.get("fy")) != int(year):
                        continue
                except Exception:
                    continue
            candidates.append((to_score(self.metric_type, float(dist)), m, int(idx)))
            if len(candidates) >= topk * max(self.pretopk_mult, 1):
                break

        if not candidates:
            return []

        # 6) contents for snippets + CE inputs
        content_root: Optional[Path] = None
        if content_path:
            content_root = Path(content_path)
        elif content_dir:
            content_root = Path(content_dir)

        contents: Dict[str, str] = {}
        if content_root:
            ids_needed = { m.get("chunk_id") for _, m, _ in candidates if m.get("chunk_id") }
            contents = fetch_contents(content_root, ids_needed)

        # prepare texts for CE (fallback to title if no content)
        ce_docs: List[str] = []
        for _, m, _ in candidates:
            cid = m.get("chunk_id")
            text = contents.get(cid) or m.get("title") or ""
            if len(text) > 4000:
                text = text[:4000]
            ce_docs.append(text if text else " ")

        dense_scores = [c[0] for c in candidates]

        # 7) optional rerank
        ce_scores: Optional[List[float]] = None
        if rerank:
            reranker = CrossEncoderReranker(
                model_name=(rerank_model or self.rerank_model_default),
                device=(self.device if self.device in {"cuda","cpu"} else None),
                max_length=self.max_length,
                batch_size=self.batch_size
            )
            ce_scores = reranker.score(query, ce_docs)

        # 8) final scoring & sort
        final_scores = fuse_scores(
            bm=None, de=dense_scores, ce=ce_scores,
            mode=self.fusion, alpha=self.alpha, norm="minmax"
        )
        ranked = sorted(
            list(zip(final_scores, candidates, ce_scores if ce_scores else [None]*len(candidates))),
            key=lambda x: x[0], reverse=True
        )[:topk]

        # 9) build structured records
        records: List[Dict[str, Any]] = []
        for i, (fsc, (dsc, m, fid), csc) in enumerate(ranked, 1):
            cid = m.get("chunk_id")
            snippet = contents.get(cid) or m.get("title") or ""
            records.append({
                "rank": i,
                "final_score": float(fsc),
                "dense_score": float(dsc),
                "ce_score": (float(csc) if csc is not None else None),
                "id": cid,
                "meta": m,
                "snippet": snippet[:500].replace("\n"," ")
            })
        return records

# ---------------- main (保持原CLI，但内部改用 DenseRetriever) ----------------
def main():
    ap = argparse.ArgumentParser(description="Dense retriever with optional cross-encoder rerank")
    ap.add_argument("--index-dir", default="data/index", help="folder with text_index.faiss + meta.jsonl")
    ap.add_argument("--faiss", default=None, help="override faiss path")
    ap.add_argument("--meta", default=None, help="override meta path")
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--normalize", action="store_true",
                    help="normalize query embeddings (use if index built with normalized vectors + inner-product)")
    ap.add_argument("--q", "--query", dest="query", required=True)
    ap.add_argument("--topk", type=int, default=8)

    # filters
    ap.add_argument("--ticker")
    ap.add_argument("--year", type=int)
    ap.add_argument("--form")

    # content for snippets
    ap.add_argument("--content-path", default=None)
    ap.add_argument("--content-dir", default=None)

    # rerank options
    ap.add_argument("--rerank", dest="rerank", action="store_true", default=True)
    ap.add_argument("--no-rerank", dest="rerank", action="store_false")
    ap.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--fusion", choices=["ce", "dense", "linear"], default="ce",
                    help="final scoring mode after rerank")
    ap.add_argument("--alpha", type=float, default=0.6, help="linear fusion weight for CE (alpha*ce + (1-alpha)*dense)")
    ap.add_argument("--pretopk-mult", type=int, default=5, help="collect topk*mult candidates before rerank")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)

    # (可选) JSON 输出，便于 Hybrid 调用
    ap.add_argument("--json-out", action="store_true", help="print JSON array records instead of pretty text")
    args = ap.parse_args()

    retr = DenseRetriever(
        index_dir=args.index_dir,
        faiss_path=args.faiss,
        meta_path=args.meta,
        model=args.model,
        device=args.device,
        normalize=args.normalize,
        pretopk_mult=args.pretopk_mult,
        batch_size=args.batch_size,
        max_length=args.max_length,
        fusion=args.fusion,
        alpha=args.alpha,
    )

    records = retr.search(
        query=args.query,
        topk=args.topk,
        ticker=args.ticker,
        form=args.form,
        year=args.year,
        content_path=args.content_path,
        content_dir=args.content_dir,
        rerank=args.rerank,
        rerank_model=args.rerank_model
    )

    if not records:
        print("[INFO] No hits after filters. Try removing filters or --topk larger.")
        return

    if args.json_out:
        print(json.dumps(records, ensure_ascii=False))
        return

    # 保持原有人类可读输出风格
    print(f"Query: {args.query}")
    print("="*80)
    for r in records:
        m = r["meta"]
        cid = r["id"]
        ce_part = (f" | ce={r['ce_score']:.4f}" if r["ce_score"] is not None else "")
        print(f"[{r['rank']:02d}] score={r['final_score']:.4f} | dense={r['dense_score']:.4f}{ce_part} "
              f"| {m.get('ticker')} {m.get('fy')} {m.get('form')} | chunk={m.get('chunk_index')} | id={cid}")
        print("     ", r["snippet"])
        print("-"*80)

if __name__ == "__main__":
    main()
