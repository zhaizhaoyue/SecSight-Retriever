# src/rag/retriever_dense.py
"""
Pure dense (bi-encoder) retriever with optional Cross-Encoder rerank.

Usage:
python -m src.rag.retriever.dense `
  --q "Tell me about Google’s expenses related to R&D." `
  --ticker GOOGL --form 10-K --year 2024 `
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

# ---------------- main ----------------
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
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--fusion", choices=["ce", "dense", "linear"], default="ce",
                    help="final scoring mode after rerank")
    ap.add_argument("--alpha", type=float, default=0.6, help="linear fusion weight for CE (alpha*ce + (1-alpha)*dense)")
    ap.add_argument("--pretopk-mult", type=int, default=5, help="collect topk*mult candidates before rerank")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    faiss_path = Path(args.faiss) if args.faiss else (index_dir / "text_index.faiss")
    meta_path  = Path(args.meta)  if args.meta  else (index_dir / "meta.jsonl")
    if not faiss_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing files. faiss={faiss_path} meta={meta_path}")

    # 1) load FAISS & metas
    index = faiss.read_index(str(faiss_path))
    metas = load_metas(meta_path)
    if index.ntotal != len(metas):
        raise ValueError(f"index.ntotal={index.ntotal} != meta lines={len(metas)} (one-to-one required)")
    metric_type = getattr(index, "metric_type", faiss.METRIC_INNER_PRODUCT)

    # optional IDMap
    def try_get_labels_from_idmap(index):
        try:
            arr = faiss.vector_to_array(getattr(index, "id_map"))
            return arr.astype(np.int64)
        except Exception:
            return None

    labels = try_get_labels_from_idmap(index)
    label2meta = {int(lbl): metas[i] for i, lbl in enumerate(labels)} if labels is not None else None

    # 2) encoder (query)
    try:
        enc = SentenceTransformer(args.model, device=args.device)
    except Exception:
        enc = SentenceTransformer(args.model, device="cpu")

    # 3) encode query
    qvec = enc.encode([args.query], normalize_embeddings=args.normalize, show_progress_bar=False)
    qvec = qvec.astype("float32")

    # 4) vector search (NOTE: uses args.pretopk_mult)
    k_search = max(args.topk * max(args.pretopk_mult, 1) * 5, 2000)
    D, I = index.search(qvec, k_search)

    # 5) filter by meta
    tick = args.ticker.upper() if args.ticker else None
    form = args.form.upper() if args.form else None
    year = args.year

    candidates: List[Tuple[float, Dict[str, Any], int]] = []  # (dense_score, meta, faiss_id)
    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        m = label2meta.get(int(idx)) if label2meta is not None else (metas[int(idx)] if 0 <= idx < len(metas) else None)
        if m is None:
            continue
        if tick and str(m.get("ticker","")).upper() != tick:
            continue
        if form and str(m.get("form","")).upper() != form:
            continue
        if year is not None:
            try:
                if int(m.get("fy")) != int(year):
                    continue
            except Exception:
                continue
        candidates.append((to_score(metric_type, float(dist)), m, int(idx)))
        if len(candidates) >= args.topk * max(args.pretopk_mult, 1):
            break

    if not candidates:
        print("[INFO] No hits after filters. Try removing filters or --topk larger.")
        return

    # 6) contents for snippets + CE inputs
    content_root: Optional[Path] = None
    if args.content_path:
        content_root = Path(args.content_path)
    elif args.content_dir:
        content_root = Path(args.content_dir)

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
    if args.rerank:
        reranker = CrossEncoderReranker(
            model_name=args.rerank_model,
            device=(args.device if args.device in {"cuda","cpu"} else None),
            max_length=args.max_length,
            batch_size=args.batch_size
        )
        ce_scores = reranker.score(args.query, ce_docs)

    # 8) final scoring & sort
    final_scores = fuse_scores(
        bm=None, de=dense_scores, ce=ce_scores,
        mode=args.fusion, alpha=args.alpha, norm="minmax"
    )
    ranked = sorted(
        list(zip(final_scores, candidates, ce_scores if ce_scores else [None]*len(candidates))),
        key=lambda x: x[0], reverse=True
    )[:args.topk]

    # 9) print
    print(f"Query: {args.query}")
    print("="*80)
    for i, (fsc, (dsc, m, fid), csc) in enumerate(ranked, 1):
        cid = m.get("chunk_id")
        snippet = contents.get(cid) or m.get("title") or ""
        print(f"[{i:02d}] score={fsc:.4f} | dense={dsc:.4f}" + (f" | ce={csc:.4f}" if csc is not None else "") +
              f" | {m.get('ticker')} {m.get('fy')} {m.get('form')} | chunk={m.get('chunk_index')} | id={cid}")
        print("     ", snippet[:240].replace("\n"," "))
        print("-"*80)

if __name__ == "__main__":
    main()
