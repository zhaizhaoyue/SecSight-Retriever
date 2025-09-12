# -*- coding: utf-8 -*-
"""
BM25 retriever — TEXT-ONLY, compact & generalized

python -m src.rag.retriever.bm25_text `
  --q "Tell me about Google’s expenses related to Research and Development?" `
  --ticker GOOGL --form 10-K --year 2024 `
  --content-dir data/chunked --index-dir data/index 

"""

from __future__ import annotations
import argparse, json, math, re, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from collections import Counter, defaultdict

# [ADDED for cross-encoder]
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# =============================================================================
# Constants & small helpers
# =============================================================================
RID_RE   = re.compile(r"^\d{10}-\d{2}-\d{6}::text::chunk-\d+$")
CHUNK_RE = re.compile(r"::text::chunk-(\d+)$")

TOKEN_RE = re.compile(r"[A-Za-z0-9%$€¥£\.-]+", re.UNICODE)
STOPWORDS = frozenset({"the", "a", "an", "and", "or", "of", "in", "to", "for", "on", "by", "as", "at", "is", "are",
    "with", "its", "this", "that", "from", "which", "be", "we", "our", "their", "it", "was", "were", "has", "have",
    "had", "but", "not", "no", "can", "could", "would", "should", "may", "might", "will", "shall", "into", "than",
    "then", "there", "here", "also"})

# =============================================================================
# IO: meta & text
# =============================================================================
def _chunk_no(rid: str) -> str:
    m = CHUNK_RE.search(rid or ""); return m.group(1) if m else "NA"

def load_metas(meta_path: Path) -> List[Dict[str, Any]]:
    metas: List[Dict[str, Any]] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            if isinstance(raw.get("meta"), dict):
                flat = dict(raw); inner = flat.pop("meta") or {}
                for k, v in inner.items(): flat.setdefault(k, v)
            else:
                flat = raw
            rid = flat.get("id") or flat.get("chunk_id")
            flat["chunk_id"] = rid
            flat.setdefault("title", flat.get("title") or "")
            metas.append(flat)
    return metas

def audit_metas(metas: Sequence[Dict[str, Any]], strict: bool = False, tag: str = "meta.jsonl") -> Tuple[int,int,int]:
    bad, mismatch = 0, 0
    for i, m in enumerate(metas, 1):
        rid = m.get("id") or m.get("chunk_id")
        if not rid or not RID_RE.match(str(rid)):
            bad += 1; print(f"[AUDIT]({tag}) bad rid format at line {i}: {rid}", file=sys.stderr); continue
        real = _chunk_no(rid)
        stored = m.get("chunk_index") or (m.get("meta", {}) or {}).get("chunk_index")
        if stored is not None:
            try:
                if str(int(stored)) != str(real):
                    mismatch += 1
                    print(f"[AUDIT]({tag}) chunk_index mismatch at line {i}: stored={stored} real={real} id={rid}", file=sys.stderr)
            except Exception:
                mismatch += 1
                print(f"[AUDIT]({tag}) chunk_index not int at line {i}: stored={stored} id={rid}", file=sys.stderr)
    if strict and (bad or mismatch):
        raise ValueError(f"[AUDIT] strict failed: bad_format={bad}, chunknum_mismatch={mismatch}")
    return len(metas), bad, mismatch

def fetch_contents(content_root: Optional[Path], rids: Sequence[str], strict_id: bool = True) -> Dict[str, str]:
    if not content_root or not rids: return {}
    wanted = set(rids); found: Dict[str, str] = {}
    # 遍历 content_dir 下所有 jsonl，按 id 匹配文本
    for fp in ( [content_root] if content_root.is_file() else sorted(content_root.rglob("*.jsonl"), key=lambda x: (x.parent.as_posix(), x.name)) ):
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    key = obj.get("id") or obj.get("chunk_id") or obj.get("chunkId")
                    if not key: continue
                    if strict_id and not RID_RE.match(str(key)): continue
                    if key in wanted:
                        txt = obj.get("content") or obj.get("text") or obj.get("raw_text") or obj.get("page_text") or obj.get("body") or ""
                        if txt:
                            found[key] = txt; wanted.remove(key)
                            if not wanted: return found
        except Exception:
            continue
    for miss in sorted(wanted): print(f"[WARN] content not found for rid={miss}", file=sys.stderr)
    return found

# =============================================================================
# Tokenize & snippets
# =============================================================================
def tokenize(text: str, lower: bool = True, min_len: int = 2) -> List[str]:
    if not text: return []
    if lower: text = text.lower()
    return [t for t in TOKEN_RE.findall(text) if len(t) >= min_len and t not in STOPWORDS]

def best_snippet(raw: str, q_terms: set[str], win_chars: int = 700) -> str:
    txt = (raw or "").replace("\n"," ").strip()
    if len(txt) <= win_chars: return txt
    words = txt.split(); qset = {t.lower().strip(".,;:()") for t in q_terms if t}
    pos = [i for i,w in enumerate(words) if w.lower().strip(".,;:()") in qset]
    if not pos: return txt[:win_chars]
    best, bestL = -1, 0
    for p in pos:
        L = max(0, p-20); R = min(len(words), p+20)
        score = sum(1 for w in words[L:R] if w.lower().strip(".,;:()") in qset)
        if score > best: best, bestL = score, L
    return " ".join(words[bestL: bestL+80])[:win_chars]

# =============================================================================
# BM25 (TEXT ONLY)
# =============================================================================
class BM25Index:
    def __init__(self, docs: Sequence[List[str]], k1=1.5, b=0.75) -> None:
        self.k1, self.b = float(k1), float(b)
        self.N = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avg_len = (sum(self.doc_len)/self.N) if self.N else 0.0
        self.inv: Dict[str, List[Tuple[int,int]]] = defaultdict(list)
        df = defaultdict(int)
        for i, toks in enumerate(docs):
            c = Counter(toks)
            for term, tf in c.items():
                self.inv[term].append((i, tf))
            for term in c.keys():
                df[term] += 1
        self.idf = {term: math.log((self.N - dfi + 0.5) / (dfi + 0.5) + 1.0) for term, dfi in df.items()}

    def _norm(self, i: int) -> float:
        L = self.doc_len[i]
        return self.k1 * (1 - self.b + self.b * (L / (self.avg_len + 1e-9)))

    def scores(self, q_tokens: Sequence[str]) -> List[float]:
        if not q_tokens or not self.N: return [0.0]*self.N
        acc: Dict[int, float] = defaultdict(float); seen = set()
        for term in q_tokens:
            if term in seen: continue
            seen.add(term)
            idf = self.idf.get(term)
            if idf is None: continue
            postings = self.inv.get(term, [])
            for i, tf in postings:
                denom = tf + self._norm(i)
                acc[i] += idf * ( (tf * (self.k1 + 1.0)) / (denom + 1e-12) )
        out = [0.0]*self.N
        for i,v in acc.items(): out[i] = float(v)
        return out

# =============================================================================
# Retriever (TEXT ONLY)
# =============================================================================
class CrossEncoderReranker:
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def rerank(self, query: str, docs: List[str]) -> List[float]:
        pairs = [(query, d) for d in docs]
        enc = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = self.model(**{k:v.to(self.device) for k,v in enc.items()})
            if outputs.logits.shape[1] == 1:
                scores = outputs.logits.squeeze(-1).cpu().tolist()
            else:
                scores = torch.softmax(outputs.logits, dim=1)[:,1].cpu().tolist()
        return scores

# =============================================================================
# Retriever (TEXT ONLY, with optional rerank)
# =============================================================================
@dataclass
class BM25TextConfig:
    index_dir: str = "data/index"
    meta: Optional[str] = None
    content_path: Optional[str] = None
    content_dir: Optional[str] = None
    min_token_len: int = 2
    k1: float = 1.5
    b: float = 0.75
    strict_id: bool = True
    audit_only: bool = False
    rerank: bool = False
    rerank_model: Optional[str] = None

class BM25TextRetriever:
    def __init__(self, cfg: BM25TextConfig):
        self.cfg = cfg
        meta_path = Path(cfg.meta) if cfg.meta else Path(cfg.index_dir) / "meta.jsonl"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.jsonl not found at {meta_path}")
        self.metas = load_metas(meta_path)
        total, bad, mis = audit_metas(self.metas, strict=False, tag=meta_path.name)
        if bad or mis:
            print(f"[AUDIT] total={total} bad_format={bad} chunknum_mismatch={mis}", file=sys.stderr)
        self.content_root = Path(cfg.content_path or cfg.content_dir) if (cfg.content_path or cfg.content_dir) else None

        self.reranker = None
        if cfg.rerank and cfg.rerank_model:
            print(f"[INFO] Loading cross-encoder reranker: {cfg.rerank_model}", file=sys.stderr)
            self.reranker = CrossEncoderReranker(cfg.rerank_model)

    def _filter_meta(self, ticker: Optional[str], form: Optional[str], year: Optional[int]) -> List[Dict[str, Any]]:
        tick = ticker.upper() if ticker else None
        frm  = form.upper() if form else None
        out: List[Dict[str, Any]] = []
        for m in self.metas:
            rid = m.get("id") or m.get("chunk_id")
            if not rid or (self.cfg.strict_id and not RID_RE.match(str(rid))):
                continue
            if tick and str(m.get("ticker","")).upper() != tick: continue
            if frm  and str(m.get("form",""  )).upper() != frm:  continue
            if year is not None:
                try:
                    if int(m.get("fy")) != int(year): continue
                except Exception: pass
            m["_rid"] = str(rid); out.append(m)
        return out

    def search(self, query: str, topk: int = 8, ticker: Optional[str] = None, form: Optional[str] = None, year: Optional[int] = None) -> List[Dict[str, Any]]:
        if self.cfg.audit_only:
            print("[INFO] audit-only mode: no search executed."); return []
        cands = self._filter_meta(ticker, form, year)
        if not cands: return []

        id2txt = fetch_contents(self.content_root, [m["_rid"] for m in cands], strict_id=self.cfg.strict_id)
        doc_metas, docs_tokens, raw_texts = [], [], []
        for m in cands:
            rid = m.get("_rid"); raw = id2txt.get(rid, "")
            if not raw: continue
            toks = tokenize(raw, min_len=self.cfg.min_token_len)
            if not toks: continue
            doc_metas.append(m); docs_tokens.append(toks); raw_texts.append(raw)
        if not docs_tokens: return []

        bm25 = BM25Index(docs_tokens, k1=self.cfg.k1, b=self.cfg.b)
        q_tokens = tokenize(query, min_len=self.cfg.min_token_len)
        base_scores = bm25.scores(q_tokens)

        results = [(i, s) for i,s in enumerate(base_scores) if s > 0.0]
        if not results: return []
        results.sort(key=lambda x: x[1], reverse=True)

        seen, dedup = set(), []
        for i, s in results:
            rid = doc_metas[i].get("_rid")
            if rid in seen: continue
            seen.add(rid); dedup.append((i,s))
            if len(dedup) >= max(1, topk*3): break   # [ADDED] take more for rerank
        q_terms = set(q_tokens)

        # build candidate outputs
        outs = []
        for i,s in dedup:
            rid, raw = doc_metas[i]["_rid"], raw_texts[i]
            snippet = best_snippet(raw, q_terms, 500)
            outs.append({
                "bm25_score": float(s),
                "id": rid,
                "chunk": _chunk_no(rid),
                "meta": doc_metas[i],
                "snippet": snippet,
                "raw": raw
            })

        # optional rerank
        if self.reranker:
            scores = self.reranker.rerank(query, [o["raw"] for o in outs])
            for o, sc in zip(outs, scores):
                o["rerank_score"] = float(sc)
            outs.sort(key=lambda x: x["rerank_score"], reverse=True)
        else:
            outs.sort(key=lambda x: x["bm25_score"], reverse=True)

        # trim to topk
        final = []
        for rank,o in enumerate(outs[:topk],1):
            final.append({
                "rank": rank,
                "score": o.get("rerank_score", o["bm25_score"]),
                "id": o["id"],
                "chunk": o["chunk"],
                "meta": o["meta"],
                "snippet": o["snippet"],
                "source": "bm25_text_rerank" if self.reranker else "bm25_text"
            })
        return final

# =============================================================================
# CLI
# =============================================================================
def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--content-path", default=None)
    ap.add_argument("--content-dir", default=None)
    ap.add_argument("--q", "--query", dest="query", required=True)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--ticker"); ap.add_argument("--form"); ap.add_argument("--year", type=int)
    ap.add_argument("--min-token-len", type=int, default=2)
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--strict-id", dest="strict_id", action="store_true", default=True)
    ap.add_argument("--no-strict-id", dest="strict_id", action="store_false")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--rerank", dest="rerank", action="store_true", default=True)
    ap.add_argument("--no-rerank", dest="rerank", action="store_false")
    ap.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="HuggingFace cross-encoder model name")
    args = ap.parse_args()

    cfg = BM25TextConfig(
        index_dir=args.index_dir, meta=args.meta,
        content_path=args.content_path, content_dir=args.content_dir,
        min_token_len=args.min_token_len, k1=args.k1, b=args.b,
        strict_id=args.strict_id, audit_only=args.audit_only,
        rerank=args.rerank, rerank_model=args.rerank_model
    )

    retr = BM25TextRetriever(cfg)
    if args.audit_only:
        print("[DONE] audit-only finished."); return

    hits = retr.search(args.query, topk=args.topk, ticker=args.ticker, form=args.form, year=args.year)
    if not hits:
        print("[INFO] No hits."); return

    print(f"Query: {args.query}\n" + "="*80)
    for r in hits:
        m = r["meta"]; heading = str(m.get("heading") or m.get("title") or m.get("section") or "")[:160]
        print(f"[{r['rank']:02d}] score={r['score']:.6f} | {m.get('ticker')} {m.get('fy')} {m.get('form')} | chunk={r['chunk']} | id={r['id']}")
        if heading: print(f"     heading: {heading}")
        print(f"     {r['snippet']}")
        print("-"*80)

if __name__ == "__main__":
    _cli()
