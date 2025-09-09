# src/rag/retriever/hybrid.py
# -*- coding: utf-8 -*-
"""
python -m src.rag.retriever.hybrid `
  --q "How does Tesla describe risks related to supply chain disruptions?" `
  --ticker TSLA --form 10-K --year 2023 `
  --topk 8 `
  --index-dir data/index `
  --content-dir data/chunked `
  --model "BAAI/bge-base-en-v1.5" `
  --normalize `
  --w-dense 0.6 --w-bm25 0.4 `
  --strict-id

  """

from __future__ import annotations
import argparse, json, math, re, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterable, Tuple, Set
from collections import Counter, defaultdict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ===================== RID & audit utils =====================
RID_RE   = re.compile(r'^\d{10}-\d{2}-\d{6}::text::chunk-\d+$')
CHUNK_RE = re.compile(r'::text::chunk-(\d+)$')

def chunk_no_from_id(rid: str) -> str:
    m = CHUNK_RE.search(rid or "")
    return m.group(1) if m else "NA"

def canon_rid(m: Dict[str, Any]) -> Optional[str]:
    rid = m.get("id") or m.get("chunk_id")
    if not rid:
        return None
    return rid if RID_RE.match(str(rid)) else None

def audit_metas(metas: List[Dict[str, Any]], strict: bool = False) -> Tuple[int, int, int]:
    bad_format = 0
    mismatch = 0
    for i, m in enumerate(metas, 1):
        rid = m.get("id") or m.get("chunk_id")
        if not rid or not RID_RE.match(str(rid)):
            bad_format += 1
            print(f"[AUDIT] bad rid format at meta line {i}: {rid}", file=sys.stderr)
            continue
        real = chunk_no_from_id(rid)
        stored = m.get("chunk_index") or (m.get("meta", {}) or {}).get("chunk_index")
        if stored is not None:
            try:
                if str(int(stored)) != str(real):
                    mismatch += 1
                    print(f"[AUDIT] chunk_index mismatch at line {i}: stored={stored} real={real} id={rid}", file=sys.stderr)
            except Exception:
                mismatch += 1
                print(f"[AUDIT] chunk_index not int at line {i}: stored={stored} id={rid}", file=sys.stderr)
    total = len(metas)
    if strict and (bad_format or mismatch):
        raise ValueError(f"[AUDIT] strict failed: bad_format={bad_format}, chunknum_mismatch={mismatch}")
    return total, bad_format, mismatch

# ----------------- meta I/O -----------------
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
            rid = flat.get("id") or flat.get("chunk_id")
            flat["chunk_id"] = rid
            flat.setdefault("title", flat.get("title") or "")
            metas.append(flat)
    return metas

def _iter_jsonl_files(base: Path) -> List[Path]:
    if base.is_file():
        return [base]
    return sorted([p for p in base.rglob("*.jsonl")], key=lambda x: (x.parent.as_posix(), x.name))

def fetch_contents_strict(content_path: Optional[Path], rids: Set[str], strict_id: bool = True) -> Dict[str, str]:
    if not content_path or not rids:
        return {}
    wanted = set(rids)
    found: Dict[str, str] = {}
    for fp in _iter_jsonl_files(content_path):
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    key = obj.get("id") or obj.get("chunk_id") or obj.get("chunkId")
                    if not key:
                        continue
                    if strict_id and not RID_RE.match(str(key)):
                        continue
                    if key in wanted:
                        txt = (obj.get("content") or obj.get("text") or obj.get("raw_text")
                               or obj.get("page_text") or obj.get("body") or "")
                        if txt:
                            found[key] = txt
                            wanted.remove(key)
                            if not wanted:
                                return found
        except Exception:
            continue
    for miss in sorted(wanted):
        print(f"[WARN] content not found for rid={miss}", file=sys.stderr)
    return found

# ----------------- tokenization -----------------
_TOKEN_RE = re.compile(r"[A-Za-z0-9%$€¥£\.-]+", re.UNICODE)
STOPWORDS = {
    "the","a","an","and","or","of","in","to","for","on","by","as","at","is","are",
    "with","its","this","that","from","which","be","we","our","their","it","was",
    "were","has","have","had","but","not","no","can","could","would","should",
    "may","might","will","shall","into","than","then","there","here","also"
}

def tokenize(text: str, lower: bool = True, min_len: int = 2) -> List[str]:
    if not text:
        return []
    if lower:
        text = text.lower()
    toks = _TOKEN_RE.findall(text)
    toks = [t for t in toks if len(t) >= min_len and t not in STOPWORDS]
    return toks

# ----------------- snippets -----------------
def best_snippet(raw: str, q_terms: Set[str], win_chars: int = 700) -> str:
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

# ----------------- BM25F core -----------------
class BM25FIndex:
    def __init__(self,
                 docs: List[Tuple[List[str], List[str]]],
                 k1: float = 1.5,
                 b_title: float = 0.20,
                 b_body: float = 0.75,
                 w_title: float = 0.5,
                 w_body: float = 1.0):
        self.k1 = float(k1)
        self.b_title = float(b_title)
        self.b_body = float(b_body)
        self.w_title = float(w_title)
        self.w_body = float(w_body)
        self.N = len(docs)

        self.title_len = [len(t) for t, _ in docs]
        self.body_len  = [len(b) for _, b in docs]
        self.avg_t = (sum(self.title_len) / self.N) if self.N else 0.0
        self.avg_b = (sum(self.body_len) / self.N) if self.N else 0.0

        self.inv_t: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.inv_b: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        df = defaultdict(int)

        for i, (toks_t, toks_b) in enumerate(docs):
            ct, cb = Counter(toks_t), Counter(toks_b)
            for term, tf in ct.items():
                self.inv_t[term].append((i, tf))
            for term, tf in cb.items():
                self.inv_b[term].append((i, tf))
            terms_union = set(ct.keys()) | set(cb.keys())
            for term in terms_union:
                df[term] += 1

        self.idf: Dict[str, float] = {
            term: math.log((self.N - dfi + 0.5) / (dfi + 0.5) + 1.0)
            for term, dfi in df.items()
        }

    def _norm_title(self, i: int) -> float:
        return self.k1 * (1 - self.b_title + self.b_title * (self.title_len[i] / (self.avg_t + 1e-9)))

    def _norm_body(self, i: int) -> float:
        return self.k1 * (1 - self.b_body + self.b_body * (self.body_len[i] / (self.avg_b + 1e-9)))

    def scores(self, q_tokens: List[str]) -> List[float]:
        if not q_tokens or not self.N:
            return [0.0] * self.N
        acc: Dict[int, float] = defaultdict(float)
        seen_terms: Set[str] = set()

        for term in q_tokens:
            if term in seen_terms:
                continue
            seen_terms.add(term)
            idf = self.idf.get(term)
            if idf is None:
                continue

            for i, ft in self.inv_t.get(term, []):
                denom_t = ft + self._norm_title(i)
                contrib_t = self.w_title * (ft * (self.k1 + 1.0)) / (denom_t + 1e-12)
                acc[i] += idf * contrib_t

            for i, fb in self.inv_b.get(term, []):
                denom_b = fb + self._norm_body(i)
                contrib_b = self.w_body * (fb * (self.k1 + 1.0)) / (denom_b + 1e-12)
                acc[i] += idf * contrib_b

        out = [0.0] * self.N
        for i, v in acc.items():
            out[i] = float(v)
        return out

# ----------------- BM25F heuristics -----------------
INCLUDE_RE = re.compile(
    r"(item\s+1a\b|risk\s+factors?|item\s+7\b|md&a|management[’']s\s+discussion|"
    r"consolidated\s+(statements?|balance|income|cash\s*flows)|"
    r"net\s+sales|revenue|disaggregated|sources?\s+of\s+revenue|segment\s+information|"
    r"geographic\s+data|by\s+product|by\s+region|by\s+category|outlook|strategy|trends?)",
    re.I
)
EXCLUDE_RE = re.compile(
    r"(report\s+of\s+independent|internal\s+control|certification|exhibit|"
    r"signatures?\b|cover\s+page|documents\s+incorporated)", re.I
)
REV_BY_PRODUCT_HINT = re.compile(r"(sources?\s+of\s+revenue|by\s+product|product\s+categor(?:y|ies))", re.I)

def intent_prior_bonus(m: Dict[str, Any], query: str) -> float:
    head = " ".join(str(m.get(k, "")) for k in ("heading", "title", "section")).lower()
    if REV_BY_PRODUCT_HINT.search(query):
        if ("note 2" in head) or (" revenue" in head):
            return 0.25
        if ("segment information" in head) or ("geographic" in head):
            return -0.10
    return 0.0

def extract_phrases(q: str) -> List[str]:
    return [m.group(1).strip().lower() for m in re.finditer(r'"([^"]+)"', q)]

def phrase_bonus(text: str, phrases: List[str], per_hit: float = 0.15, cap: float = 0.45) -> float:
    t = (text or "").lower()
    b = 0.0
    for p in phrases:
        if p and p in t:
            b += per_hit
    return min(cap, b)

def year_proximity_bonus(m: Dict[str, Any], target_year: Optional[int]) -> float:
    try:
        if not target_year:
            return 0.0
        d = str(m.get("doc_date", ""))[:4]
        year = int(d) if len(d) == 4 and d.isdigit() else int(m.get("fy"))
        gap = abs(year - int(target_year))
        return 0.20 if gap == 0 else (0.10 if gap == 1 else 0.0)
    except Exception:
        return 0.0

def soft_section_boost(m: Dict[str, Any], body_text: str) -> float:
    head = " ".join(str(m.get(k, "")) for k in ("heading", "title", "section")).lower()
    bonus = 0.0
    if "segment information" in head or re.search(r"\bnote\s+1?\d\b", head):
        bonus += 0.25
    if "item 7" in head or "md&a" in head:
        bonus += 0.15
    if "net sales" in head or "revenue" in head:
        bonus += 0.15
    t = (body_text or "").lower()
    if re.search(r"\bnet\s+sales\b", t):
        bonus += 0.10
    if sum(bool(re.search(p, t)) for p in [r"\biphone\b", r"\bmac\b", r"\bipad\b", r"\bwearables\b", r"\bservices\b"]) >= 2:
        bonus += 0.10
    return min(0.60, bonus)

def expand_query(q: str, ticker: Optional[str]) -> str:
    ql = q.lower(); extra: List[str] = []
    if ("revenue" in ql) or ("sources of revenue" in ql) or ("source of revenue" in ql):
        extra += ["net sales", "by product", "by region", "segment information", "geographic data"]
    if any(k in ql for k in ["q1","q2","q3","q4","quarter","three months ended"]):
        extra += ["three months ended", "unaudited","condensed consolidated","statements of income","statements of operations"]
    if "net income" in ql:
        extra += ["earnings per share","statements of income","consolidated statements of income"]
    if any(k in ql for k in ["risk","risk factors","supply chain","disruption","disruptions"]):
        extra += ["item 1a","risk factors"]
    if ticker and ticker.upper() == "AAPL":
        extra += ["iphone","mac","ipad","wearables","services"]
    return q + " " + " ".join(extra)

# ----------------- Dense helpers -----------------
def to_score(metric_type: int, dist: float) -> float:
    if metric_type == faiss.METRIC_INNER_PRODUCT:
        return float(dist)
    return 1.0 / (1.0 + float(dist))

def try_get_labels_from_idmap(index):
    try:
        arr = faiss.vector_to_array(getattr(index, "id_map"))
        return arr.astype(np.int64)
    except Exception:
        return None

# ----------------- normalization -----------------
def minmax_normalize(xs: Dict[str, float]) -> Dict[str, float]:
    if not xs:
        return {}
    vals = np.array(list(xs.values()), dtype=float)
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if math.isclose(vmin, vmax):
        return {k: 1.0 for k in xs}  # 单点集直接给 1.0
    return {k: (v - vmin) / (vmax - vmin) for k, v in xs.items()}

def zscore_normalize(xs: Dict[str, float]) -> Dict[str, float]:
    if not xs:
        return {}
    vals = np.array(list(xs.values()), dtype=float)
    mu, sigma = float(np.mean(vals)), float(np.std(vals)) + 1e-9
    zs = {k: (v - mu) / sigma for k, v in xs.items()}
    # 将 z 转到 0-1（稳定融合）
    vvals = np.array(list(zs.values()))
    vmin, vmax = float(np.min(vvals)), float(np.max(vvals))
    if math.isclose(vmin, vmax):
        return {k: 1.0 for k in zs}
    return {k: (v - vmin) / (vmax - vmin) for k, v in zs.items()}

# ----------------- config -----------------
@dataclass
class HybridConfig:
    index_dir: str = "data/index"
    meta: Optional[str] = None
    faiss_path: Optional[str] = None
    content_path: Optional[str] = None
    content_dir: Optional[str] = None
    model: str = "BAAI/bge-base-en-v1.5"
    device: str = "cuda"
    normalize_query: bool = False
    # tokenization
    min_token_len: int = 2
    # BM25F params
    k1: float = 1.5
    b_title: float = 0.20
    b_body: float = 0.75
    w_title: float = 0.5
    w_body: float = 1.0
    # fusion
    w_dense: float = 0.6
    w_bm25: float = 0.4
    norm: str = "minmax"   # or "zscore"
    dense_k_search: int = 2000
    dense_overfetch: int = 100
    # 防错位
    strict_id: bool = True
    audit_only: bool = False

# ----------------- Hybrid Retriever -----------------
class HybridRetriever:
    def __init__(self, cfg: HybridConfig):
        self.cfg = cfg
        meta_path = Path(cfg.meta) if cfg.meta else Path(cfg.index_dir) / "meta.jsonl"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.jsonl not found at {meta_path}")
        self.metas = load_metas(meta_path)

        total, bad, mis = audit_metas(self.metas, strict=False)
        if bad or mis:
            print(f"[AUDIT] total={total} bad_format={bad} chunknum_mismatch={mis}", file=sys.stderr)
            if cfg.strict_id:
                print("[AUDIT] strict_id=True -> content fetch enforces full rid matching.", file=sys.stderr)
        if cfg.audit_only:
            return

        # content root
        if cfg.content_path:
            self.content_root = Path(cfg.content_path)
        elif cfg.content_dir:
            self.content_root = Path(cfg.content_dir)
        else:
            self.content_root = None

        # faiss
        fpath = Path(cfg.faiss_path) if cfg.faiss_path else (Path(cfg.index_dir) / "text_index.faiss")
        if fpath.exists():
            self.index = faiss.read_index(str(fpath))
            self.metric_type = getattr(self.index, "metric_type", faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = None
            self.metric_type = faiss.METRIC_INNER_PRODUCT
            print(f"[WARN] FAISS not found at {fpath}, dense branch disabled.", file=sys.stderr)

        # idmap labels (if any)
        self.labels = try_get_labels_from_idmap(self.index) if self.index is not None else None
        self.label2meta = None
        if self.index is not None:
            if self.labels is not None:
                if len(self.labels) != len(self.metas):
                    raise ValueError(f"idmap labels ({len(self.labels)}) != metas ({len(self.metas)})")
                self.label2meta = {int(lbl): self.metas[i] for i, lbl in enumerate(self.labels)}
            else:
                if self.index.ntotal != len(self.metas):
                    raise ValueError(f"index.ntotal={self.index.ntotal} != meta lines={len(self.metas)} (one-to-one required)")

        # encoder
        self.encoder = None
        if self.index is not None:
            try:
                self.encoder = SentenceTransformer(cfg.model, device=cfg.device)
            except Exception:
                self.encoder = SentenceTransformer(cfg.model, device="cpu")

    # ---- filters ----
    def _apply_meta_filters(self, ticker: Optional[str], form: Optional[str], year: Optional[int]) -> List[Dict[str, Any]]:
        tick = ticker.upper() if ticker else None
        frm = form.upper() if form else None
        out: List[Dict[str, Any]] = []
        for m in self.metas:
            rid = canon_rid(m)
            if not rid:
                continue
            if tick and str(m.get("ticker", "")).upper() != tick:
                continue
            if frm and str(m.get("form", "")).upper() != frm:
                continue
            if year is not None:
                try:
                    if int(m.get("fy")) != int(year):
                        continue
                except Exception:
                    continue
            m["_rid"] = rid
            out.append(m)
        return out

    def _section_prefilter(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        heads = [" ".join(str(m.get(k, "")) for k in ("heading","title","section")) for m in candidates]
        pre = [m for m, h in zip(candidates, heads) if INCLUDE_RE.search(h) and not EXCLUDE_RE.search(h)]
        if pre and (len(pre) >= max(200, int(0.2 * len(candidates)))):
            return pre
        return candidates

    # ---- dense branch ----
    def _dense_search(self, query: str, ticker: Optional[str], form: Optional[str], year: Optional[int], topk: int) -> Dict[str, float]:
        if self.index is None or self.encoder is None:
            return {}
        qvec = self.encoder.encode([query], normalize_embeddings=self.cfg.normalize_query, show_progress_bar=False).astype("float32")
        k_search = max(self.cfg.dense_overfetch * topk, self.cfg.dense_k_search)
        D, I = self.index.search(qvec, k_search)

        tick = ticker.upper() if ticker else None
        frm  = form.upper() if form else None
        dense_scores: Dict[str, float] = {}
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            if self.label2meta is not None:
                m = self.label2meta.get(int(idx))
                if m is None:
                    continue
            else:
                if not (0 <= idx < len(self.metas)):
                    continue
                m = self.metas[int(idx)]

            rid = canon_rid(m)
            if not rid:
                continue
            if tick and str(m.get("ticker","")).upper() != tick:
                continue
            if frm and str(m.get("form","")).upper() != frm:
                continue
            if year is not None:
                try:
                    if int(m.get("fy")) != int(year):
                        continue
                except Exception:
                    continue
            score = to_score(self.metric_type, float(dist))
            if rid not in dense_scores:
                dense_scores[rid] = float(score)
            if len(dense_scores) >= max(1, topk * 5):  # 先收窄
                break
        return dense_scores

    # ---- bm25f branch ----
    def _bm25f_scores(self, query: str, ticker: Optional[str], form: Optional[str], year: Optional[int]) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, Dict[str, Any]]]:
        cands = self._apply_meta_filters(ticker, form, year)
        if not cands:
            return {}, {}, {}
        cands = self._section_prefilter(cands)

        if not self.content_root:
            raise ValueError("Hybrid needs text: set HybridConfig.content_dir or content_path.")
        need_rids = {m["_rid"] for m in cands if m.get("_rid")}
        id2txt = fetch_contents_strict(self.content_root, need_rids, strict_id=self.cfg.strict_id)

        # build docs
        doc_metas: List[Dict[str, Any]] = []
        docs_tokens: List[Tuple[List[str], List[str]]] = []
        raw_texts: List[str] = []
        titles_raw: List[str] = []

        for m in cands:
            rid = m.get("_rid")
            raw = id2txt.get(rid, "")
            if not raw:
                continue
            title_str = " ".join(str(m.get(k, "")) for k in ("heading","title","section")).strip()
            t_tokens = tokenize(title_str, lower=True, min_len=self.cfg.min_token_len)
            b_tokens = tokenize(raw, lower=True, min_len=self.cfg.min_token_len)
            if not (t_tokens or b_tokens):
                continue
            doc_metas.append(m)
            docs_tokens.append((t_tokens, b_tokens))
            raw_texts.append(raw)
            titles_raw.append(title_str)

        if not docs_tokens:
            return {}, {}, {}

        bm25f = BM25FIndex(
            docs_tokens,
            k1=self.cfg.k1, b_title=self.cfg.b_title, b_body=self.cfg.b_body,
            w_title=self.cfg.w_title, w_body=self.cfg.w_body
        )
        q_expanded = expand_query(query, ticker)
        q_tokens = tokenize(q_expanded, lower=True, min_len=self.cfg.min_token_len)
        phrases = extract_phrases(query)

        base_scores = bm25f.scores(q_tokens)
        bm25_scores: Dict[str, float] = {}
        rid2raw: Dict[str, str] = {}
        rid2meta: Dict[str, Dict[str, Any]] = {}

        for i, base in enumerate(base_scores):
            if base <= 0.0:
                continue
            m = doc_metas[i]
            rid = m.get("_rid")
            body_text = raw_texts[i]
            head = titles_raw[i] if i < len(titles_raw) else ""

            bonus = 0.0
            bonus += soft_section_boost(m, body_text)
            bonus += phrase_bonus(body_text, phrases, per_hit=0.15, cap=0.45)
            bonus += year_proximity_bonus(m, year)
            bonus += intent_prior_bonus(m, query)
            # 简单 narrative 提升（轻量版本）
            if ("item 7" in head.lower()) or ("md&a" in head.lower()):
                bonus += 0.20
            bonus = min(0.80, bonus)

            score = base * (1.0 + bonus)
            bm25_scores[rid] = float(score)
            rid2raw[rid] = body_text
            rid2meta[rid] = m

        return bm25_scores, rid2raw, rid2meta

    # ---- fusion ----
    def _normalize(self, xs: Dict[str, float]) -> Dict[str, float]:
        if self.cfg.norm == "zscore":
            return zscore_normalize(xs)
        return minmax_normalize(xs)

    def search(self, query: str, topk: int,
               ticker: Optional[str], form: Optional[str], year: Optional[int]) -> List[Dict[str, Any]]:
        if self.cfg.audit_only:
            print("[INFO] audit-only mode: no search executed.")
            return []

        # 先跑 BM25F（需要正文）以稳住语义范围
        bm25_scores, rid2raw, rid2meta = self._bm25f_scores(query, ticker, form, year)

        # 再跑 Dense；密集检索不依赖正文，可更大范围 overfetch
        dense_scores = self._dense_search(query, ticker, form, year, topk=topk)

        if not bm25_scores and not dense_scores:
            return []

        # 归一化
        bm25_norm = self._normalize(bm25_scores) if bm25_scores else {}
        dense_norm = self._normalize(dense_scores) if dense_scores else {}

        # 融合
        w_d = float(self.cfg.w_dense)
        w_b = float(self.cfg.w_bm25)
        rids = set(bm25_norm) | set(dense_norm)

        fused: List[Tuple[str, float]] = []
        for rid in rids:
            s_d = dense_norm.get(rid, 0.0)
            s_b = bm25_norm.get(rid, 0.0)
            score = w_d * s_d + w_b * s_b
            fused.append((rid, float(score)))

        # 排序 + 去重（rid 唯一）
        fused.sort(key=lambda x: x[1], reverse=True)
        fused = fused[:max(1, topk)]

        # 补全 snippet 所需正文（对 dense-only 命中也要补）
        if self.content_root:
            need_more = {rid for rid, _ in fused if rid not in rid2raw}
            if need_more:
                extra = fetch_contents_strict(self.content_root, need_more, strict_id=self.cfg.strict_id)
                rid2raw.update(extra)

        q_terms_for_snippet = set(tokenize(query, lower=True, min_len=self.cfg.min_token_len))

        out: List[Dict[str, Any]] = []
        for rank, (rid, s) in enumerate(fused, 1):
            m = rid2meta.get(rid)
            # 如果是 dense-only 命中，meta 需要回填
            if m is None:
                # 在 metas 中线性找（数量大时可建索引；这里依赖 label2meta 已经覆盖大部分场景）
                m = next((mm for mm in self.metas if (mm.get("id") == rid or mm.get("chunk_id") == rid)), None)
                if m is None:
                    # 最差兜底：构造一个最小 meta
                    m = {"ticker": "", "fy": "", "form": "", "heading": "", "title": "", "section": "", "_rid": rid}
            raw = rid2raw.get(rid, "")
            snippet = best_snippet(raw, q_terms_for_snippet, win_chars=500) if raw else (m.get("title") or "")
            out.append({
                "rank": rank,
                "score": float(s),
                "id": rid,
                "chunk": chunk_no_from_id(rid),
                "meta": m,
                "snippet": snippet,
                "source": "hybrid(w_d={:.2f},w_b={:.2f},{})".format(w_d, w_b, self.cfg.norm),
            })
        return out

# ----------------- CLI -----------------
def _cli():
    ap = argparse.ArgumentParser(description="Hybrid (Dense + BM25F) retriever")
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--faiss", dest="faiss_path", default=None)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--content-path", default=None)
    ap.add_argument("--content-dir", default=None)
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--normalize", dest="normalize_query", action="store_true", default=False,
                    help="Normalize query embedding (use if index vectors were normalized + inner-product).")
    ap.add_argument("--q", "--query", dest="query", required=True)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--ticker"); ap.add_argument("--form"); ap.add_argument("--year", type=int)
    # tokenization
    ap.add_argument("--min-token-len", type=int, default=2)
    # BM25F params
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b-title", type=float, default=0.20)
    ap.add_argument("--b-body", type=float, default=0.75)
    ap.add_argument("--w-title", type=float, default=0.5)
    ap.add_argument("--w-body", type=float, default=1.0)
    # fusion
    ap.add_argument("--w-dense", type=float, default=0.6)
    ap.add_argument("--w-bm25", type=float, default=0.4)
    ap.add_argument("--norm", choices=["minmax","zscore"], default="minmax")
    ap.add_argument("--dense-k-search", type=int, default=2000)
    ap.add_argument("--dense-overfetch", type=int, default=100)
    # 防错位
    ap.add_argument("--strict-id", dest="strict_id", action="store_true", default=True)
    ap.add_argument("--no-strict-id", dest="strict_id", action="store_false")
    ap.add_argument("--audit-only", action="store_true")

    args = ap.parse_args()
    cfg = HybridConfig(
        index_dir=args.index_dir, meta=args.meta, faiss_path=args.faiss_path,
        content_path=args.content_path, content_dir=args.content_dir,
        model=args.model, device=args.device, normalize_query=args.normalize_query,
        min_token_len=args.min_token_len,
        k1=args.k1, b_title=args.b_title, b_body=args.b_body,
        w_title=args.w_title, w_body=args.w_body,
        w_dense=args.w_dense, w_bm25=args.w_bm25,
        norm=args.norm, dense_k_search=args.dense_k_search, dense_overfetch=args.dense_overfetch,
        strict_id=args.strict_id, audit_only=args.audit_only
    )

    retr = HybridRetriever(cfg)

    if args.audit_only:
        print("[DONE] audit-only finished.")
        return

    hits = retr.search(args.query, topk=args.topk, ticker=args.ticker, form=args.form, year=args.year)
    if not hits:
        print("[INFO] No hits.")
        return

    print(f"Query: {args.query}\n" + "="*80)
    for r in hits:
        m = r["meta"]
        heading = " ".join(str(m.get(k,"")) for k in ("heading","title","section"))
        print(f"[{r['rank']:02d}] score={r['score']:.6f} | {m.get('ticker')} {m.get('fy')} {m.get('form')} "
              f"| chunk={r['chunk']} | id={r['id']}")
        if heading:
            print(f"     heading: {heading[:160]}")
        print(f"     {r['snippet']}")
        print("-"*80)

if __name__ == "__main__":
    _cli()
