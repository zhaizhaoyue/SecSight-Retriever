# src/rag/retriever/bm25f.py
# -*- coding: utf-8 -*-
"""
BM25F retriever with finance-friendly tokenization, fielded scoring (title/body),
inverted index acceleration, meta filters, soft prior boosts (section/year/phrases),
and better snippets.

CLI (example)
python -m src.rag.retriever.bm25f `
  --q "What is the year-over-year change in Amazon’s advertising revenue in 2023?" `
  --ticker AMZN --form "10-K" --year 2023 `
  --topk 8 `
  --index-dir data/index `
  --content-dir data/chunked

"""
from __future__ import annotations
import argparse, json, math, re, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterable, Tuple, Set
from collections import Counter, defaultdict

# ===================== 防错位工具（NEW） =====================
RID_RE   = re.compile(r'^\d{10}-\d{2}-\d{6}::text::chunk-\d+$')
CHUNK_RE = re.compile(r'::text::chunk-(\d+)$')

def chunk_no_from_id(rid: str) -> str:
    m = CHUNK_RE.search(rid or "")
    return m.group(1) if m else "NA"

def canon_rid(m: Dict[str, Any]) -> Optional[str]:
    """从 meta 里提取唯一键（完整 rid）。优先 id，其次 chunk_id；并做格式校验。"""
    rid = m.get("id") or m.get("chunk_id")
    if not rid:
        return None
    return rid if RID_RE.match(rid) else None

def audit_metas(metas: List[Dict[str, Any]], strict: bool = False) -> Tuple[int, int, int]:
    """
    返回: (total, bad_format, chunknum_mismatch)
    - bad_format: rid 缺失或不符合 RID_RE
    - chunknum_mismatch: 若存在 chunk_index，则与 rid 解析不一致
    在 strict=True 时，发现异常直接 raise。
    """
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

# ----------------- I/O utils -----------------
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
            # 规范化：chunk_id 一律用完整 rid；title 兜底
            rid = flat.get("id") or flat.get("chunk_id")
            flat["chunk_id"] = rid
            flat.setdefault("title", flat.get("title") or "")
            metas.append(flat)
    return metas

def _iter_jsonl_files(base: Path) -> List[Path]:
    if base.is_file():
        return [base]
    # 稳定排序，避免文件系统顺序漂移导致行为差异
    return sorted([p for p in base.rglob("*.jsonl")], key=lambda x: (x.parent.as_posix(), x.name))

def fetch_contents_strict(content_path: Optional[Path], rids: Set[str], strict_id: bool = True) -> Dict[str, str]:
    """
    按【完整 rid】取正文：只接受形如 '0001628280-24-002390::text::chunk-275' 的键。
    严格模式下，遇到短键（仅 'chunk-XXX'）直接忽略，防止跨 accno/FY 串读。
    """
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
                        # 严格模式下跳过短键，避免串档
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
    # 未找齐时不会报错，以免影响主流程；打印提示即可
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
NARR_RE = re.compile(r"\b(item\s+7\b|md&a|item\s*1a\b|risk\s+factors|strategy|outlook|trend)\b", re.I)
FIN_PHRASES = [
    "net sales","total net sales","research and development","operating expenses",
    "foreign currency","other income (expense)","advertising services",
    "disaggregated revenue","segment information","note 2","note 13"
]
def phrase_hint_bonus(text: str, per_hit: float = 0.08, cap: float = 0.32) -> float:
    t = (text or "").lower()
    hits = sum(1 for p in FIN_PHRASES if p in t)
    return min(cap, hits * per_hit)

def narrative_boost(head: str, body: str, intents: Dict[str,bool], ql: str = "") -> float:
    h = (head or "").lower(); b = (body or "").lower()
    boost = 0.0
    # 仅当查询含 risk 相关且正文也出现 risk 时再加
    if ("risk" in ql or "risk factor" in ql or "item 1a" in ql) and ("risk" in b or "risk factor" in b):
        boost += 0.20
    if ("md&a" in h or "item 7" in h) and any(w in ql for w in ["trend","outlook","driver","strategy"]):
        boost += 0.15
    # 对与查询明显不符的“网络安全”给微弱负分，避免把所有风险都顶上来
    if any(k in ql for k in ["supply chain","supplier","shortage","logistics"]) and "cyber" in b:
        boost -= 0.10
    return max(-0.10, min(0.30, boost))


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

# ----------------- BM25F with inverted index -----------------
class BM25FIndex:
    """
    Fielded BM25 (title/body) with per-term postings to avoid full scans.
    docs: list of (title_tokens, body_tokens)
    """
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

# ----------------- retriever heuristics -----------------
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
REV_BY_PRODUCT_HINT = re.compile(
    r"(sources?\s+of\s+revenue|by\s+product|product\s+categor(?:y|ies))",
    re.I
)

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
    bonus += phrase_hint_bonus(body_text)
    return min(0.60, bonus)

# ----------------- config & retriever -----------------
@dataclass
class BM25FConfig:
    index_dir: str = "data/index"
    meta: Optional[str] = None
    content_path: Optional[str] = None
    content_dir: Optional[str] = None
    # tokenization
    min_token_len: int = 2
    # BM25F params
    k1: float = 1.5
    b_title: float = 0.20
    b_body: float = 0.75
    w_title: float = 2.5
    w_body: float = 1.0
    # 防错位
    strict_id: bool = True
    audit_only: bool = False

class BM25FRetriever:
    def __init__(self, cfg: BM25FConfig):
        self.cfg = cfg
        meta_path = Path(cfg.meta) if cfg.meta else Path(cfg.index_dir) / "meta.jsonl"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.jsonl not found at {meta_path}")
        self.metas = load_metas(meta_path)

        # 审计：格式 & chunk_index 一致性
        total, bad, mis = audit_metas(self.metas, strict=False)
        if bad or mis:
            print(f"[AUDIT] total={total} bad_format={bad} chunknum_mismatch={mis}", file=sys.stderr)
            if cfg.strict_id:
                print("[AUDIT] strict_id=True -> proceeding but will enforce rid matching at content stage.", file=sys.stderr)
        if cfg.audit_only:
            # 仅审计模式，直接返回
            return

        if cfg.content_path:
            self.content_root = Path(cfg.content_path)
        elif cfg.content_dir:
            self.content_root = Path(cfg.content_dir)
        else:
            self.content_root = None

    # helpers
    def _apply_meta_filters(self, ticker: Optional[str], form: Optional[str], year: Optional[int]) -> List[Dict[str, Any]]:
        tick = ticker.upper() if ticker else None
        frm = form.upper() if form else None
        yr = year
        out: List[Dict[str, Any]] = []
        for m in self.metas:
            rid = canon_rid(m)
            if not rid:
                # rid 不合规直接跳过（避免短键）
                continue
            if tick and str(m.get("ticker", "")).upper() != tick:
                continue
            if frm and str(m.get("form", "")).upper() != frm:
                continue
            if yr is not None:
                try:
                    if int(m.get("fy")) != int(yr):
                        continue
                except Exception:
                    pass
            m["_rid"] = rid  # 缓存一份
            out.append(m)
        return out

    def _expand_query(self, q: str, ticker: Optional[str]) -> str:
        ql = q.lower(); extra = []

        # —— 常见财报等价词 —— #
        if any(k in ql for k in ["advertising","ads","ad revenue"]):
            extra += ["advertising services","sponsored ads","ad sales","ad revenue growth","marketing services"]
        if any(k in ql for k in ["research and development","r&d","rnd"]):
            extra += ["research & development","research and development expense"]
        if any(k in ql for k in ["operating expenses","opex"]):
            extra += ["selling general and administrative","sga","research and development"]
        if any(k in ql for k in ["data center","datacenter","dc revenue"]):
            extra += ["compute & networking","segment information"]

        # —— YoY/增长类 —— #
        if any(k in ql for k in ["yoy","year-over-year","year over year","compare","vs","versus"]):
            extra += ["compared to 2022","increased","decreased","change","%","percentage","growth"]

        # —— FX 影响 —— #
        if "foreign exchange" in ql or "fx" in ql or "currency" in ql:
            extra += ["foreign currency","fx impact","exchange rates","other income (expense)"]

        # —— 供应链风险 —— #
        if any(k in ql for k in ["supply chain","supplier","disruption","shortage"]):
            extra += ["logistics","raw materials","components","semiconductor","shortages","delays","suppliers"]

        # —— 季度（Q1~Q4）与 10-Q 语言映射 —— #
        if any(k in ql for k in ["q1","q2","q3","q4","quarter"]):
            extra += ["three months ended","(unaudited)","condensed consolidated","statements of operations",
                    "income statements"]

        # —— 产品拆分 —— #
        if "breakdown" in ql or "by product" in ql:
            extra += ["net sales","disaggregated","significant products and services","note 2","segment information"]

        # —— 品牌特定线索（已有 AAPL，可再加） —— #
        if ticker:
            t = ticker.upper()
            if t == "AAPL": extra += ["iphone","mac","ipad","wearables","services","note 2 – revenue"]
            if t == "MSFT": extra += ["office 365","azure","intelligent cloud","productivity and business processes"]
            if t == "AMZN": extra += ["aws","advertising services","sponsored products"]
            if t == "NVDA": extra += ["data center","gaming","segment information"]

        return q + " " + " ".join(extra)


    # public API
    def search(self, query: str, topk: int = 8,
               ticker: Optional[str] = None, form: Optional[str] = None, year: Optional[int] = None) -> List[Dict[str, Any]]:
        if self.cfg.audit_only:
            print("[INFO] audit-only mode: no search executed.")
            return []

        candidates = self._apply_meta_filters(ticker, form, year)
        if not candidates:
            return []

        heads = [" ".join(str(m.get(k, "")) for k in ("heading","title","section")) for m in candidates]
        pre = [m for m, h in zip(candidates, heads) if INCLUDE_RE.search(h) and not EXCLUDE_RE.search(h)]
        if pre and (len(pre) >= max(200, int(0.2 * len(candidates)))):
            candidates = pre

        if not self.content_root:
            raise ValueError("BM25F needs text: set BM25FConfig.content_dir or content_path.")

        # —— 关键：严格用完整 rid 拉正文 —— #
        need_rids = {m["_rid"] for m in candidates if m.get("_rid")}
        id2txt = fetch_contents_strict(self.content_root, need_rids, strict_id=self.cfg.strict_id)

        # build docs
        doc_metas: List[Dict[str, Any]] = []
        docs_tokens: List[Tuple[List[str], List[str]]] = []
        raw_texts: List[str] = []
        titles_raw: List[str] = []

        for m in candidates:
            rid = m.get("_rid")  # 已经过 canon_rid 校验
            raw = id2txt.get(rid, "")
            if not raw:
                continue
            title_str = " ".join(str(m.get(k, "")) for k in ("heading", "title", "section")).strip()
            t_tokens = tokenize(title_str, lower=True, min_len=self.cfg.min_token_len)
            b_tokens = tokenize(raw, lower=True, min_len=self.cfg.min_token_len)
            if not (t_tokens or b_tokens):
                continue
            doc_metas.append(m)
            docs_tokens.append((t_tokens, b_tokens))
            raw_texts.append(raw)
            titles_raw.append(title_str)

        if not docs_tokens:
            return []

        bm25f = BM25FIndex(
            docs_tokens,
            k1=self.cfg.k1,
            b_title=self.cfg.b_title,
            b_body=self.cfg.b_body,
            w_title=self.cfg.w_title,
            w_body=self.cfg.w_body,
        )

        q_expanded = self._expand_query(query, ticker)
        q_tokens = tokenize(q_expanded, lower=True, min_len=self.cfg.min_token_len)
        phrases = extract_phrases(query)

        base_scores = bm25f.scores(q_tokens)

        results: List[Tuple[int, float]] = []
        q_terms_for_snippet = set(q_tokens)
        for i, base in enumerate(base_scores):
            if base <= 0.0:
                continue
            m = doc_metas[i]
            body_text = raw_texts[i]
            head = titles_raw[i] if i < len(titles_raw) else ""

            bonus = 0.0
            bonus += soft_section_boost(m, body_text)
            bonus += phrase_bonus(body_text, phrases, per_hit=0.15, cap=0.45)
            bonus += year_proximity_bonus(m, year)
            bonus += intent_prior_bonus(m, query)
            bonus += narrative_boost(head, body_text, {})

            bonus = min(0.80, bonus)
            score = base * (1.0 + bonus)
            results.append((i, score))

        if not results:
            results = [(i, s) for i, s in enumerate(base_scores) if s > 0.0]
            if not results:
                return []

        results.sort(key=lambda x: x[1], reverse=True)
        unique = {}
        for i, s in results:
            rid = doc_metas[i].get("_rid")
            if rid not in unique:
                unique[rid] = (i, s)
        results = list(unique.values())[:max(1, topk)]

        out: List[Dict[str, Any]] = []
        for rank, (i, s) in enumerate(results, 1):
            m = doc_metas[i]
            rid = m.get("_rid")
            raw = raw_texts[i]
            snippet = best_snippet(raw, q_terms_for_snippet, win_chars=500)
            out.append({
                "rank": rank,
                "score": float(s),
                "id": rid,                               # 标准字段：完整 rid
                "chunk": chunk_no_from_id(rid),          # 展示用 chunk 号（从 rid 解析）
                "meta": m,
                "snippet": snippet,
                "source": "bm25f",
            })
        return out

# ----------------- CLI -----------------
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--content-path", default=None)
    ap.add_argument("--content-dir", default=None)
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
    # 防错位参数
    ap.add_argument("--strict-id", dest="strict_id", action="store_true", default=True,
                    help="Only match full rid (accno::text::chunk-N) when fetching content.")
    ap.add_argument("--no-strict-id", dest="strict_id", action="store_false")
    ap.add_argument("--audit-only", action="store_true",
                    help="Audit meta.jsonl for rid/chunk_index consistency and exit.")

    args = ap.parse_args()

    cfg = BM25FConfig(
        index_dir=args.index_dir, meta=args.meta,
        content_path=args.content_path, content_dir=args.content_dir,
        min_token_len=args.min_token_len,
        k1=args.k1, b_title=args.b_title, b_body=args.b_body,
        w_title=args.w_title, w_body=args.w_body,
        strict_id=args.strict_id, audit_only=args.audit_only
    )

    retr = BM25FRetriever(cfg)

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
