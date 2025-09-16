# -*- coding: utf-8 -*-
"""
RAG Answerer — plugs HybridRetrieverRRF into an LLM (OpenAI-compatible or OpenAI).
Usage example:

python -m src.rag.retriever.answer_api `
  --query "What were Pfizer’s total revenues in its 2023 10-K?" `
  --ticker PFE --form 10-K --year 2023 `
  --index-dir data/index --content-dir data/chunked `
  --model BAAI/bge-base-en-v1.5 `
  --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --topk 8 --bm25-topk 200 --dense-topk 200 --ce-candidates 256 `
  --w-bm25 1.0 --w-dense 2.0 --ce-weight 0.4 `
  --llm-base-url https://api.deepseek.com/v1 `
  --llm-model deepseek-chat `
  --llm-api-key "sk-f6220301c405405a8ca5c65a06a75f7b" `
  --json-out

"""
from __future__ import annotations
import os, json, re, argparse, math, textwrap
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set

# ---- Optional token estimator (fallback to naive if tiktoken not installed)
def _estimate_tokens(s: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(s))
    except Exception:
        # crude fallback: ~4 chars per token
        return max(1, len(s) // 4)

# ---- Import hybrid retriever (your file)
from src.rag.retriever.hybrid import HybridRetrieverRRF, CrossEncoderReranker
from src.rag.retriever.bm25_text import BM25TextRetriever, BM25TextConfig
from src.rag.retriever.dense import DenseRetriever

@dataclass
class Evidence:
    rid: str
    text: str
    score: float
    meta: Dict[str, Any]

# ---------------- Prompt templates ----------------
SYS_PROMPT = """
You are a precise analyst answering questions strictly from the provided evidence blocks.
- Use ONLY the evidence. Ignore irrelevant blocks completely.
- If at least one block is relevant, you MUST answer and cite it; only say "insufficient" if none is relevant.
- Every factual claim needs ≥1 citation using the [RID] identifiers.
- Output valid JSON only, no extra text:

{
  "answer": "concise markdown with inline refs like [^1]",
  "citations": [{"rid":"<RID>","quote":"<short supporting quote>"}]
}



"""

USER_TMPL = """
Question:
{question}

Evidence blocks (each prefixed with [RID]):
{evidence}

Instructions:
1) Identify which blocks directly address the question; ignore the rest.
2) Write a concise factual answer in markdown and add inline [^n] markers.
3) Fill "citations" with used RIDs and short verbatim quotes (1–2 sentences).
4) If no blocks are relevant, return: "The evidence provided does not contain information to answer this question."
5) Use at most 3–7 citations unless necessary.
6) Also treat passages describing constraints/conditions tied to the risk as part of the risk context
(e.g., practical solutions that must not undermine affordability or reliability).
If evidence mentions affordability/reliability, net-zero, or NDCs, include them if relevant.
7) When amounts/dates appear, quote the exact figure and unit; prefer the most recent within the evidence.
8) When multiple periods are present, state the period explicitly (YYYY-MM-DD or FY/FQ) and cite both sides.
9) If evidence conflicts, state the discrepancy in one sentence and cite both sources.




"""

import re, hashlib
from typing import List, Dict, Any, Set, Tuple, Optional

# ==========================
# Token estimation utilities
# ==========================
try:
    # Optional, more accurate for OpenAI-like encodings
    import tiktoken  # type: ignore
    _ENC = tiktoken.get_encoding("cl100k_base")
    def _estimate_tokens(s: str) -> int:
        return max(1, len(_ENC.encode(s)))
except Exception:
    def _estimate_tokens(s: str) -> int:
        # Mixed-language heuristic; tighter than len//4 for CJK
        if not s:
            return 1
        # Roughly assume: English ~4 chars/token; CJK ~1 char/token
        cjk = sum(1 for ch in s if '\u4e00' <= ch <= '\u9fff')
        non = len(s) - cjk
        return max(1, cjk + non // 4)

# ==========================
# Text splitting utilities
# ==========================
# Expanded sentence/phrase boundaries (EN + ZH + punctuation variants)
SENT_RE = re.compile(
    r"(?<=[\.|\?|\!|。|！|？|；|;|：|:])\s+|"   # end punctuation + whitespace
    r"(?<=\u2026)\s+|"                          # … ellipsis
    r"(?<=[—–\-])\s+"                           # dashes + whitespace
)

BULLET_RE = re.compile(
    r"^\s*(?:[•\-–—]|\(\s*(?:[ivxlcdm]+|\d+|[a-zA-Z])\s*\)|(?:\d+\.|[a-zA-Z]\.))\s+"
)

STOPWORDS: Set[str] = {
    # Minimal English stop set; extend as needed
    "the","a","an","and","or","of","to","in","for","on","by","with","as","at","from",
    "this","that","these","those","is","are","was","were","be","been","being","it","its",
}

_DEF_MIN_FIRST_SENT_TOKENS = 18


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _sent_split(s: str) -> List[str]:
    out: List[str] = []
    for line in (s or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if BULLET_RE.match(line):
            out.append(line)
        else:
            parts = [p.strip() for p in SENT_RE.split(line) if p.strip()]
            out.extend(parts)
    return out


def _wordset(s: str) -> Set[str]:
    ws = set(w for w in re.findall(r"[\w\u4e00-\u9fff]+", _norm(s)) if w and w not in STOPWORDS)
    return ws


# ==========================
# Exported API 1: pretty block builder
# ==========================

def build_evidence_block(evs: List["Evidence"], max_chars_per_block: int = 1200, mode="max_coverage") -> str:
    if mode == "max_coverage":
        max_chars_per_block = 10**9 

    """
    改进：
    - 截断优先在句/要点边界（与 pack_evidence 共用 _sent_split）
    - 保证首句尽量完整且不少于 _DEF_MIN_FIRST_SENT_TOKENS token（若可能）
    - 仍保持原有输出格式
    """
    lines: List[str] = []
    for e in evs:
        txt = (getattr(e, "text", None) or "").strip()
        if not txt:
            lines.append(f"[{e.rid}]\n\n")
            continue

        if len(txt) > max_chars_per_block:
            parts = _sent_split(txt)
            buf: List[str] = []
            char_budget = max_chars_per_block
            # Ensure first sentence presence
            if parts:
                first = parts[0]
                ftxt = first
                if len(ftxt) > char_budget:
                    # If the first sentence alone exceeds budget, hard-cut but try to keep minimal tokens
                    # Prefer token-aware cut
                    acc = []
                    for ch in ftxt:
                        acc.append(ch)
                        if len("".join(acc)) >= char_budget:
                            break
                    ftxt = "".join(acc).rstrip() + " …"
                    buf.append(ftxt)
                    txt = " ".join(buf)
                else:
                    buf.append(first)
                    used = len(first)
                    # Try to add more sentences until budget full
                    for p in parts[1:]:
                        if used + 1 + len(p) <= char_budget:
                            buf.append(p)
                            used += 1 + len(p)
                        else:
                            break
                    # If first sentence too short (< tokens), try append more fragments
                    if _estimate_tokens(" ".join(buf)) < _DEF_MIN_FIRST_SENT_TOKENS:
                        for p in parts[len(buf):]:
                            if used + 1 + len(p) <= char_budget:
                                buf.append(p)
                                used += 1 + len(p)
                            else:
                                break
                    # trailing ellipsis if truncated
                    if len(" ".join(parts)) > used:
                        buf[-1] = buf[-1].rstrip() + " …"
                    txt = " ".join(buf)
            else:
                # No sentence split => raw cut
                txt = txt[:max_chars_per_block].rstrip() + " …"

        lines.append(f"[{e.rid}]\n{txt}\n")
    return "\n".join(lines)


# ==========================
# Exported API 2: evidence packer with MMR & coverage
# ==========================

class Evidence:
    def __init__(self, rid: str, text: str, score: float, meta: Optional[Dict[str, Any]] = None):
        self.rid = rid
        self.text = text
        self.score = score
        self.meta = meta or {}


def pack_evidence(records: List[Dict[str, Any]],
                  max_tokens: int = 5000,
                  mode: str = "max_coverage",
                  k_per_heading: int = 3) -> List[Evidence]:
    """
    泛化增强版：
    1) 轻去重：签名加入 (ticker, form, fy, heading) 盐 + per-sig 上限
    2) 轻覆盖：对不同 heading 做保底，空 heading 限额更低
    3) 软对齐：对锚 (ticker/form/fy) 的偏离使用指数衰减，参数可调
    4) MMR：失败不提前 break；有句级回退；缓存分词/词集
    5) 统一句级截断工具，与 build_evidence_block 一致
    6) 更稳的 token 估计兜底
    """
    if not records:
        return []

    # ---------- helpers ----------
    def _sig_of(r: Dict[str, Any], n_tokens: int = 80) -> str:
        meta = (r.get("meta") or {})
        salt = f"{meta.get('ticker','')}|{meta.get('form','')}|{meta.get('fy','')}|{meta.get('heading','')}"
        text = (r.get("content") or r.get("snippet") or "").strip()
        toks = _norm(text).split()[:n_tokens]
        return hashlib.sha1((salt + "|" + " ".join(toks)).encode("utf-8")).hexdigest()

    def _boost_score(r: Dict[str, Any]) -> float:
        s = float(r.get("final_score", 0.0))
        m = r.get("meta", {}) or {}
        tag = (m.get("section_tag") or "").lower()
        # Section priors
        if tag == "risk_factors":
            s *= 1.20
        elif tag == "fwd_statements":
            s *= 0.92
        elif "md&a" in tag:
            s *= 0.85
        # Cross-year mention prior
        if m.get("mentions_other_years"):
            s *= 0.90
        return s

    def _tok_est(s: str) -> int:
        return _estimate_tokens(s)

    def _mmr_value(rel: float, txt: str, selected_sets: List[Set[str]], lam: float = 0.65) -> float:
        ws = _wordset(txt)
        if not selected_sets:
            return lam * rel
        # max Jaccard to selected
        max_sim = 0.0
        for ss in selected_sets:
            if not ss:
                continue
            inter = len(ws & ss)
            union = len(ws | ss)
            if union == 0:
                continue
            j = inter / union
            if j > max_sim:
                max_sim = j
        return lam * rel - (1 - lam) * max_sim

    # ---------- 1) initial scoring & sort ----------
    recs: List[Dict[str, Any]] = []
    for r in records:
        txt = (r.get("content") or r.get("snippet") or "").strip()
        if not txt:
            continue
        rr = dict(r)
        rr["_boosted"] = _boost_score(r)
        recs.append(rr)
    if not recs:
        return []

    recs.sort(key=lambda x: x["_boosted"], reverse=True)

    # ---------- 2) soft alignment to anchors ----------
    head = recs[: min(20, len(recs))]
    def _mode(values):
        from collections import Counter
        cnt = Counter([str(v).lower() for v in values if v is not None and str(v) != "None"])
        return (cnt.most_common(1)[0][0] if cnt else None)

    anchor_ticker = _mode([x.get("meta", {}).get("ticker") for x in head])
    anchor_form   = _mode([x.get("meta", {}).get("form") for x in head])
    anchor_fy     = _mode([x.get("meta", {}).get("fy") for x in head])

    # Exponential decay parameters (configurable)
    if mode == "max_coverage":
        YEAR_LAMBDA = 0.2  # 原来 0.6
        TICKER_PENALTY = 0.8  # 原来 0.5
        FORM_PENALTY   = 0.9  # 原来 0.8
    else:
        YEAR_LAMBDA = 0.6
        TICKER_PENALTY = 0.5
        FORM_PENALTY   = 0.8
    for r in recs:
        m = r.get("meta", {}) or {}
        tk = str(m.get("ticker", "")).lower()
        fm = str(m.get("form", "")).lower()
        fy = m.get("fy")
        # ticker/form mismatches — multiplicative light penalty
        if anchor_ticker and tk and tk != anchor_ticker: r["_boosted"] *= TICKER_PENALTY
        if anchor_form and fm and fm != anchor_form:     r["_boosted"] *= FORM_PENALTY
        # year gap — exponential decay
        if anchor_fy and fy is not None:
            try:
                gap = abs(int(fy) - int(anchor_fy))
                if gap > 0:
                    r["_boosted"] *= (0.9 * (2.71828 ** (-YEAR_LAMBDA * max(0, gap - 1))))
            except Exception:
                pass

    recs.sort(key=lambda x: x["_boosted"], reverse=True)

    # ---------- 3) light dedup by signature ----------
    if mode == "max_coverage":
        PER_SIG_CAP = 10  # 原来2
    else:
        PER_SIG_CAP = 2
    sig_count: Dict[str, int] = {}
    uniq: List[Dict[str, Any]] = []
    for r in recs:
        sg = _sig_of(r)
        if sig_count.get(sg, 0) >= PER_SIG_CAP:
            continue
        sig_count[sg] = sig_count.get(sg, 0) + 1
        uniq.append(r)
    if not uniq:
        return []

    # ---------- 4) heading safeguard (~30% budget) ----------
    heading_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in uniq:
        h = (r.get("meta", {}) or {}).get("heading") or ""
        heading_groups.setdefault(h, []).append(r)

    budget_tokens = max_tokens
    safeguard_tokens = int(budget_tokens * 0.30)
    selected: List[Dict[str, Any]] = []
    selected_sets: List[Set[str]] = []
    cur_tokens = 0

    # Sort groups by best boosted score inside
    groups_sorted = sorted(heading_groups.items(), key=lambda kv: max(x["_boosted"] for x in kv[1]), reverse=True)

    for h, items in groups_sorted:
        items.sort(key=lambda x: x["_boosted"], reverse=True)
        txt0 = (items[0].get("content") or items[0].get("snippet") or "").strip()
        tks0 = _tok_est(txt0)
        # Empty/None heading gets a smaller cap (avoid hogging)
        if not h and (cur_tokens + tks0 > safeguard_tokens // 3):
            continue
        if cur_tokens + tks0 <= safeguard_tokens:
            selected.append(items[0])
            selected_sets.append(_wordset(txt0))
            cur_tokens += tks0
     # ---------- 4.5) 按 heading 配额扩张 + 相邻 chunk 缝合 ----------
    def _rid_chunk_id(rid: str) -> Optional[int]:
        m = re.search(r"::text::chunk-(\d+)$", rid or "")
        return int(m.group(1)) if m else None

    def _stitch_adjacent(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 要求同一 filing、同一 heading、chunk 连号
        if not items: return items
        # 先按 rid chunk id 排序
        items = sorted(items, key=lambda x: _rid_chunk_id(x.get("id","")) or 10**9)
        stitched: List[Dict[str, Any]] = []
        buf = []
        prev = None
        def _flush():
            nonlocal buf
            if not buf: return
            if len(buf) == 1:
                stitched.append(buf[0])
            else:
                # 合并文本
                txt = "\n\n".join((it.get("content") or it.get("snippet") or "").strip() for it in buf)
                merged = dict(buf[0]); merged["content"] = txt
                stitched.append(merged)
            buf = []

        for it in items:
            if not prev:
                buf = [it]; prev = it; continue
            # 同 heading + 同 filing 前缀 + chunk 连号才合并
            h1 = (prev.get("meta") or {}).get("heading")
            h2 = (it.get("meta") or {}).get("heading")
            r1, r2 = prev.get("id",""), it.get("id","")
            c1, c2 = _rid_chunk_id(r1), _rid_chunk_id(r2)
            same_head = (h1 == h2)
            same_file = r1.split("::text::")[0] == r2.split("::text::")[0]
            if same_head and same_file and c1 is not None and c2 is not None and c2 == c1 + 1:
                buf.append(it); prev = it
            else:
                _flush(); buf = [it]; prev = it
        _flush()
        return stitched

    # 在 heading_groups 构建后、MMR 前：
    if mode == "max_coverage":
        expanded: List[Dict[str, Any]] = []
        for h, items in heading_groups.items():
            items.sort(key=lambda x: x["_boosted"], reverse=True)
            # 取前 k_per_heading
            pick = items[:k_per_heading]
            # 缝合相邻
            pick = _stitch_adjacent(pick)
            expanded.extend(pick)
        # 与已选的 safeguard 去重合并（保留顺序）
        seen = set(id(x) for x in selected)
        expanded = [x for x in expanded if id(x) not in seen]
        selected.extend(expanded)
        # 跳过 MMR：remaining = []
        remaining = []

    # ---------- 5) MMR fill for remaining budget ----------
    remaining = [r for r in uniq if r not in selected]
    mmr_lambda = 0.65

    def _would_fit(txt: str) -> bool:
        return (cur_tokens + _tok_est(txt)) <= budget_tokens

    tried: Set[str] = set()
    while remaining and cur_tokens < budget_tokens:
        best_idx = -1
        best_val = -1e18
        for i, r in enumerate(remaining):
            rid = r.get("id")
            if rid in tried:
                continue
            txt = (r.get("content") or r.get("snippet") or "").strip()
            v = _mmr_value(r.get("_boosted", 0.0), txt, selected_sets, lam=mmr_lambda)
            if v > best_val:
                best_val = v
                best_idx = i
        if best_idx < 0:
            break

        r = remaining[best_idx]
        rid = r.get("id")
        tried.add(rid)
        txt = (r.get("content") or r.get("snippet") or "").strip()

        if _would_fit(txt):
            selected.append(r)
            selected_sets.append(_wordset(txt))
            cur_tokens += _tok_est(txt)
            remaining.pop(best_idx)
            continue

        # Sentence/point-level fallback within remaining budget
        remain = budget_tokens - cur_tokens
        if remain <= 50:
            # Not enough room to add meaningful content — stop
            break
        parts = _sent_split(txt)
        buf: List[str] = []
        bt = 0
        for p in parts:
            pt = _tok_est(p + " ")
            if bt + pt > remain:
                break
            buf.append(p)
            bt += pt
        if buf:
            trunc_txt = " ".join(buf).rstrip() + " …"
            trunc = dict(r)
            trunc["content"] = trunc_txt
            selected.append(trunc)
            selected_sets.append(_wordset(trunc_txt))
            cur_tokens += bt
        # Do not break here — continue searching for other candidates that may fit better
        remaining.pop(best_idx)

    # ---------- 6) build outputs ----------
    packed: List[Evidence] = []
    for r in selected:
        rid = r["id"]
        txt = (r.get("content") or r.get("snippet") or "").strip()
        meta = r.get("meta", {}) or {}
        packed.append(Evidence(rid=rid, text=txt, score=r.get("_boosted", r.get("final_score", 0.0)), meta=meta))
    return packed



# ---------------- LLM client (OpenAI-compatible) ----------------
class LLMClient:
    """
    Works with:
      - Official OpenAI API (base_url=https://api.openai.com/v1)
      - Local OpenAI-compatible servers (vLLM/Ollama) e.g. http://localhost:8000/v1
    """
    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.timeout = timeout
        # Prefer openai python SDK if available & base_url is OpenAI; else fallback to requests
        self._use_sdk = False
        try:
            from openai import OpenAI  # type: ignore
            self._use_sdk = True
            self._sdk_client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        except Exception:
            self._use_sdk = False
        # If base_url looks custom, we still can try SDK (it supports base_url), else requests.
        if not self._use_sdk:
            import requests  # ensure import

    def chat_json(self, system: str, user: str, temperature: float = 0.25) -> Dict[str, Any]:
        # Ask model to return JSON via response_format
        if self._use_sdk:
            from openai import APIConnectionError, RateLimitError  # type: ignore
            resp = self._sdk_client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[{"role":"system","content":system},
                          {"role":"user","content":user}],
            )
            content = resp.choices[0].message.content
            try:
                return json.loads(content)
            except Exception:
                # try to extract JSON blob
                m = re.search(r"\{.*\}", content, re.S)
                return json.loads(m.group(0)) if m else {"answer":"", "citations":[]}
        else:
            import requests
            url = f"{self.base_url}/chat/completions"
            headers = {"Content-Type":"application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {
                "model": self.model,
                "temperature": temperature,
                "response_format": {"type":"json_object"},
                "messages": [
                    {"role":"system","content":system},
                    {"role":"user","content":user},
                ],
            }
            r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            try:
                return json.loads(content)
            except Exception:
                m = re.search(r"\{.*\}", content, re.S)
                return json.loads(m.group(0)) if m else {"answer":"", "citations":[]}

# ---------------- Main answer function ----------------
def answer_with_llm(query: str,
                    hybrid_records: List[Dict[str, Any]],
                    llm: LLMClient,
                    max_ctx_tokens: int = 2400) -> Dict[str, Any]:
    if not hybrid_records:
        return {"answer": "I couldn't find relevant evidence in the indexed filings.", "citations": []}

    packed = pack_evidence(hybrid_records, max_tokens=max_ctx_tokens)
    if not packed:
        return {"answer": "I couldn't pack any usable evidence (token limit).", "citations": []}

    evidence_block = build_evidence_block(packed)
    # print("=== Evidence block sent to LLM ===")
    # print(evidence_block)
    # print("==================================")
    user_prompt = USER_TMPL.format(question=query, evidence=evidence_block)

    out = llm.chat_json(SYS_PROMPT, user_prompt, temperature=0.2)
    # guardrail: keep only citations whose rid exists in packed
    valid_rids = {e.rid for e in packed}
    cits = []
    for c in (out.get("citations") or []):
        rid = (c or {}).get("rid")
        quote = (c or {}).get("quote", "")
        if rid in valid_rids and quote:
            # shrink quote
            quote = quote.strip()
            if len(quote) > 240: quote = quote[:240] + " ..."
            cits.append({"rid": rid, "quote": quote})
    out["citations"] = cits
    # soft requirement: add at least one citation if possible
    if not out.get("answer"):
        out["answer"] = "I couldn't synthesize an answer from the provided evidence."
    return out

# ---------------- CLI: bridge Hybrid -> LLM ----------------
def main():
    ap = argparse.ArgumentParser(description="RAG Answerer: HybridRetrieverRRF + LLM (OpenAI-compatible).")
    # query/task
    ap.add_argument("--query", required=True)

    # hybrid configs (mirror your hybrid CLI)
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--k", type=float, default=60.0)
    ap.add_argument("--content-path", default=None)
    ap.add_argument("--content-dir", default=None)
    ap.add_argument("--show-chars", type=int, default=800)
    ap.add_argument("--w-bm25", type=float, default=1.0)
    ap.add_argument("--w-dense", type=float, default=2.0)
    ap.add_argument("--ce-weight", type=float, default=0.5)
    ap.add_argument("--bm25-topk", type=int, default=200)
    ap.add_argument("--dense-topk", type=int, default=200)
    ap.add_argument("--ce-candidates", type=int, default=256)
    ap.add_argument("--topk", type=int, default=8)

    # filters
    ap.add_argument("--ticker")
    ap.add_argument("--form")
    ap.add_argument("--year", type=int)

    # LLM configs
    ap.add_argument("--llm-base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "gpt-4o-mini"))
    ap.add_argument("--llm-api-key", default=os.getenv("OPENAI_API_KEY", ""))
    ap.add_argument("--max-context-tokens", type=int, default=2400)
    ap.add_argument("--json-out", action="store_true", help="Print JSON only (no pretty).")

    args = ap.parse_args()

    # init hybrid components (same as your hybrid CLI)
    bm25_cfg = BM25TextConfig(index_dir=args.index_dir)
    dense = DenseRetriever(index_dir=args.index_dir, model=args.model, device=args.device)

    reranker = None
    if (args.ce_weight is None) or (args.ce_weight > 1e-6 and args.ce_candidates > 0):
        reranker = CrossEncoderReranker(model_name=args.rerank_model, device=args.device)

    hybrid = HybridRetrieverRRF(
        bm25_cfg=bm25_cfg,
        dense=dense,
        reranker=reranker,
        k=args.k,
        w_bm25=args.w_bm25,
        w_dense=args.w_dense,
        ce_weight=args.ce_weight,
    )

    # retrieve
    records = hybrid.search(
        query=args.query, topk=args.topk,
        content_path=args.content_path, content_dir=args.content_dir,
        bm25_topk=args.bm25_topk, dense_topk=args.dense_topk, ce_candidates=args.ce_candidates,
        ticker=args.ticker, form=args.form, year=args.year
    )

    # answer
    llm = LLMClient(base_url=args.llm_base_url, model=args.llm_model, api_key=args.llm_api_key)
    out = answer_with_llm(args.query, records, llm, max_ctx_tokens=args.max_context_tokens)

    if args.json_out:
        print(json.dumps(out, ensure_ascii=False))
        return

    # Pretty print
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
