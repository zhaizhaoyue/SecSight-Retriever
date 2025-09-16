# -*- coding: utf-8 -*-
"""
RAG Answerer — plugs HybridRetrieverRRF into an LLM (OpenAI-compatible or OpenAI).
Usage example:

python -m src.rag.retriever.answer_api `
  --query "What were Google’s advertising revenues as reported in Alphabet’s 2023 10-K?" `
  --ticker GOOGL --form 10-K --year 2024 `
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
5) Use at most 3–5 citations unless necessary.
6) Also treat passages describing constraints/conditions tied to the risk as part of the risk context
(e.g., practical solutions that must not undermine affordability or reliability).
If evidence mentions affordability/reliability, net-zero, or NDCs, include them if relevant.
7) When amounts/dates appear, quote the exact figure and unit; prefer the most recent within the evidence.
8) When multiple periods are present, state the period explicitly (YYYY-MM-DD or FY/FQ) and cite both sides.
9) If evidence conflicts, state the discrepancy in one sentence and cite both sources.




"""

def build_evidence_block(evs: List[Evidence], max_chars_per_block: int = 1200) -> str:
    lines = []
    for e in evs:
        txt = e.text.strip()
        if len(txt) > max_chars_per_block:
            txt = txt[:max_chars_per_block] + " ..."
        lines.append(f"[{e.rid}]\n{txt}\n")
    return "\n".join(lines)

# ---------------- Context packing ----------------
def pack_evidence(records: List[Dict[str, Any]],
                  max_tokens: int = 2400) -> List[Evidence]:
    # sort by fused final_score (desc)
    recs = sorted(records, key=lambda r: r.get("final_score", 0.0), reverse=True)

    # simple dedupe by prefix signature
    seen: Set[str] = set()
    uniq: List[Evidence] = []
    for r in recs:
        rid = r["id"]
        text = (r.get("content") or r.get("snippet") or "").strip()
        sig = text[:80].lower().strip()
        if not text or sig in seen:
            continue
        seen.add(sig)
        uniq.append(Evidence(rid=rid, text=text, score=r.get("final_score", 0.0), meta=r.get("meta", {}) or {}))

    # token budget
    packed: List[Evidence] = []
    total = 0
    for e in uniq:
        tks = _estimate_tokens(e.text)
        if total + tks <= max_tokens:
            packed.append(e); total += tks
            continue
        # try to truncate the last one
        remain = max_tokens - total
        if remain > 50:
            # rough truncation by chars proportional to tokens
            keep_chars = remain * 4
            trunc = Evidence(e.rid, e.text[:keep_chars] + " ...", e.score, e.meta)
            packed.append(trunc)
        break
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
