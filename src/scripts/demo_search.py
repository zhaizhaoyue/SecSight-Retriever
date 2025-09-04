# src/scripts/test_tsla_noncurrent.py
from __future__ import annotations
from pathlib import Path
import re
import json
import numpy as np

from src.rag.retriever.dense_retriever import HybridRetriever
from src.rag.reranker import Reranker

INDEX_DIR = "data/index/faiss_bge_base_en"
TICKER = "TSLA"
YEAR = 2023
FORM = "10-K"

QUERY = (
    "Total noncurrent liabilities (long-term liabilities) amount — "
    "Consolidated Balance Sheets — Tesla 2023"
)

# ---- 关键词&同义词（用于轻量预过滤）----
KEYS = (
    "noncurrent liabilities", "non-current liabilities", "long-term liabilities",
    "noncurrent liability", "non-current liability", "long-term liability",
)

# ---- 如果 meta 里有 heading/section，优先这些关键词 ----
SECTION_HINTS = ("consolidated balance sheet", "balance sheets", "statement of financial position")

def has_keywords(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in KEYS)

def in_target_section(meta: dict) -> bool:
    h = (meta.get("heading") or meta.get("section") or "").lower()
    return any(s in h for s in SECTION_HINTS)

def pseudo_page(meta: dict) -> int:
    p = meta.get("page")
    if p is not None:
        try:
            return int(p)
        except Exception:
            pass
    # 生成“伪页码”：每4个chunk算一页（可按你的切块策略调整）
    return 1 + int(meta.get("chunk_index", 0)) // 4

def brief(item: dict) -> str:
    m = item["meta"]
    return f'{m.get("ticker")} {m.get("year")} {m.get("form")} p{pseudo_page(m)} | {m.get("chunk_id")}'

def print_top(title: str, items: list[dict], k=5, score_key="score"):
    print(f"\n== {title} ==")
    for i, x in enumerate(items[:k], 1):
        m = x["meta"]
        score = x.get(score_key, x.get("rerank_score", 0.0))
        snip = (x.get("text") or m.get("text") or m.get("text_preview") or "")[:220].replace("\n", " ")
        print(f"{i:>2}. {score_key}={score:.4f} | {brief(x)}")
        print(f"    ↳ {snip}")

def keyword_prefilter(items: list[dict], min_keep: int = 40) -> list[dict]:
    hits = []
    # 1) 先挑“在目标版块”的
    for x in items:
        if in_target_section(x["meta"]):
            hits.append(x)
    # 2) 再补充“正文包含关键词”的
    if len(hits) < min_keep:
        for x in items:
            t = x.get("text") or x["meta"].get("text") or x["meta"].get("text_preview") or ""
            if has_keywords(t) and x not in hits:
                hits.append(x)
            if len(hits) >= min_keep:
                break
    # 不足则回退
    return hits if hits else items

def build_context(chunks: list[dict], k: int = 8) -> str:
    parts = []
    for i, c in enumerate(chunks[:k], 1):
        m = c["meta"]
        text = c.get("text") or m.get("text") or m.get("text_preview") or ""
        cite = f"[{i}] {brief(c)}"
        parts.append(f"{cite}\n{text}")
    return "\n\n".join(parts)

# 简单正则：在“noncurrent/long-term liabilities”附近抓一个金额（百万/十亿）
MONEY_RE = re.compile(
    r"(?:total\s+)?(?:non[- ]?current|long[- ]?term)\s+liabilit(?:y|ies)[^0-9$]{0,50}"
    r"(\$?\s?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?\s*(?:billion|million)?)",
    flags=re.IGNORECASE
)

def heuristic_extract_amount(context: str) -> str | None:
    m = MONEY_RE.search(context)
    if m:
        return m.group(1).strip()
    # 备选：抓“$ 12,345”这类离关键词最近的数字
    # （可选增强：基于行切分，向上/下两行搜金额）
    return None

def main():
    print("[demo] TSLA noncurrent liabilities test")

    # 1) 混合检索（扩大召回）
    ret = HybridRetriever(INDEX_DIR)
    cands = ret.search_hybrid(
        QUERY,
        ticker=TICKER, year=YEAR, form=FORM,
        rr_k_dense=300, rr_k_bm25=400,  # 先多取一些
        top_k=180
    )

    print_top("Hybrid candidates (pre-rerank)", cands, k=5, score_key="score")

    # 2) 关键词/版块预过滤（保留一批更相关的给重排）
    cands = keyword_prefilter(cands, min_keep=50)

    # 3) 重排
    rer = Reranker()  # 如需CPU：Reranker(device="cpu")
    top = rer.rerank(QUERY, cands, top_k=12)

    print_top("Reranked Top", top, k=8, score_key="rerank_score")

    # 4) 构建上下文
    ctx = build_context(top, k=8)
    print("\n=== Context preview ===")
    print(ctx[:1400], "...\n")

    # 5) 规则抽取（无LLM也先给个“估值”）
    amount = heuristic_extract_amount(ctx)
    if amount:
        print(f"=== Heuristic Answer (no-LLM) ===\nTesla total noncurrent liabilities (FY{YEAR}): {amount}\n"
              f"(source: snippets [1..8] above)")
    else:
        print("=== Heuristic Answer (no-LLM) ===\nNot found in current context. "
              "Try increasing rr_k/top_k, or ensure Balance Sheet section chunks are present.")

    # 6) 如需 LLM 生成，把 ctx + 问题 丢给你的本地/云端模型：
    # answer = my_llm.generate(
    #   system="You are a financial analyst. Answer only using the CONTEXT. Cite [n] snippets.",
    #   user=f"Question: What were Tesla's total noncurrent liabilities in {YEAR}?\n\nCONTEXT:\n{ctx}"
    # )
    # print("\n=== LLM Answer ===\n", answer)

if __name__ == "__main__":
    # 基本检查
    idx = Path(INDEX_DIR)
    assert (idx / "hnsw.index").exists() and (idx / "meta.jsonl").exists(), "index/meta 文件不存在"
    main()

#python -m src.scripts.demo_search