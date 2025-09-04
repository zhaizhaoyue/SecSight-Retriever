# src/scripts/test_tsla_noncurrent.py
from __future__ import annotations
from pathlib import Path
import re
import json
import numpy as np
from collections import defaultdict
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
# 更严格的金额匹配：要求要么带 $ 且有千位分隔（如 $ 12,345），
# 要么带单位（million/billion），避免 0.001/“1”这类噪声。
MONEY_TIGHT = re.compile(
    r"(?:"
    r"\$\s?[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?"
    r"|"
    r"[0-9]+(?:\.[0-9]+)?\s*(?:million|billion)"
    r"|"
    r"\$\s?[0-9]+(?:\.[0-9]+)?\s*(?:million|billion)"
    r")",
    flags=re.IGNORECASE
)

BLACKLIST_LINE = (
    "par value", "preferred stock", "common stock",
    "stockholders’ equity", "stockholders' equity", "equity", "authorized"
)


def load_all_metas(index_dir):
    metas=[]
    with open(f"{index_dir}/meta.jsonl","r",encoding="utf-8") as f:
        for line in f: metas.append(json.loads(line))
    return metas

def expand_neighbors(top_items, metas, window=2):
    # 根据 chunk_index 找邻居
    by_id = {m.get("chunk_id"): m for m in metas}
    out = []
    seen = set()
    for it in top_items:
        m = it["meta"]; base = int(m.get("chunk_index", -1))
        for d in range(-window, window+1):
            ci = base + d
            key = f"{m.get('accno')}::text::chunk-{ci}"
            nm = by_id.get(key)
            if not nm or key in seen: continue
            seen.add(key)
            out.append({
                "text": nm.get("text") or nm.get("text_preview") or "",
                "meta": nm,
                "score": 0.0,  # 占位
                "faiss_id": ci,  # 仅用于排序展示
            })
    return out

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

# --- line-wise money extractor (add near heuristic_extract_amount) ---
LINE_MONEY = re.compile(r"\$?\s?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?", re.I)

def linewise_extract_amount(context: str) -> str | None:
    lines = [l.strip() for l in context.splitlines() if l.strip()]
    for i, l in enumerate(lines):
        low = l.lower()
        # 必须同时包含 liabilities 和 noncurrent/long-term
        if ("liabilit" in low) and ("noncurrent" in low or "non-current" in low or "long-term" in low):
            # 跳过明显的股本/面值等干扰行
            if any(b in low for b in BLACKLIST_LINE):
                continue
            # 先在本行找严格金额
            money = MONEY_TIGHT.findall(l)
            if money:
                return money[-1]
            # 向下看 1-2 行（表格断行常见）
            for j in (i + 1, i + 2):
                if j < len(lines):
                    lowj = lines[j].lower()
                    if any(b in lowj for b in BLACKLIST_LINE):
                        continue
                    mm = MONEY_TIGHT.findall(lines[j])
                    if mm:
                        return mm[-1]
    return None



def main():
    print("[demo] TSLA noncurrent liabilities test")

    # 1) 混合检索（扩大召回）
    ret = HybridRetriever(INDEX_DIR)
    cands = ret.search_hybrid(
        QUERY,
        ticker=TICKER, year=YEAR, form=FORM,
        rr_k_dense=600, rr_k_bm25=800,   # ↑ 召回更大
        top_k=300
    )

    print_top("Hybrid candidates (pre-rerank)", cands, k=5, score_key="score")

    # 2) 关键词/版块预过滤（保留一批更相关的给重排）
    cands = keyword_prefilter(cands, min_keep=80)

    # 3) 重排
    # 3) 重排
    rer = Reranker()
    top = rer.rerank(QUERY, cands, top_k=20)

    # 3.1 加载所有 meta（用于找相邻块）
    metas_all = load_all_metas(INDEX_DIR)

    # 3.2 取每个命中块的邻居（chunk_index±4）
    neighbors = expand_neighbors(top, metas_all, window=4)
    if neighbors is None:
        neighbors = []  # 防御式：保证是 list

    # 3.3 合并：先邻居后命中；按 chunk_id 去重
    combo = neighbors + top  # 即使 neighbors 为空也没关系
    seen = set()
    merged = []
    for x in combo:
        m = x.get("meta") or {}
        cid = m.get("chunk_id")
        if not cid:
            # 防御：没有 chunk_id 的也不要让它炸
            continue
        if cid in seen:
            continue
        seen.add(cid)
        # 确保 text 字段可用（用于后续显示/抽取）
        if not x.get("text"):
            x["text"] = m.get("text") or m.get("text_preview") or ""
        merged.append(x)

    # 兜底：如果 merged 还是空，用 top 兜底，避免 UnboundLocalError
    if not merged:
        merged = top

    # 4) 构建上下文（把 k 放大一些以覆盖完整表格）
    ctx = build_context(merged, k=20)

    print("\n=== Context preview ===")
    print(ctx[:1400], "...\n")

    # 5) 规则抽取（无LLM也先给个“估值”）
    amount = heuristic_extract_amount(ctx) or linewise_extract_amount(ctx)

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


#python -m src.scripts.test_tsla_noncurrent