# scripts/demo_search.py
from pathlib import Path
import json
import re
import numpy as np

from src.rag.retriever import FaissRetriever, HybridRetriever
from src.rag.reranker import Reranker

INDEX_DIR = "data/index/faiss_bge_base_en"

def main():
    query = "What were Apple's noncurrent liabilities in 2023?"

    # 任选其一：dense 或 hybrid
    # ---- dense ----
    ret_dense = FaissRetriever(INDEX_DIR)
    dense_cands = ret_dense.search(query, top_k=100, ticker="AAPL", year=2023, form="10-K")

    # ---- hybrid (dense+bm25) ----
    ret_hybrid = HybridRetriever(INDEX_DIR)
    hybrid_cands = ret_hybrid.search_hybrid(query, top_k=100, ticker="AAPL", year=2023, form="10-K")

    # 重排（对 hybrid 结果重排）
    rer = Reranker()
    top = rer.rerank(query, hybrid_cands, top_k=10)

    print("\n=== Reranked (Hybrid) Top 5 ===")
    for i, x in enumerate(top[:5], 1):
        m = x["meta"]
        print(f"{i:>2}. rerank={x['rerank_score']:.3f} | {m.get('ticker')} {m.get('year')} {m.get('form')} "
              f"| page={m.get('page')} | id={m.get('chunk_id')}")
        snip = (x.get("text") or x["meta"].get("text") or "")[:220].replace("\n", " ")
        print(f"    ↳ {snip}")

    # 需要融合对比就打开：
    demo_fusion(query, INDEX_DIR, ticker="AAPL", year=2023, form="10-K", top_k=10)

# ------------------------------
# Fusion rules demo
# ------------------------------
from rank_bm25 import BM25Okapi

def _tok(s: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9$%\.]+", (s or "").lower())

def _ensure_text(meta: dict) -> str:
    return meta.get("text") or meta.get("raw_text") or meta.get("text_preview") or ""

def build_bm25_from_index(index_dir: str) -> tuple[BM25Okapi, list[dict]]:
    metas = []
    with open(f"{index_dir}/meta.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    docs = [_tok(_ensure_text(m)) for m in metas]
    bm25 = BM25Okapi(docs)
    return bm25, metas

def rrf_fuse(dense: list[dict], bm25: list[dict], c: float = 60.0, top_k: int = 10) -> list[dict]:
    rank_d = {it["faiss_id"]: r for r, it in enumerate(dense, start=1)}
    rank_b = {it["faiss_id"]: r for r, it in enumerate(bm25, start=1)}
    ids = set(rank_d) | set(rank_b)

    fused = []
    for i in ids:
        s = 0.0
        if i in rank_d: s += 1.0 / (c + rank_d[i])
        if i in rank_b: s += 1.0 / (c + rank_b[i])
        src = next((x for x in dense if x["faiss_id"] == i), None) or \
              next((x for x in bm25 if x["faiss_id"] == i), None)
        fused.append({**src, "fuse_score": float(s)})
    fused.sort(key=lambda x: x["fuse_score"], reverse=True)
    return fused[:top_k]

def demo_fusion(query: str, index_dir: str, ticker=None, year=None, form=None, top_k=10):
    print("\n=== [Fusion Demo] ===")
    dense_ret = FaissRetriever(index_dir)
    dense = dense_ret.search(query, top_k=100, ticker=ticker, year=year, form=form)

    bm25, metas = build_bm25_from_index(index_dir)
    scores = bm25.get_scores(_tok(query))
    top_idx = np.argsort(scores)[::-1][:400]
    bm25_list = []
    for i in top_idx:
        m = metas[i]
        if not dense_ret._passes_filter(m, ticker, year, form):
            continue
        bm25_list.append({
            "text": _ensure_text(m),
            "score": float(scores[i]),
            "meta": m,
            "faiss_id": int(i),
        })
        if len(bm25_list) >= 100:
            break

    fused = rrf_fuse(dense, bm25_list, c=60.0, top_k=top_k)

    def _brief(item):
        m = item["meta"]
        return f"{m.get('ticker')} {m.get('year')} {m.get('form')} p{m.get('page')} | {m.get('chunk_id')}"

    print("\n-- Dense-only Top 5 --")
    for i, x in enumerate(dense[:5], 1):
        print(f"{i:>2}. score={x['score']:.4f} | {_brief(x)}")
        snip = (x.get("text") or x["meta"].get("text") or "")[:220].replace("\n", " ")
        print(f"    ↳ {snip}")

    print("\n-- BM25-only Top 5 --")
    for i, x in enumerate(bm25_list[:5], 1):
        print(f"{i:>2}. score={x['score']:.4f} | {_brief(x)}")
        snip = (x.get("text") or x["meta"].get("text") or "")[:220].replace("\n", " ")
        print(f"    ↳ {snip}")

    print("\n-- RRF Fused Top 5 --")
    for i, x in enumerate(fused[:5], 1):
        print(f"{i:>2}. fuse={x['fuse_score']:.4f} | {_brief(x)}")
        snip = (x.get("text") or x["meta"].get("text") or "")[:220].replace("\n", " ")
        print(f"    ↳ {snip}")

    ids_dense = {x["faiss_id"] for x in dense[:top_k]}
    ids_bm25  = {x["faiss_id"] for x in bm25_list[:top_k]}
    ids_fused = {x["faiss_id"] for x in fused[:top_k]}
    print(f"\n[stats] top{top_k} overlap: dense∩bm25={len(ids_dense & ids_bm25)}, "
          f"dense∩fused={len(ids_dense & ids_fused)}, bm25∩fused={len(ids_bm25 & ids_fused)}")


if __name__ == "__main__":
    print("[demo] starting…")
    # 基本检查
    idx = Path(INDEX_DIR)
    assert (idx/"hnsw.index").exists() and (idx/"meta.jsonl").exists(), "index/meta 文件不存在"
    main()


#python -m src.scripts.demo_search