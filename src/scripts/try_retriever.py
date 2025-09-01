# scripts/try_retriever.py
from src.rag.retriever import FaissRetriever, HybridRetriever

INDEX_DIR = "data/index/faiss_bge_base_en"
QUERY = "What were Apple's noncurrent liabilities in 2023?"

def show(items, title, k=5):
    print(f"\n== {title} ==")
    for i, x in enumerate(items[:k], 1):
        m = x["meta"]
        print(f"{i:>2}. score={x['score']:.4f} | {m.get('ticker')} {m.get('year')} {m.get('form')} p{m.get('page')} | {m.get('chunk_id')}")

def main():
    # 仅向量检索
    ret = FaissRetriever(INDEX_DIR)
    dense = ret.search(QUERY, top_k=50, ticker="AAPL", year=2023, form="10-K")
    show(dense, "Dense Top-5")

    # 混合检索（dense + BM25 + RRF）
    hyr = HybridRetriever(INDEX_DIR)
    fused = hyr.search_hybrid(QUERY, top_k=50, ticker="AAPL", year=2023, form="10-K")
    show(fused, "Hybrid (RRF) Top-5")

if __name__ == "__main__":
    main()

#python -m src.scripts.try_retriever.py
