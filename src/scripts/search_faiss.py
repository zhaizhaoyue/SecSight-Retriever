# scripts/search_faiss.py
import numpy as np
from pathlib import Path
from src.embed.encoder import TextEncoder
from src.indexer.faiss_index import FaissHNSWIndex

INDEX_DIR = Path("data/index/faiss_bge_base_en")

def main():
    enc = TextEncoder("BAAI/bge-base-en-v1.5", device="cuda")
    index = FaissHNSWIndex.load(dim=enc.dim, dir_=INDEX_DIR)

    query = "What were Apple's noncurrent liabilities in 2023?"
    qv = enc.encode([query], batch_size=1, normalize=True)
    results = index.search(qv, top_k=10)[0]

    for r in results:
        print(f"{r['score']:.4f} | {r['meta'].get('ticker')} {r['meta'].get('year')} "
              f"| page {r['meta'].get('page')} | {r['meta'].get('chunk_id')}")

if __name__ == "__main__":
    main()
