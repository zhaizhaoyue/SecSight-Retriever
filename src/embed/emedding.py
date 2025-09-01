from sentence_transformers import SentenceTransformer
import faiss, numpy as np

model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")  # 或 intfloat/e5-base-v2
def embed(texts: list[str]) -> np.ndarray:
    X = model.encode(texts, batch_size=64, normalize_embeddings=True, device="cuda", show_progress_bar=False)
    return X.astype("float32")

# 建 HNSW 索引
dim = 768  # bge-base-en-v1.5 的维度
index = faiss.IndexHNSWFlat(dim, 32)  # M=32
index.hnsw.efSearch = 128
# add vectors
vecs = embed(corpus_chunks)         # 你的切块文本
index.add(vecs)

# 查询
qv = embed([query])[0:1]
D, I = index.search(qv, 50)         # 先取 top-50 再去重排
candidates = [corpus_chunks[i] for i in I[0]]
