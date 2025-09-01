from sentence_transformers import CrossEncoder

# 建议：base 版，显存友好
model = CrossEncoder("BAAI/bge-reranker-base", device="cuda")

def rerank(query: str, candidates: list[str], top_k: int = 10):
    pairs = [(query, c) for c in candidates]
    # 你可以调 max_length（默认 512），超长会截断
    scores = model.predict(pairs, batch_size=64, convert_to_numpy=True)
    order = scores.argsort()[::-1]
    top = [(candidates[i], float(scores[i])) for i in order[:top_k]]
    return top

# 示例
query = "What are Apple's noncurrent liabilities?"
cands = ["...candidate text 1...", "...candidate text 2...", "..."]
print(rerank(query, cands, top_k=5))
