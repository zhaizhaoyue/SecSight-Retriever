from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-base", device="cuda", max_length=512):
        # 在初始化时传 max_length
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)  # 这里不要传 max_length
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]

