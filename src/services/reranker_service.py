from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
import torch

app = FastAPI()
model = CrossEncoder("BAAI/bge-reranker-base", device="cuda")
MAX_LEN = 512

class RerankRequest(BaseModel):
    query: str
    candidates: List[str]
    top_k: int = 10

class RerankItem(BaseModel):
    text: str
    score: float

class RerankResponse(BaseModel):
    results: List[RerankItem]

@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    pairs = [(req.query, c) for c in req.candidates]
    # 可调 max_length
    scores = model.predict(pairs, batch_size=64, show_progress_bar=False, max_length=MAX_LEN)
    order = scores.argsort()[::-1][:req.top_k]
    results = [RerankItem(text=req.candidates[i], score=float(scores[i])) for i in order]
    return RerankResponse(results=results)
