# src/embed/encoder.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class TextEncoder:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: str = "cuda"):
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        vecs = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=normalize,
            show_progress_bar=False, device=None  # 由模型内部处理
        )
        return vecs.astype("float32")  # faiss 习惯用 float32
