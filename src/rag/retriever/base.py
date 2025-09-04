from __future__ import annotations
from typing import Protocol, TypedDict, List, Dict, Any, Optional

class Hit(TypedDict, total=False):
    chunk_id: str
    title: str
    content: str
    heading: str
    snippet: str
    meta: Dict[str, Any]
    score: float
    score_dense: float
    score_sparse: float

class Retriever(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 8,
        ticker: Optional[str] = None,
        year: Optional[int] = None,
        form: Optional[str] = None,
    ) -> List[Hit]:
        ...
