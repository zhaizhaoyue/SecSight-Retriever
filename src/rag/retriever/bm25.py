from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from pathlib import Path
import json

from .metadata import MappingStore
from .filters import filter_hits

@dataclass
class BM25Doc:
    doc_id: str
    tokens: List[str]
    raw: Dict[str, Any]  # original chunk dict {id,title,content,meta}

class BM25Retriever:
    def __init__(self, bm25: BM25Okapi, docs: List[BM25Doc], mappings: MappingStore):
        self.bm25 = bm25
        self.docs = docs
        self.mappings = mappings

    @classmethod
    def from_jsonl(cls, root: str | Path, mappings: MappingStore, pattern: str = "**/text_chunks.jsonl") -> "BM25Retriever":
        root = Path(root)
        docs: List[BM25Doc] = []
        for p in root.rglob(pattern):
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ch = json.loads(line)
                    doc_id = ch.get("id") or ch.get("meta", {}).get("chunk_id")
                    heading = (ch.get("meta", {}) or {}).get("heading") or ""
                    content = ch.get("content") or ""
                    # annotate concepts for weighting fields
                    concepts = mappings.annotate_concepts((heading + "\n" + content))
                    # save back for later rerank
                    ch.setdefault("meta", {})
                    ch["meta"]["concepts"] = list(concepts)

                    # build weighted tokens: concepts*3 + heading*2 + content*1
                    tokens = []
                    # concepts: use main labels
                    concept_tokens = [mappings.concept_to_main.get(c, c) for c in concepts]
                    tokens += mappings.tokenize(" ".join(concept_tokens)) * 3
                    tokens += mappings.tokenize(heading) * 2
                    tokens += mappings.tokenize(content)

                    docs.append(BM25Doc(doc_id=doc_id, tokens=tokens, raw=ch))
        corpus = [d.tokens for d in docs]
        bm25 = BM25Okapi(corpus)
        return cls(bm25, docs, mappings)

    def search(self, query: str, top_k: int = 8, ticker=None, year=None, form=None) -> List[Dict[str, Any]]:
        # expand query with mapping tokens
        exp = self.mappings.expand_query(query)
        q_tokens = self.mappings.tokenize(query) + list(exp["tokens"])  # inject synonyms/years
        scores = self.bm25.get_scores(q_tokens)
        # collect and sort
        cand = []
        for i, sc in enumerate(scores):
            ch = self.docs[i].raw
            cand.append({
                "chunk_id": ch.get("id") or ch.get("meta", {}).get("chunk_id"),
                "score_sparse": float(sc),
                "score_dense": 0.0,
                "meta": ch.get("meta", {}),
                "title": ch.get("title"),
                "content": ch.get("content"),
                "heading": ch.get("meta", {}).get("heading"),
                "snippet": (ch.get("content") or "")[:280]
            })
        cand.sort(key=lambda x: x["score_sparse"], reverse=True)
        cand = cand[: max(top_k*4, top_k)]  # keep a larger pool for later fusion
        # metadata filtering
        cand = filter_hits(cand, ticker=ticker, year=year, form=form)
        return cand[:top_k]