from __future__ import annotations
from typing import List, Dict, Any
import math
import re
from .metadata import MappingStore

_YOY_PAT = re.compile(r"\b(yoy|year[- ]over[- ]year|compared to 20\d{2}|increase[d]? \d+%|decrease[d]? \d+%)\b", re.I)
_FX_PAT = re.compile(r"\b(foreign exchange|fx|currency|currencies|u\.s\. dollar|usd)\b", re.I)
_GM_PAT = re.compile(r"\bgross margin( percentage| %)\b", re.I)
_RD_PAT = re.compile(r"\b(r&d|research and development)\b", re.I)
_TABLE_3Y_PAT = re.compile(r"\b20\d{2}\b.*\b20\d{2}\b.*\b20\d{2}\b")  # 2023 2022 2021 pattern


def _minmax(scores: List[float]) -> List[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


class RuleReranker:
    def __init__(self, mappings: MappingStore):
        self.m = mappings

    def _rule_score(self, query: str, hit: Dict[str, Any]) -> float:
        q = query.lower()
        h = (hit.get("heading") or "").lower()
        txt = (hit.get("content") or "")
        m = hit.get("meta", {})
        concepts = set(m.get("concepts", []) or [])
        rule = 0.0

        # Heading and concept boosts
        if concepts:
            rule += 0.15
        if any(lbl and lbl.lower() in h for lbl in [self.m.concept_to_main.get(c) for c in concepts]):
            rule += 0.05

        # Query intent heuristics
        if _YOY_PAT.search(q):
            if "compared to" in txt.lower() or _TABLE_3Y_PAT.search(txt):
                rule += 0.08
        if _FX_PAT.search(q):
            if any(k in txt.lower() for k in ["foreign", "currency", "u.s. dollar", "usd"]):
                rule += 0.08
        if _GM_PAT.search(q):
            if "gross margin" in h or "%" in txt:
                rule += 0.08
        if _RD_PAT.search(q):
            if "operating expenses" in h or "research and development" in txt.lower():
                rule += 0.08

        # Penalize signature/cover boilerplate
        if any(x in h for x in ["signature", "cover", "table of contents", "index"]):
            rule -= 0.15

        return max(-0.2, rule)

    def rerank(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not hits:
            return []
        # normalize dense/sparse then fuse with rule
        d = _minmax([h.get("score_dense", 0.0) for h in hits])
        s = _minmax([h.get("score_sparse", 0.0) for h in hits])
        for i, h in enumerate(hits):
            fused = 0.6 * d[i] + 0.3 * s[i] + self._rule_score(query, h)
            h["score"] = fused
        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits
