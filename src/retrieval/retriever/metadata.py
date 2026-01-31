# ================================
# file: retriever/metadata.py
# ================================
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
import yaml
import re

_LOWER = lambda s: (s or "").lower()
_TOKEN_RE = re.compile(r"[A-Za-z0-9%\.\-/]+")

@dataclass
class MappingStore:
    labels: Dict[str, Dict[str, Any]]
    labels_best: Dict[str, Dict[str, Any]]
    defs: Dict[str, Dict[str, Any]]
    calcs: Dict[str, Dict[str, Any]]

    # Derived indexes
    concept_to_forms: Dict[str, Set[str]]
    concept_to_main: Dict[str, str]
    form_to_concepts: Dict[str, Set[str]]

    @classmethod
    def load_all(cls, yaml_dir: str | Path) -> "MappingStore":
        yaml_dir = Path(yaml_dir)
        def _load(name: str) -> Dict[str, Any]:
            p = yaml_dir / name
            return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}

        labels = _load("labels.yaml")
        labels_best = _load("labels_best.yaml")
        defs = _load("def.yaml")
        calcs = _load("calc.yaml")

        concept_to_forms: Dict[str, Set[str]] = {}
        concept_to_main: Dict[str, str] = {}
        form_to_concepts: Dict[str, Set[str]] = {}

        # Build concept -> surface forms
        for c, obj in (labels or {}).items():
            forms: Set[str] = set()
            lbl = obj.get("label")
            if lbl: forms.add(lbl)
            for a in obj.get("aliases", []) or []:
                if a: forms.add(a)
            concept_to_forms[c] = {f.lower() for f in forms if f}

        # Best labels
        for c, obj in (labels_best or {}).items():
            if isinstance(obj, dict):
                ml = obj.get("label")
            else:
                ml = obj
            if ml:
                concept_to_main[c] = ml

        # Reverse: surface form -> concepts
        for c, forms in concept_to_forms.items():
            for f in forms:
                form_to_concepts.setdefault(f, set()).add(c)

        return cls(labels, labels_best, defs, calcs, concept_to_forms, concept_to_main, form_to_concepts)

    # Tokenizer used by BM25 indexer & query expander
    def tokenize(self, text: str) -> List[str]:
        return [t.lower() for t in _TOKEN_RE.findall(text or "")]

    # Find concepts mentioned in a text via surface forms
    def annotate_concepts(self, text: str) -> Set[str]:
        low = text.lower()
        hits: Set[str] = set()
        for surface, concepts in self.form_to_concepts.items():
            if surface and surface in low:
                hits.update(concepts)
        return hits

    # Expand query to target concepts and surface tokens (for BM25)
    def expand_query(self, q: str) -> Dict[str, Any]:
        q_low = q.lower()
        targets: Set[str] = set()
        # direct surface hits
        for surface, concepts in self.form_to_concepts.items():
            if surface in q_low:
                targets.update(concepts)
        # naive year keywords
        years = set(re.findall(r"20\d{2}", q))

        # Expand via def.yaml (children/siblings if present)
        expanded: Set[str] = set(targets)
        for c in list(targets):
            node = (self.defs or {}).get(c) or {}
            for k in ("children", "siblings", "related"):
                for cc in node.get(k, []) or []:
                    expanded.add(cc)

        # Collect surface tokens to inject to BM25 query
        tokens: Set[str] = set()
        for c in expanded or targets:
            main = self.concept_to_main.get(c)
            if main: tokens.add(main)
            for f in self.concept_to_forms.get(c, set()):
                tokens.add(f)
        tokens.update(years)

        return {"concepts": expanded or targets, "tokens": tokens}

