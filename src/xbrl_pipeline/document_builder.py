"""Utilities to convert DB fact records into text/segments for the numeric pipeline."""

from __future__ import annotations

from typing import Iterable, List, Tuple
from dataclasses import dataclass

try:
    from src.aie_for_numeric_retrieval.segmentation import DocumentSegment  # type: ignore
    AIE_MISSING = False
except ImportError:
    # AIE module was removed; provide a minimal placeholder so the XBRL helpers can still be imported.
    AIE_MISSING = True

    @dataclass
    class DocumentSegment:  # type: ignore
        id: str
        content: str
        segment_type: str
        start_pos: int
        end_pos: int
        metadata: dict

from .db_access import FactRecord


def _format_fact_line(fact: FactRecord) -> str:
    label = fact.label_text or fact.qname
    parts: List[str] = [f"{label}: {fact.value_raw}"]
    unit = fact.format_unit()
    if unit:
        parts.append(f"Unit={unit}")
    period = fact.format_period()
    if period:
        parts.append(f"Period={period}")
    if fact.entity:
        parts.append(f"Entity={fact.entity}")
    dims = fact.format_dimensions()
    if dims:
        parts.append(f"Dimensions={dims}")
    if fact.footnote_refs:
        refs = ",".join(fact.footnote_refs)
        parts.append(f"Footnotes={refs}")
    return " | ".join(parts)


def build_fact_segments(facts: Iterable[FactRecord]) -> Tuple[List[DocumentSegment], List[str]]:
    """Create segments and the raw lines (for concatenated document text)."""

    segments: List[DocumentSegment] = []
    lines: List[str] = []
    for idx, fact in enumerate(facts):
        line = _format_fact_line(fact)
        lines.append(line)
        metadata = {
            "qname": fact.qname,
            "label": fact.label_text,
            "unit_id": fact.unit_id,
            "context_id": fact.context_id,
            "doc_order": fact.doc_order,
            "ticker": fact.ticker,
            "accession_no": fact.accession_no,
        }
        segments.append(
            DocumentSegment(
                id=f"fact_{idx:06d}",
                content=line,
                segment_type="fact",
                start_pos=idx,
                end_pos=idx,
                metadata=metadata,
            )
        )
    return segments, lines


def build_document_text(lines: Iterable[str]) -> str:
    """Compose the textual document used by the pipeline."""

    return "\n".join(line.strip() for line in lines if line)


__all__ = ["build_fact_segments", "build_document_text"]
