"""Utility helpers to read processed XBRL data from the relational store."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


def _ensure_engine(db_url_or_engine: Engine | str) -> Engine:
    if isinstance(db_url_or_engine, Engine):
        return db_url_or_engine
    return create_engine(db_url_or_engine, future=True)


def _safe_json_load(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


@dataclass
class FactRecord:
    """Joined representation of a fact with its context/unit metadata."""

    filing_id: str
    qname: str
    value_raw: Optional[str]
    is_numeric: bool
    decimals: Optional[str]
    unit_id: Optional[str]
    context_id: Optional[str]
    doc_order: Optional[int]
    label_text: Optional[str]
    label_role: Optional[str]
    label_lang: Optional[str]
    accession_no: Optional[str]
    ticker: Optional[str]
    source_path: Optional[str]
    footnote_refs: Optional[Iterable[str]]
    measures: Optional[Dict[str, Any]]
    context_dimensions: Optional[List[Dict[str, Any]]]
    entity: Optional[str]
    period_start: Optional[str]
    period_end: Optional[str]
    period_instant: Optional[str]
    is_forever: Optional[bool]

    def format_period(self) -> str:
        if self.is_forever:
            return "Forever"
        if self.period_instant:
            return str(self.period_instant)
        if self.period_start or self.period_end:
            return f"{self.period_start or '?'} → {self.period_end or '?'}"
        return "Unknown period"

    def format_unit(self) -> str:
        if not self.measures:
            return ""
        numerators = self.measures.get("numerators") or []
        denominators = self.measures.get("denominators") or []
        if denominators:
            return f"{' * '.join(numerators)}/{ ' * '.join(denominators)}"
        if numerators:
            return " * ".join(numerators)
        return ""

    def format_dimensions(self) -> str:
        if not self.context_dimensions:
            return ""
        parts = [f"{d.get('dimension')}: {d.get('member')}" for d in self.context_dimensions if d]
        return ", ".join(parts)


FACT_BASE_QUERY = """
WITH ranked_labels AS (
    SELECT
        l.*, ROW_NUMBER() OVER (
            PARTITION BY l.filing_id, l.qname
            ORDER BY
                CASE WHEN l.role = :preferred_role THEN 0 ELSE 1 END,
                COALESCE(l.lang, ''),
                l.qname
        ) AS rn
    FROM labels l
)
SELECT
    f.filing_id,
    f.qname,
    f.value_raw,
    f.is_numeric,
    f.decimals,
    f.unit_id,
    f.context_id,
    f.doc_order,
    f.label_role,
    f.footnote_refs,
    f.accession_no,
    f.ticker,
    f.source_path,
    ctx.entity,
    ctx.period_start,
    ctx.period_end,
    ctx.period_instant,
    ctx.is_forever,
    ctx.dimensions,
    u.measures,
    rl.text   AS label_text,
    rl.role   AS label_role,
    rl.lang   AS label_lang
FROM facts f
LEFT JOIN contexts ctx
  ON f.filing_id = ctx.filing_id AND f.context_id = ctx.context_id
LEFT JOIN units u
  ON f.filing_id = u.filing_id AND f.unit_id = u.unit_id
LEFT JOIN ranked_labels rl
  ON rl.filing_id = f.filing_id AND rl.qname = f.qname AND rl.rn = 1
WHERE 1=1
"""


def fetch_fact_records(
    db_url_or_engine: Engine | str,
    *,
    filing_id: Optional[str] = None,
    accession_no: Optional[str] = None,
    ticker: Optional[str] = None,
    concept_like: Optional[str] = None,
    numeric_only: bool = False,
    limit: Optional[int] = 500,
) -> List[FactRecord]:
    """Return fact rows enriched with context/unit/label information."""

    eng = _ensure_engine(db_url_or_engine)
    sql = [FACT_BASE_QUERY]
    params: Dict[str, Any] = {"preferred_role": "http://www.xbrl.org/2003/role/label"}

    if filing_id:
        sql.append("AND f.filing_id = :filing_id")
        params["filing_id"] = filing_id
    if accession_no:
        sql.append("AND f.accession_no = :accession_no")
        params["accession_no"] = accession_no
    if ticker:
        sql.append("AND f.ticker = :ticker")
        params["ticker"] = ticker
    if concept_like:
        sql.append("AND LOWER(f.qname) LIKE :concept_like")
        params["concept_like"] = f"%{concept_like.lower()}%"
    if numeric_only:
        sql.append("AND f.is_numeric = :numeric_true")
        params["numeric_true"] = True

    sql.append("ORDER BY CASE WHEN f.doc_order IS NULL THEN 1 ELSE 0 END, f.doc_order, f.qname")
    if limit:
        sql.append("LIMIT :limit")
        params["limit"] = int(limit)

    statement = text("\n".join(sql))

    with eng.connect() as conn:
        rows = conn.execute(statement, params).mappings().all()

    facts: List[FactRecord] = []
    for row in rows:
        facts.append(
            FactRecord(
                filing_id=row.get("filing_id"),
                qname=row.get("qname"),
                value_raw=row.get("value_raw"),
                is_numeric=bool(row.get("is_numeric")),
                decimals=row.get("decimals"),
                unit_id=row.get("unit_id"),
                context_id=row.get("context_id"),
                doc_order=row.get("doc_order"),
                label_text=row.get("label_text"),
                label_role=row.get("label_role"),
                label_lang=row.get("label_lang"),
                accession_no=row.get("accession_no"),
                ticker=row.get("ticker"),
                source_path=row.get("source_path"),
                footnote_refs=_safe_json_load(row.get("footnote_refs")) or [],
                measures=_safe_json_load(row.get("measures")) or {},
                context_dimensions=_safe_json_load(row.get("dimensions")) or [],
                entity=row.get("entity"),
                period_start=row.get("period_start"),
                period_end=row.get("period_end"),
                period_instant=row.get("period_instant"),
                is_forever=row.get("is_forever"),
            )
        )

    logger.info("Fetched %d facts from DB", len(facts))
    return facts
