# -*- coding: utf-8 -*-
import json
import logging
from typing import List, Dict, Any
import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

DDL = {
"facts": """
CREATE TABLE IF NOT EXISTS facts(
  id BIGSERIAL PRIMARY KEY,
  filing_id TEXT,
  accession_no TEXT,
  ticker TEXT,
  fact_oid TEXT,
  qname TEXT NOT NULL,
  value_raw TEXT,
  is_numeric BOOLEAN,
  decimals TEXT,
  unit_id TEXT,
  context_id TEXT,
  label_role TEXT,
  footnote_refs JSONB,
  doc_order INT,
  source_path TEXT,
  load_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""",
"contexts": """
CREATE TABLE IF NOT EXISTS contexts(
  filing_id TEXT,
  accession_no TEXT,
  ticker TEXT,
  context_id TEXT,
  entity TEXT,
  period_start DATE,
  period_end DATE,
  period_instant DATE,
  is_forever BOOLEAN,
  dimensions JSONB,
  source_path TEXT,
  load_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""",
"units": """
CREATE TABLE IF NOT EXISTS units(
  filing_id TEXT,
  accession_no TEXT,
  ticker TEXT,
  unit_id TEXT,
  measures JSONB,
  source_path TEXT,
  load_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""",
"labels": """
CREATE TABLE IF NOT EXISTS labels(
  filing_id TEXT,
  accession_no TEXT,
  ticker TEXT,
  qname TEXT,
  role TEXT,
  lang TEXT,
  text TEXT,
  source_path TEXT,
  load_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""",
"calc_edges": """
CREATE TABLE IF NOT EXISTS calc_edges(
  filing_id TEXT,
  accession_no TEXT,
  ticker TEXT,
  from_qname TEXT,
  to_qname TEXT,
  weight NUMERIC,
  "order" NUMERIC,
  preferred_label TEXT,
  arcrole TEXT,
  source_path TEXT,
  load_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""",
"pre_edges": """
CREATE TABLE IF NOT EXISTS pre_edges(
  filing_id TEXT,
  accession_no TEXT,
  ticker TEXT,
  from_qname TEXT,
  to_qname TEXT,
  weight NUMERIC,
  "order" NUMERIC,
  preferred_label TEXT,
  arcrole TEXT,
  source_path TEXT,
  load_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""",
"def_edges": """
CREATE TABLE IF NOT EXISTS def_edges(
  filing_id TEXT,
  accession_no TEXT,
  ticker TEXT,
  from_qname TEXT,
  to_qname TEXT,
  weight NUMERIC,
  "order" NUMERIC,
  preferred_label TEXT,
  arcrole TEXT,
  source_path TEXT,
  load_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""",
}

_META_COLUMNS = {
    "facts": {
        "accession_no": "TEXT",
        "ticker": "TEXT",
        "source_path": "TEXT",
        "load_ts": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    },
    "contexts": {
        "accession_no": "TEXT",
        "ticker": "TEXT",
        "source_path": "TEXT",
        "load_ts": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    },
    "units": {
        "accession_no": "TEXT",
        "ticker": "TEXT",
        "source_path": "TEXT",
        "load_ts": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    },
    "labels": {
        "accession_no": "TEXT",
        "ticker": "TEXT",
        "source_path": "TEXT",
        "load_ts": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    },
    "calc_edges": {
        "accession_no": "TEXT",
        "ticker": "TEXT",
        "source_path": "TEXT",
        "load_ts": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    },
    "pre_edges": {
        "accession_no": "TEXT",
        "ticker": "TEXT",
        "source_path": "TEXT",
        "load_ts": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    },
    "def_edges": {
        "accession_no": "TEXT",
        "ticker": "TEXT",
        "source_path": "TEXT",
        "load_ts": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    }
}


def _column_exists(conn, table: str, column: str) -> bool:
    """Check whether a column exists in a table for the current dialect."""
    dialect = conn.dialect.name.lower()
    try:
        if "sqlite" in dialect:
            res = conn.execute(text(f"PRAGMA table_info({table})"))
            return any(row[1] == column for row in res)
        query = text(
            """
            SELECT 1 FROM information_schema.columns
            WHERE table_name = :table AND column_name = :column
            LIMIT 1
            """
        )
        return conn.execute(query, {"table": table, "column": column}).first() is not None
    except Exception:
        return False


def _ensure_meta_columns(conn) -> None:
    """Ensure metadata columns exist for each table, adding them if missing."""
    for table, columns in _META_COLUMNS.items():
        for column, decl in columns.items():
            if not _column_exists(conn, table, column):
                try:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {decl}"))
                except Exception as exc:
                    logger.warning("Failed to add column %s to %s: %s", column, table, exc)


def init_db(db_url: str):
    eng = create_engine(db_url, future=True)
    with eng.begin() as conn:
        for ddl in DDL.values():
            conn.execute(text(ddl))
        _ensure_meta_columns(conn)
    return eng

def _df(table_rows: List[Dict[str, Any]]):
    if not table_rows:
        return pd.DataFrame()
    return pd.DataFrame(table_rows)

def save_all(db_url: str,
             filing_id: str,
             meta: Dict[str, Any],
             facts: List[Dict[str, Any]],
             contexts: List[Dict[str, Any]],
             units: List[Dict[str, Any]],
             labels: List[Dict[str, Any]],
             calc_edges: List[Dict[str, Any]],
             pre_edges: List[Dict[str, Any]],
             def_edges: List[Dict[str, Any]]):
    eng = init_db(db_url)

    # 附加 filing_id / 元数据（source_path）
    def attach(rows):
        for r in rows:
            r["filing_id"] = filing_id
            if meta.get("source_path"):
                r.setdefault("source_path", meta["source_path"])
            if meta.get("accession_no"):
                r.setdefault("accession_no", meta["accession_no"])
            if meta.get("ticker_hint"):
                r.setdefault("ticker", meta["ticker_hint"])
        return rows

    facts = attach(facts)
    contexts = attach(contexts)
    units = attach(units)
    labels = attach(labels)
    calc_edges = attach(calc_edges)
    pre_edges = attach(pre_edges)
    def_edges = attach(def_edges)

    # JSON 列序列化
    if facts:
        for r in facts:
            r["footnote_refs"] = json.dumps(r.get("footnote_refs", []))
    if contexts:
        for r in contexts:
            r["dimensions"] = json.dumps(r.get("dimensions", []))
    if units:
        for r in units:
            r["measures"] = json.dumps(r.get("measures", {}))

    with eng.begin() as conn:
        if facts:      _df(facts).to_sql("facts", conn, if_exists="append", index=False)
        if contexts:   _df(contexts).to_sql("contexts", conn, if_exists="append", index=False)
        if units:      _df(units).to_sql("units", conn, if_exists="append", index=False)
        if labels:     _df(labels).to_sql("labels", conn, if_exists="append", index=False)
        if calc_edges: _df(calc_edges).to_sql("calc_edges", conn, if_exists="append", index=False)
        if pre_edges:  _df(pre_edges).to_sql("pre_edges", conn, if_exists="append", index=False)
        if def_edges:  _df(def_edges).to_sql("def_edges", conn, if_exists="append", index=False)

    logger.info("Saved filing_id=%s to DB", filing_id)
