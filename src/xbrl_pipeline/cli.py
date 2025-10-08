# -*- coding: utf-8 -*-
"""Command line interface for XBRL ingestion and numeric retrieval.
python -m src.xbrl_pipeline.cli ingest `
    --entry "data/raw_reports/sec-edgar-filings/AAPL/10-K/0000320193-20-000096/FilingSummary.xml" `
    --db sqlite:///finxbrl.db `
    --filing-id AAPL_2020_10K


python -m src.xbrl_pipeline.cli run `
  --db sqlite:///finxbrl.db `
  --filing-id AAPL_2020_10K `
  --query "What was Apple's 2020 revenue?" `
  --config configs/aie_numeric_config.yaml `
  --preview 5


"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from .document_builder import build_fact_segments
from .loader import close, guess_filing_meta, load_model_xbrl
from .extract_core import extract_contexts, extract_facts, extract_units
from .extract_labels import extract_labels
from .extract_links import extract_calc_edges, extract_def_edges, extract_pre_edges
from .numeric_pipeline import PipelineRequest, XbrlNumericPipeline
from .db_access import fetch_fact_records
from .save_db import save_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_targets(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    data = _load_yaml(path)
    if isinstance(data, list):
        return data
    for key in ("extraction_targets", "targets"):
        if isinstance(data.get(key), list):
            return data[key]
    raise ValueError(f"Unsupported targets format in {path}")


def _run_ingest(args: argparse.Namespace) -> None:
    filing_id = (
        args.filing_id
        or os.path.basename(os.path.dirname(os.path.abspath(args.entry)))
        or os.path.basename(args.entry)
    )
    meta = guess_filing_meta(args.entry)

    cntlr, model = load_model_xbrl(args.entry)

    try:
        facts = extract_facts(model)
        ctxs = extract_contexts(model)
        units = extract_units(model)
        labels = extract_labels(model)
        calc_e = extract_calc_edges(model)
        pre_e = extract_pre_edges(model)
        def_e = extract_def_edges(model)

        save_all(args.db, filing_id, meta, facts, ctxs, units, labels, calc_e, pre_e, def_e)
        logger.info("Ingestion complete for filing_id=%s", filing_id)
    finally:
        close(cntlr)


def _resolve_llm_config(pipeline_cfg: Dict[str, Any], llm_config_path: Optional[str]) -> Dict[str, Any]:
    if llm_config_path:
        return _load_yaml(llm_config_path)
    if pipeline_cfg.get("llm"):
        return dict(pipeline_cfg["llm"])
    models_cfg = pipeline_cfg.get("models", {})
    if models_cfg.get("llm"):
        return dict(models_cfg["llm"])
    raise ValueError("LLM configuration not found; provide --llm-config or add 'llm' section to config file")


def _preview_facts(facts, count: int) -> None:
    if count <= 0 or not facts:
        return
    _, lines = build_fact_segments(facts[:count])
    print("\nPreview of fact snippets:")
    for line in lines:
        print(f"  - {line}")
    print()


def _run_numeric(args: argparse.Namespace) -> None:
    pipeline_cfg = _load_yaml(args.config)
    if not pipeline_cfg:
        raise ValueError(f"Empty pipeline config: {args.config}")

    llm_cfg = _resolve_llm_config(pipeline_cfg, args.llm_config)
    targets = _load_targets(args.targets_config)

    runner = XbrlNumericPipeline(args.db, pipeline_cfg, llm_cfg)
    request = PipelineRequest(
        filing_id=args.filing_id,
        query=args.query,
        concept_like=args.concept,
        limit=args.limit,
        ticker=args.ticker,
        accession_no=args.accession,
    )

    facts = fetch_fact_records(
        args.db,
        filing_id=request.filing_id,
        accession_no=request.accession_no,
        ticker=request.ticker,
        concept_like=request.concept_like,
        limit=request.limit,
        numeric_only=False,
    )

    if not facts:
        raise RuntimeError(
            f"No facts retrieved for filing_id={request.filing_id!r}. Ensure the filing was ingested first."
        )

    _preview_facts(facts, args.preview)

    result = runner.run(request, extraction_targets=targets, facts=facts)

    print(f"\nQuery: {request.query}")
    print(f"Document ID: {result.document_id}")
    print(f"Summary length: {len(result.summary.summary)} characters")
    print("Extractions:")
    for extraction in result.extractions:
        value = extraction.value if extraction.value is not None else "<null>"
        print(
            f"  - {extraction.target.name}: {value} | confidence={extraction.confidence:.3f}"
        )

    if args.output:
        payload = result.to_dict()
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Result JSON saved to %s", output_path.as_posix())


# ---------------------------------------------------------------------------
# Argument parsing


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XBRL ingestion and numeric retrieval")
    sub = parser.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="Parse XBRL/iXBRL filing and store into DB")
    ing.add_argument("--entry", required=True, help="*_htm.xml / primary-document.html / instance .xml")
    ing.add_argument("--db", required=True, help="SQLAlchemy URL, e.g. sqlite:///finxbrl.db")
    ing.add_argument("--filing-id", default=None, help="Override filing identifier (default uses path heuristics)")

    run = sub.add_parser("run", help="Execute numeric retrieval pipeline against ingested data")
    run.add_argument("--db", required=True, help="SQLAlchemy URL for the processed facts database")
    run.add_argument("--filing-id", required=True, help="Filing ID used during ingestion")
    run.add_argument("--query", required=True, help="Natural language query")
    run.add_argument("--concept", help="Optional substring filter applied to concept qnames")
    run.add_argument("--limit", type=int, default=500, help="Maximum number of facts to load (default: 500)")
    run.add_argument("--ticker", help="Ticker filter when multiple filings share the same ID")
    run.add_argument("--accession", help="Accession number filter")
    run.add_argument("--config", default="configs/aie_numeric_config.yaml", help="Pipeline YAML configuration")
    run.add_argument("--llm-config", help="Explicit LLM config file (overrides config.llm)")
    run.add_argument("--targets-config", help="Optional YAML with extraction target definitions")
    run.add_argument("--preview", type=int, default=0, help="Print first N fact lines before running")
    run.add_argument("--output", help="Path to save result JSON payload")

    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "ingest":
        _run_ingest(args)
    elif args.command == "run":
        _run_numeric(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
