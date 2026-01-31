"""XBRL-backed numeric retrieval pipeline wiring AIE to the DB."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

try:
    from src.aie_for_numeric_retrieval.extraction import ExtractionTarget
    from src.aie_for_numeric_retrieval.models.llm_interface import LLMInterface
    from src.aie_for_numeric_retrieval.pipeline import AIEPipeline, AIEPipelineResult
    AIE_MISSING = False
except ImportError:
    AIE_MISSING = True
    ExtractionTarget = None  # type: ignore
    LLMInterface = None  # type: ignore
    AIEPipeline = None  # type: ignore
    AIEPipelineResult = None  # type: ignore

from .db_access import FactRecord, fetch_fact_records
from .document_builder import build_document_text, build_fact_segments

logger = logging.getLogger(__name__)


@dataclass
class PipelineRequest:
    filing_id: str
    query: str
    concept_like: Optional[str] = None
    limit: int = 500
    ticker: Optional[str] = None
    accession_no: Optional[str] = None


class XbrlNumericPipeline:
    """Glue code to execute the numeric AIE pipeline on DB-backed XBRL facts."""

    def __init__(
        self,
        db_url: str,
        pipeline_config: Dict[str, Any],
        llm_config: Dict[str, Any],
    ) -> None:
        if AIE_MISSING:
            raise ImportError(
                "Numeric AIE components are not available (src/aie_for_numeric_retrieval was removed). "
                "Restore the module to run the XBRL numeric pipeline."
            )
        self.db_url = db_url
        self.pipeline_config = pipeline_config or {}
        self.llm_config = llm_config or {}

        logger.info("Initialising LLM interface for numeric pipeline")
        self.llm_interface = LLMInterface(self.llm_config)
        self.pipeline = AIEPipeline(self.pipeline_config, self.llm_interface)

    # ------------------------------------------------------------------
    def _prepare_targets(
        self,
        targets: Optional[Iterable[Dict[str, Any]]],
        query: str,
    ) -> List[ExtractionTarget]:
        prepared: List[ExtractionTarget] = []
        if targets:
            for cfg in targets:
                prepared.append(
                    ExtractionTarget(
                        name=cfg.get("name", "answer"),
                        description=cfg.get("description", query),
                        data_type=cfg.get("data_type", cfg.get("type", "text")),
                        required=cfg.get("required", False),
                        format_pattern=cfg.get("format_pattern"),
                        unit=cfg.get("unit"),
                    )
                )
        if not prepared:
            prepared.append(
                ExtractionTarget(
                    name="numeric_answer",
                    description=f"Numeric answer for query: {query}",
                    data_type="number",
                    required=False,
                )
            )
        return prepared

    # ------------------------------------------------------------------
    def _fetch_facts(self, request: PipelineRequest) -> List[FactRecord]:
        facts = fetch_fact_records(
            self.db_url,
            filing_id=request.filing_id,
            accession_no=request.accession_no,
            ticker=request.ticker,
            concept_like=request.concept_like,
            numeric_only=False,
            limit=request.limit,
        )
        if not facts:
            raise RuntimeError(
                f"No facts found in DB for filing_id={request.filing_id!r}. Did you ingest the filing?"
            )
        return facts

    # ------------------------------------------------------------------
    def run(
        self,
        request: PipelineRequest,
        *,
        extraction_targets: Optional[Iterable[Dict[str, Any]]] = None,
        facts: Optional[List[FactRecord]] = None,
    ) -> AIEPipelineResult:
        if facts is None:
            facts = self._fetch_facts(request)
        segments, lines = build_fact_segments(facts)
        document_text = build_document_text(lines)

        targets = self._prepare_targets(extraction_targets, request.query)

        logger.info(
            "Running numeric pipeline: filing_id=%s, segments=%d, targets=%d",
            request.filing_id,
            len(segments),
            len(targets),
        )

        result = self.pipeline.process_document(
            document_text=document_text,
            query=request.query,
            extraction_targets=targets,
            document_id=request.filing_id,
            injected_segments=segments,
        )

        result.metadata.setdefault("facts_count", len(segments))
        result.metadata.setdefault("source", "xbrl_db")
        return result


__all__ = ["XbrlNumericPipeline", "PipelineRequest"]
