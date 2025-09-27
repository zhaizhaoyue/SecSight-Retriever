# -*- coding: utf-8 -*-
"""
AIE Framework Main Pipeline (Optimized)
- 可裁剪阶段：segmentation / retrieval / summarization / extraction
- 细粒度计时与错误分类；更丰富的 metadata
- 批处理并行（可限流 LLM 并发）；线程安全
- 可选索引复用（按 document_id 缓存检索器）
- RETA 指标评测（方便与论文对齐）
- 与现有模块完全兼容：segmentation / retrieval / summarization / extraction
"""

from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
from dataclasses import dataclass, field
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .segmentation import DocumentSegmenter, DocumentSegment
from .retrieval import DocumentRetriever, RetrievalResult
from .summarization import DocumentSummarizer, SummaryResult
from .extraction import InformationExtractor, ExtractionTarget, ExtractionResult
from ..models.llm_interface import LLMInterface

logger = logging.getLogger(__name__)


# ---------------------------
# Metrics & Helpers
# ---------------------------

def _now() -> float:
    return time.perf_counter()

@dataclass
class StageTiming:
    start: float
    end: float
    ok: bool
    error: Optional[str] = None

    @property
    def elapsed(self) -> float:
        return max(0.0, self.end - self.start)

@dataclass
class StageReport:
    name: str
    timing: StageTiming
    meta: Dict[str, Any] = field(default_factory=dict)

def reta_accuracy(y_true: List[Optional[float]],
                  y_pred: List[Optional[float]],
                  tol: float = 0.03) -> float:
    """
    Relative Error Tolerance Accuracy: share of |pred-true|/|true| <= tol
    忽略 None / 非数值项。
    """
    num, den = 0, 0
    for t, p in zip(y_true, y_pred):
        if t is None or p is None:
            continue
        try:
            t = float(t); p = float(p)
            den += 1
            if (abs(p - t) / (abs(t) + 1e-12)) <= tol:
                num += 1
        except Exception:
            continue
    return (num / den) if den else 0.0


# ---------------------------
# Pipeline Result
# ---------------------------

@dataclass
class AIEPipelineResult:
    document_id: str
    query: str
    segments: List[DocumentSegment]
    retrieved_segments: List[RetrievalResult]
    summary: SummaryResult
    extractions: List[ExtractionResult]
    processing_time: float
    stage_reports: List[StageReport]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "query": self.query,
            "summary": self.summary.summary if self.summary else "",
            "extractions": [ext.to_dict() for ext in (self.extractions or [])],
            "processing_time": self.processing_time,
            "stages": [{
                "name": r.name,
                "elapsed": r.timing.elapsed,
                "ok": r.timing.ok,
                "error": r.timing.error,
                "meta": r.meta
            } for r in self.stage_reports],
            "metadata": {
                **self.metadata,
                "segments_count": len(self.segments or []),
                "retrieved_count": len(self.retrieved_segments or []),
                "summary_length": len(self.summary.summary) if self.summary else 0,
                "extraction_count": len(self.extractions or [])
            }
        }


# ---------------------------
# Main Pipeline (Optimized)
# ---------------------------

class AIEPipeline:
    """
    优化版 AIE 主流水线
    配置示例:
    {
      "stages": {"segmentation": True, "retrieval": True, "summarization": True, "extraction": True},
      "batch": {"max_workers": 4},
      "llm_max_concurrency": 2,
      "retrieval": {"cache_index": True},   # 若按 doc 复用索引
      "summarization": {... 原有 summarizer 配置 ...},
      "extraction": {... 原有 extractor 配置 ...},
      "segmentation": {... 原有 segmenter 配置 ...}
    }
    """
    def __init__(self, config: Dict[str, Any], llm_interface: LLMInterface):
        self.config = config or {}
        self.llm = llm_interface

        # Stage switches with sensible defaults
        st = self.config.get("stages", {})
        self.enable_seg = st.get("segmentation", True)
        self.enable_ret = st.get("retrieval", True)
        self.enable_sum = st.get("summarization", True)
        self.enable_ext = st.get("extraction", True)

        # LLM concurrency guard (用于批量/并发场景下限制大模型并行度)
        self._llm_sema = threading.Semaphore(int(self.config.get("llm_max_concurrency", 2)))

        # 初始化模块（注意：retriever 可按 doc 重新实例化以便索引隔离/缓存）
        self.segmenter = DocumentSegmenter(self.config.get("segmentation", {}))
        self._retrieval_cfg = dict(self.config.get("retrieval", {}))
        self.summarizer = DocumentSummarizer(self.config.get("summarization", {}), llm_interface)
        self.extractor = InformationExtractor(self.config.get("extraction", {}), llm_interface)

        # 简易索引缓存（可选）
        self._retriever_cache: Dict[str, DocumentRetriever] = {}
        self._cache_index_enabled = bool(self._retrieval_cfg.get("cache_index", False))

        logger.info("AIE Pipeline (Optimized) initialized.")

    # -------- Retrieval helper (per document) --------
    def _get_retriever(self, document_id: str) -> DocumentRetriever:
        if self._cache_index_enabled and document_id in self._retriever_cache:
            return self._retriever_cache[document_id]
        # 每个文档独立实例，避免并发污染
        retr = DocumentRetriever(self._retrieval_cfg)
        if self._cache_index_enabled:
            self._retriever_cache[document_id] = retr
        return retr

    # -------- Single Document --------
    def process_document(self,
                         document_text: str,
                         query: str,
                         extraction_targets: List[ExtractionTarget],
                         document_id: Optional[str] = None,
                         *,
                         injected_segments: Optional[List[DocumentSegment]] = None,
                         injected_retrieval: Optional[List[RetrievalResult]] = None
                         ) -> AIEPipelineResult:
        """
        - injected_segments: 若外部已做分段，可直接注入并跳过分段
        - injected_retrieval: 若外部已做检索，可直接注入并跳过检索（但需与 segments 对齐）
        """
        t0 = _now()
        document_id = document_id or f"doc_{int(time.time())}"

        stage_reports: List[StageReport] = []
        segments: List[DocumentSegment] = injected_segments or []
        retrieved_results: List[RetrievalResult] = injected_retrieval or []
        summary_result: SummaryResult = SummaryResult("", [], {})
        extraction_results: List[ExtractionResult] = []

        meta_root: Dict[str, Any] = {"config": self.config, "success": True, "errors": []}

        # Storage manager (if enabled)
        storage_manager = self.config.get("storage", {}).get("manager")
        cache_segments = self.config.get("storage", {}).get("cache_segments", False)
        cache_retrieval = self.config.get("storage", {}).get("cache_retrieval", False)

        # 1) Segmentation (with caching)
        if self.enable_seg and not injected_segments:
            s = _now()
            try:
                # Try to load from cache first
                if storage_manager and cache_segments:
                    seg_config = self.config.get("segmentation", {})
                    cached_segments = storage_manager.load_segments(document_text, seg_config)
                    if cached_segments:
                        segments = cached_segments
                        logger.debug(f"Loaded {len(segments)} segments from cache")
                    else:
                        segments = self.segmenter.segment_document(document_text)
                        storage_manager.save_segments(document_text, segments, seg_config)
                        logger.debug(f"Segmented and cached {len(segments)} segments")
                else:
                    segments = self.segmenter.segment_document(document_text)
                
                ok, err = True, None
                meta = {"segments": len(segments), "cached": bool(storage_manager and cache_segments)}
            except Exception as e:
                ok, err = False, str(e)
                meta_root["success"] = False
                meta_root["errors"].append({"stage": "segmentation", "error": err})
                segments = []
                meta = {}
                logger.exception("Segmentation failed.")
            stage_reports.append(StageReport("segmentation", StageTiming(s, _now(), ok, err), meta))

        # 2) Retrieval
        if self.enable_ret and not injected_retrieval:
            s = _now()
            try:
                if not segments:
                    raise RuntimeError("No segments to index.")
                retr = self._get_retriever(document_id)
                retr.build_index(segments)
                retrieved_results = retr.retrieve(query)
                ok, err = True, None
                meta = {"retrieved": len(retrieved_results)}
            except Exception as e:
                ok, err = False, str(e)
                meta_root["success"] = False
                meta_root["errors"].append({"stage": "retrieval", "error": err})
                retrieved_results = []
                meta = {}
                logger.exception("Retrieval failed.")
            stage_reports.append(StageReport("retrieval", StageTiming(s, _now(), ok, err), meta))

        # 3) Summarization
        if self.enable_sum:
            s = _now()
            try:
                # 限制并发 LLM（若 summarizer 内部也有缓存并发，这里只是额外保护）
                with self._llm_sema:
                    if retrieved_results:
                        summary_result = self.summarizer.summarize_retrieval_results(retrieved_results, query)
                    else:
                        # 无检索结果时仍可直接对全部 segments 做摘要（可选）
                        if segments:
                            summary_result = self.summarizer.summarize_segments(segments, query)
                        else:
                            summary_result = SummaryResult("", [], {"note": "no content to summarize"})
                ok, err = True, None
                meta = {"summary_len": len(summary_result.summary)}
            except Exception as e:
                ok, err = False, str(e)
                meta_root["success"] = False
                meta_root["errors"].append({"stage": "summarization", "error": err})
                summary_result = SummaryResult("", [], {"error": err})
                meta = {}
                logger.exception("Summarization failed.")
            stage_reports.append(StageReport("summarization", StageTiming(s, _now(), ok, err), meta))

        # 4) Extraction
        if self.enable_ext:
            s = _now()
            try:
                with self._llm_sema:
                    if summary_result and summary_result.summary:
                        extraction_results = self.extractor.extract_from_summary(summary_result, extraction_targets)
                    else:
                        # 也可回退到直接对原文/段落抽取（若想要）
                        extraction_results = self.extractor.extract_from_text(document_text, extraction_targets)
                ok, err = True, None
                succ = sum(1 for r in extraction_results if r.value is not None)
                meta = {"extractions": len(extraction_results), "success": succ}
            except Exception as e:
                ok, err = False, str(e)
                meta_root["success"] = False
                meta_root["errors"].append({"stage": "extraction", "error": err})
                extraction_results = []
                meta = {}
                logger.exception("Extraction failed.")
            stage_reports.append(StageReport("extraction", StageTiming(s, _now(), ok, err), meta))

        t1 = _now()
        return AIEPipelineResult(
            document_id=document_id,
            query=query,
            segments=segments or [],
            retrieved_segments=retrieved_results or [],
            summary=summary_result,
            extractions=extraction_results or [],
            processing_time=(t1 - t0),
            stage_reports=stage_reports,
            metadata=meta_root
        )

    # -------- Batch (Parallel) --------
    def batch_process(self,
                      documents: List[Dict[str, Any]],
                      extraction_targets: List[ExtractionTarget]) -> List[AIEPipelineResult]:
        """
        documents: [{"document_text": str, "query": str, "document_id": "...", ...}, ...]
        并发度由 config["batch"]["max_workers"] 控制。
        """
        logger.info("Batch processing start. count=%d", len(documents))
        max_workers = int(self.config.get("batch", {}).get("max_workers", 4))
        results: List[AIEPipelineResult] = []

        def _one(doc: Dict[str, Any]) -> AIEPipelineResult:
            return self.process_document(
                document_text=doc["document_text"],
                query=doc.get("query", ""),
                extraction_targets=extraction_targets,
                document_id=doc.get("document_id"),
                injected_segments=doc.get("injected_segments"),
                injected_retrieval=doc.get("injected_retrieval"),
            )

        if max_workers <= 1 or len(documents) <= 1:
            for i, d in enumerate(documents, 1):
                logger.info("Processing %d/%d (sequential)", i, len(documents))
                results.append(_one(d))
            return results

        # 并发
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut2idx = {ex.submit(_one, d): idx for idx, d in enumerate(documents)}
            for fut in as_completed(fut2idx):
                try:
                    results.append(fut.result())
                except Exception as e:
                    logger.exception("Batch task failed: %s", e)
                    idx = fut2idx[fut]
                    doc = documents[idx]
                    results.append(AIEPipelineResult(
                        document_id=doc.get("document_id", f"batch_doc_{idx}"),
                        query=doc.get("query", ""),
                        segments=[], retrieved_segments=[],
                        summary=SummaryResult("", [], {"error": str(e)}),
                        extractions=[], processing_time=0.0,
                        stage_reports=[], metadata={"success": False, "error": str(e)}
                    ))
        logger.info("Batch processing done.")
        return results

    # -------- Stats / Save / Evaluate --------
    def get_pipeline_statistics(self, results: List[AIEPipelineResult]) -> Dict[str, Any]:
        if not results:
            return {}
        succ = [r for r in results if r.metadata.get("success", False)]
        fail = [r for r in results if not r.metadata.get("success", False)]
        times = [r.processing_time for r in succ]
        return {
            "total_documents": len(results),
            "successful_documents": len(succ),
            "failed_documents": len(fail),
            "success_rate": (len(succ) / len(results)) if results else 0.0,
            "processing_time": {
                "total": sum(times) if times else 0.0,
                "average": (sum(times) / len(times)) if times else 0.0,
                "min": min(times) if times else 0.0,
                "max": max(times) if times else 0.0,
            },
            "segments_stats": {
                "avg_segments": (sum(len(r.segments) for r in succ) / len(succ)) if succ else 0.0,
                "avg_retrieved": (sum(len(r.retrieved_segments) for r in succ) / len(succ)) if succ else 0.0
            },
            "summary_stats": {
                "avg_length": (sum(len(r.summary.summary) for r in succ) / len(succ)) if succ else 0.0
            },
            "extraction_stats": {
                "avg_extractions": (sum(len(r.extractions) for r in succ) / len(succ)) if succ else 0.0,
                "avg_successful_extractions":
                    (sum(sum(1 for e in r.extractions if e.value is not None) for r in succ) / len(succ)) if succ else 0.0
            }
        }

    def save_results(self, results: List[AIEPipelineResult], output_path: str):
        import json
        from pathlib import Path
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "results": [r.to_dict() for r in results],
            "statistics": self.get_pipeline_statistics(results),
            "config": self.config
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        logger.info("Results saved to %s", out.as_posix())

    # 简易 RETA 评测（与论文常用指标对齐）
    def evaluate_reta(self,
                      results: List[AIEPipelineResult],
                      ground_truth: Dict[str, Dict[str, Optional[float]]],
                      target_names: List[str],
                      tolerances: Tuple[float, ...] = (0.01, 0.03, 0.05, 0.10)) -> Dict[str, Any]:
        """
        ground_truth: {document_id: {target_name: true_value(float or None)}}
        返回 { "RETA@1%":..., "RETA@3%":..., ... }
        """
        # 聚合预测
        pred_map: Dict[str, Dict[str, Optional[float]]] = {}
        for r in results:
            dm = pred_map.setdefault(r.document_id, {})
            for e in r.extractions:
                if e.target and e.target.name in target_names:
                    # 兼容新版/旧版 extractor：若无 normalized_value，用可转 float 的 value 尝试
                    val = None
                    try:
                        # 新版可能有 normalized_value；否则尝试 value
                        val = getattr(e, "normalized_value", None)
                        if val is None and e.value is not None:
                            val = float(str(e.value).replace(",", "").replace("%", ""))
                    except Exception:
                        val = None
                    dm[e.target.name] = val

        # 计算各阈值 RETA
        accs = {}
        for tol in tolerances:
            y_true, y_pred = [], []
            for doc_id, tmap in ground_truth.items():
                pmap = pred_map.get(doc_id, {})
                for tn in target_names:
                    y_true.append(tmap.get(tn))
                    y_pred.append(pmap.get(tn))
            accs[f"RETA@{int(tol*100)}%"] = reta_accuracy(y_true, y_pred, tol)
        return accs
