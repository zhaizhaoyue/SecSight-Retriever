from __future__ import annotations

"""
Document Summarization Module (Enhanced)

Drop-in replacement for your existing summarizer with:
- AUTO strategy selection (stuff / map_reduce / refine) based on token budget
- Precise prompt budgeting (accounts for instruction tokens)
- Soft splitting for overlong segments (sentence-based) to prevent overflow
- Extractive-first Map mode with inline citations (JSON bullets with segment ids)
- Table/Header aware prompting branches
- Optional parallel Map summarization and lightweight disk/in-memory caching
- Output format controls (plain / bullets / json) and length constraints
- Simple numeric consistency post-verification (flags likely-mismatched numbers)
- Hierarchical reduce when many map summaries

Compatibility: keeps SummaryResult, SummarizeStrategy, DocumentSummarizer APIs.
"""

import concurrent.futures as _cf
import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .segmentation import DocumentSegment
from .retrieval import RetrievalResult
from ..models.llm_interface import LLMInterface

logger = logging.getLogger(__name__)

__all__ = [
    "SummaryResult",
    "BaseSummarizer",
    "StuffSummarizer",
    "MapReduceSummarizer",
    "RefineSummarizer",
    "DocumentSummarizer",
    "SummarizeStrategy",
]

# =========================
# Data Models
# =========================

@dataclass(frozen=True)
class SummaryResult:
    """Summary result data class.

    Attributes
    ----------
    summary : str
        Final summary text.
    source_segments : List[DocumentSegment]
        Source segments that participated in summarization (in input order).
    metadata : Dict[str, Any]
        Additional metadata: strategy, token_count, map_count, iterations, citations, etc.
    """
    summary: str
    source_segments: List[DocumentSegment]
    metadata: Dict[str, Any]

    def __len__(self) -> int:
        return len(self.summary)

    def __str__(self) -> str:
        return f"Summary({len(self.summary)} chars, {len(self.source_segments)} segments)"


class SummarizeStrategy(str, Enum):
    STUFF = "stuff"
    MAP_REDUCE = "map_reduce"
    REFINE = "refine"
    AUTO = "auto"


# =========================
# Simple cache (memory + optional disk)
# =========================

class _SimpleCache:
    def __init__(self, cache_dir: Optional[str] = None, max_items: int = 2048) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._mem: Dict[str, str] = {}
        self._order: List[str] = []
        self._max = int(max_items)

    def _key(self, prompt: str) -> str:
        h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return h

    def get(self, prompt: str) -> Optional[str]:
        k = self._key(prompt)
        with self._lock:
            if k in self._mem:
                return self._mem[k]
        if self.cache_dir:
            fp = self.cache_dir / f"{k}.txt"
            if fp.exists():
                try:
                    return fp.read_text(encoding="utf-8")
                except Exception:
                    return None
        return None

    def put(self, prompt: str, value: str) -> None:
        k = self._key(prompt)
        with self._lock:
            if k not in self._mem:
                self._order.append(k)
                self._mem[k] = value
                if len(self._order) > self._max:
                    oldest = self._order.pop(0)
                    self._mem.pop(oldest, None)
        if self.cache_dir:
            try:
                fp = self.cache_dir / f"{k}.txt"
                fp.write_text(value, encoding="utf-8")
            except Exception:
                pass


# =========================
# Base Class
# =========================

class BaseSummarizer:
    """Base summarizer class with enhanced budgeting and soft-split support."""

    # Leave margin for prompt budget; final effective budget will subtract instruction tokens
    PROMPT_BUDGET_RATIO: float = 0.92

    def __init__(self, llm_interface: LLMInterface, *, language: str = "en",
                 output_format: str = "plain",  # plain | bullets | json
                 max_summary_tokens: Optional[int] = None,
                 extractive_first: bool = False,
                 parallel_map_workers: int = 1,
                 cache: Optional[_SimpleCache] = None) -> None:
        self.llm: LLMInterface = llm_interface
        self.language = language.lower()
        self.output_format = output_format
        self.max_summary_tokens = max_summary_tokens
        self.extractive_first = extractive_first
        self.parallel_map_workers = max(1, int(parallel_map_workers))
        self.cache = cache

    def summarize(
        self,
        segments: Sequence[DocumentSegment],
        query: Optional[str] = None,
    ) -> SummaryResult:
        raise NotImplementedError

    # -------- helpers --------

    def _concat_contents(self, segments: Iterable[DocumentSegment]) -> str:
        return "\n\n".join(seg.content for seg in segments)

    def _count_tokens(self, text: str) -> int:
        try:
            return int(self.llm.count_tokens(text))
        except Exception as exc:  # fallback
            logger.warning("count_tokens failed, char-approx used: %s", exc)
            return max(1, len(text) // 4)

    def _language_requirement(self) -> str:
        if self.language in {"en", "english"}: return "Respond in English."
        if self.language in {"nl", "dutch"}: return "Antwoord in het Nederlands."
        if self.language in {"zh", "zh-cn", "chinese"}: return "请用中文回答。"
        return "Respond in English."

    def _format_requirement(self) -> str:
        if self.output_format == "bullets":
            return "Return 5-8 concise bullet points."
        if self.output_format == "json":
            return "Return strict JSON with fields: {\"summary\": string, \"bullets\": [string]}. No extra text."
        return "Return well-structured paragraphs."

    def _length_requirement(self) -> str:
        if not self.max_summary_tokens:
            return "Keep it concise."
        return f"Hard limit: <= {self.max_summary_tokens} tokens."

    def _effective_budget(self, prompt_builder, content: str, query: Optional[str], max_tokens: int) -> int:
        """Compute budget for content by subtracting instruction token cost from max_tokens."""
        # Create a dummy segment for prompt estimation
        dummy_segment = DocumentSegment(
            id="dummy", content="", segment_type="text", 
            start_pos=0, end_pos=0, metadata={}
        )
        head = prompt_builder("", query, [dummy_segment])
        head_tokens = self._count_tokens(head)
        raw_budget = max(1, int(max_tokens * self.PROMPT_BUDGET_RATIO) - head_tokens - 32)
        return raw_budget

    def _soft_split_segment(self, seg: DocumentSegment, budget_tokens: int) -> List[DocumentSegment]:
        """Split an overlong segment at sentence-ish boundaries to respect budget."""
        sents = re.split(r"(?<=[.!?。！？])\s+", seg.content)
        out: List[DocumentSegment] = []
        cur: List[str] = []
        cur_t = 0
        for s in sents:
            t = self._count_tokens(s)
            if cur and cur_t + t > budget_tokens:
                out.append(DocumentSegment(
                    id=f"{seg.id}:part{len(out)}", content=" ".join(cur),
                    segment_type=seg.segment_type, start_pos=seg.start_pos, end_pos=seg.end_pos, metadata=seg.metadata
                ))
                cur, cur_t = [s], t
            else:
                cur.append(s); cur_t += t
        if cur:
            out.append(DocumentSegment(
                id=f"{seg.id}:part{len(out)}", content=" ".join(cur),
                segment_type=seg.segment_type, start_pos=seg.start_pos, end_pos=seg.end_pos, metadata=seg.metadata
            ))
        return out if out else [seg]

    def _token_chunk(self, segments: Sequence[DocumentSegment], max_tokens: int,
                      prompt_builder=None, query: Optional[str] = None) -> List[List[DocumentSegment]]:
        if not segments:
            return []
        chunks: List[List[DocumentSegment]] = []
        cur: List[DocumentSegment] = []
        cur_tokens = 0
        budget = max_tokens
        if prompt_builder is not None:
            budget = self._effective_budget(prompt_builder, "", query, max_tokens)

        for seg in segments:
            seg_tokens = self._count_tokens(seg.content)
            # if single segment too large, soft split it first
            segs_to_add = [seg]
            if seg_tokens > budget:
                segs_to_add = self._soft_split_segment(seg, budget)
            for piece in segs_to_add:
                t = self._count_tokens(piece.content)
                if cur and cur_tokens + t > budget:
                    chunks.append(cur)
                    cur, cur_tokens = [piece], t
                else:
                    cur.append(piece)
                    cur_tokens += t
        if cur:
            chunks.append(cur)
        return chunks

    # -------- table/header aware prompt helpers --------

    def _compose_content_hint(self, segments: Sequence[DocumentSegment]) -> str:
        types = {s.segment_type for s in segments}
        hints = []
        if "table" in types:
            hints.append("Some fragments are tables. Extract metrics with units and time columns correctly.")
        if "header" in types:
            hints.append("Headers indicate topical structure; use them to organize the summary.")
        return " ".join(hints)

    # -------- prompts --------

    def _prompt_stuff(self, content: str, query: Optional[str]) -> str:
        extra = self._compose_content_hint([DocumentSegment(id="_", content=content, segment_type="text", start_pos=0, end_pos=0, metadata={})])
        p = (
            "You are a careful analyst. Summarize the following content.\n\n"
            f"Content:\n{content}\n\n"
            "Requirements:\n"
            "- Preserve important numbers and units exactly as written.\n"
            "- No new facts. Cite inline with [SEGID] if provided.\n"
            f"- {self._language_requirement()}\n"
            f"- {self._format_requirement()}\n"
            f"- {self._length_requirement()}\n"
        )
        if query:
            p += f"- Focus on: {query}\n"
        if extra:
            p += f"- {extra}\n"
        return p

    def _prompt_map_extractive(self, content: str, seg_ids: Sequence[str], query: Optional[str]) -> str:
        ids_line = ", ".join(seg_ids)
        p = (
            "Extract key facts verbatim (no paraphrasing) as bullets from the fragment, with citations.\n\n"
            f"Fragment (segments: {ids_line}):\n{content}\n\n"
            "Return JSON strictly as:\n"
            "{\"bullets\": [{\"text\": string, \"source_segment_id\": string}]}\n"
            "Rules:\n"
            "- Copy sentences/phrases exactly from the fragment (no rewording).\n"
            "- Each bullet must include source_segment_id (one of the provided).\n"
            "- Preserve numbers and units.\n"
            f"- {self._language_requirement()}\n"
        )
        if query:
            p += f"- Prioritize content about: {query}\n"
        return p

    def _prompt_map_abstractive(self, content: str, query: Optional[str], segments: Sequence[DocumentSegment]) -> str:
        hint = self._compose_content_hint(segments)
        p = (
            "Summarize the fragment into 3-6 compact bullets capturing key facts.\n\n"
            f"Fragment:\n{content}\n\n"
            "Rules:\n"
            "- Keep numbers/units. No new facts.\n"
            f"- {self._language_requirement()}\n"
        )
        if query:
            p += f"- Emphasize: {query}\n"
        if hint:
            p += f"- {hint}\n"
        return p

    def _prompt_reduce(self, summaries: Sequence[str], query: Optional[str]) -> str:
        combined = "\n\n".join(f"Summary {i+1}:\n{txt}" for i, txt in enumerate(summaries))
        p = (
            "Merge the following partial summaries into one coherent final summary.\n\n"
            f"{combined}\n\n"
            "Rules:\n"
            "- Integrate all non-duplicate facts; keep numbers/units.\n"
            "- No new facts; if conflicts, prefer majority or mark as uncertain.\n"
            f"- {self._language_requirement()}\n"
            f"- {self._format_requirement()}\n"
            f"- {self._length_requirement()}\n"
        )
        if query:
            p += f"- Highlight: {query}\n"
        return p

    def _prompt_initial(self, content: str, query: Optional[str], segments: Sequence[DocumentSegment]) -> str:
        hint = self._compose_content_hint(segments)
        p = (
            "Create an initial summary of the content.\n\n"
            f"Content:\n{content}\n\n"
            "Rules:\n"
            "- Extract key information; preserve numbers/units.\n"
            "- No new facts.\n"
            f"- {self._language_requirement()}\n"
            f"- {self._format_requirement()}\n"
            f"- {self._length_requirement()}\n"
        )
        if query:
            p += f"- Focus: {query}\n"
        if hint:
            p += f"- {hint}\n"
        return p

    def _prompt_refine(self, existing: str, new_content: str, query: Optional[str]) -> str:
        p = (
            "Refine the existing summary by integrating the new content.\n\n"
            f"Existing Summary:\n{existing}\n\n"
            f"New Content:\n{new_content}\n\n"
            "Rules:\n"
            "- Keep existing correct facts; add new key info; remove duplicates.\n"
            "- Preserve numbers/units; no new facts.\n"
            f"- {self._language_requirement()}\n"
            f"- {self._format_requirement()}\n"
            f"- {self._length_requirement()}\n"
        )
        if query:
            p += f"- Emphasize: {query}\n"
        return p

    # -------- numeric verify --------

    _NUM_PAT = re.compile(r"[\$€¥£]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?%?")

    def _numeric_verify(self, summary: str, sources: Sequence[DocumentSegment]) -> Dict[str, Any]:
        nums = [n for n in self._NUM_PAT.findall(summary) if n.strip()]
        if not nums:
            return {"numbers_found": 0, "mismatches": []}
        src_text = "\n".join(s.content for s in sources)
        mismatches = []
        for n in nums:
            if n not in src_text:
                mismatches.append(n)
        return {"numbers_found": len(nums), "mismatches": mismatches}

    # -------- cache wrapper --------

    def _gen_with_cache(self, prompt: str) -> str:
        if self.cache:
            cached = self.cache.get(prompt)
            if cached is not None:
                return cached
        out = self.llm.generate_with_retry(prompt)
        if self.cache:
            self.cache.put(prompt, out)
        return out


# =========================
# Strategies
# =========================

class StuffSummarizer(BaseSummarizer):
    def __init__(self, llm_interface: LLMInterface, max_tokens: int = 3000, **kw: Any) -> None:
        super().__init__(llm_interface, **kw)
        self.max_tokens = max_tokens

    def summarize(self, segments: Sequence[DocumentSegment], query: Optional[str] = None) -> SummaryResult:
        if not segments:
            return SummaryResult("", [], {"strategy": SummarizeStrategy.STUFF.value})
        combined = self._concat_contents(segments)
        budget = self._effective_budget(self._prompt_stuff, combined, query, self.max_tokens)
        # Truncate softly by adding segments until budget
        parts: List[str] = []
        cur = 0
        for seg in segments:
            t = self._count_tokens(seg.content)
            if cur + t > budget:
                break
            parts.append(seg.content); cur += t
        content = "\n\n".join(parts)
        prompt = self._prompt_stuff(content, query)
        summary = self._gen_with_cache(prompt)
        meta = {
            "strategy": SummarizeStrategy.STUFF.value,
            "token_count": self._count_tokens(summary),
            "input_token_est": cur,
            "language": self.language,
            "format": self.output_format,
        }
        meta["num_verify"] = self._numeric_verify(summary, list(segments))
        return SummaryResult(summary=summary, source_segments=list(segments), metadata=meta)


class MapReduceSummarizer(BaseSummarizer):
    def __init__(self, llm_interface: LLMInterface, chunk_size: int = 2000, **kw: Any) -> None:
        super().__init__(llm_interface, **kw)
        self.chunk_size = chunk_size

    def _map_one(self, chunk: Sequence[DocumentSegment], query: Optional[str]) -> Dict[str, Any]:
        content = self._concat_contents(chunk)
        seg_ids = [s.id for s in chunk]
        if self.extractive_first:
            prompt = self._prompt_map_extractive(content, seg_ids, query)
            raw = self._gen_with_cache(prompt)
            bullets: List[Dict[str, str]] = []
            try:
                # allow code-fenced JSON
                m = re.search(r"```json\s*(.*?)```", raw, re.S)
                if m:
                    raw = m.group(1)
                obj = json.loads(raw)
                bullets = obj.get("bullets", []) if isinstance(obj, dict) else []
            except Exception:
                # fallback: produce a single bullet with whole content
                bullets = [{"text": content[:400], "source_segment_id": seg_ids[0] if seg_ids else ""}]
            return {"type": "extractive", "bullets": bullets}
        else:
            prompt = self._prompt_map_abstractive(content, query, chunk)
            text = self._gen_with_cache(prompt)
            return {"type": "abstractive", "text": text, "seg_ids": seg_ids}

    def summarize(self, segments: Sequence[DocumentSegment], query: Optional[str] = None) -> SummaryResult:
        if not segments:
            return SummaryResult("", [], {"strategy": SummarizeStrategy.MAP_REDUCE.value})
        chunks = self._token_chunk(segments, self.chunk_size, prompt_builder=self._prompt_map_abstractive, query=query)
        logger.info("Map phase: %d chunks", len(chunks))

        results: List[Dict[str, Any]] = []
        if self.parallel_map_workers > 1 and len(chunks) > 1:
            with _cf.ThreadPoolExecutor(max_workers=self.parallel_map_workers) as ex:
                futs = [ex.submit(self._map_one, ch, query) for ch in chunks]
                for f in futs:
                    try:
                        results.append(f.result())
                    except Exception as exc:
                        logger.warning("Map chunk failed: %s", exc)
                        results.append({"type": "abstractive", "text": ""})
        else:
            for ch in chunks:
                try:
                    results.append(self._map_one(ch, query))
                except Exception as exc:
                    logger.warning("Map chunk failed: %s", exc)
                    results.append({"type": "abstractive", "text": ""})

        map_texts: List[str] = []
        citations: List[Dict[str, Any]] = []
        if self.extractive_first:
            # Convert extractive bullets -> one concatenated list with citations
            bullets_all: List[str] = []
            for r in results:
                if r.get("type") == "extractive":
                    for b in r.get("bullets", []) or []:
                        txt = str(b.get("text") or "").strip()
                        sid = str(b.get("source_segment_id") or "").strip()
                        if not txt:
                            continue
                        bullets_all.append(f"- {txt} [{sid}]")
                        citations.append({"text": txt, "source_segment_id": sid})
                elif r.get("type") == "abstractive":
                    if r.get("text"):
                        map_texts.append(r["text"])  # mixed mode tolerance
            # Merge bullets into synthetic map summary to feed reduce
            if bullets_all:
                map_texts.append("\n".join(bullets_all))
        else:
            for r in results:
                if r.get("type") == "abstractive" and r.get("text"):
                    map_texts.append(r["text"]) 

        # Hierarchical reduce if many
        def _reduce_batch(items: List[str]) -> str:
            reduce_prompt = self._prompt_reduce(items, query)
            return self._gen_with_cache(reduce_prompt)

        final_summary: str
        if not map_texts:
            final_summary = ""
        elif len(map_texts) == 1:
            final_summary = map_texts[0]
        elif len(map_texts) <= 8:
            final_summary = _reduce_batch(map_texts)
        else:
            # chunk map_texts into groups of 6, reduce each, then final reduce
            groups = [map_texts[i:i+6] for i in range(0, len(map_texts), 6)]
            mids = [_reduce_batch(g) for g in groups]
            final_summary = _reduce_batch(mids)

        meta = {
            "strategy": SummarizeStrategy.MAP_REDUCE.value,
            "map_count": len(results),
            "chunk_count": len(chunks),
            "token_count": self._count_tokens(final_summary),
            "language": self.language,
            "format": self.output_format,
        }
        if citations:
            meta["citations"] = citations
        meta["num_verify"] = self._numeric_verify(final_summary, list(segments))
        return SummaryResult(summary=final_summary, source_segments=list(segments), metadata=meta)


class RefineSummarizer(BaseSummarizer):
    def __init__(self, llm_interface: LLMInterface, chunk_size: int = 2000, **kw: Any) -> None:
        super().__init__(llm_interface, **kw)
        self.chunk_size = chunk_size

    def summarize(self, segments: Sequence[DocumentSegment], query: Optional[str] = None) -> SummaryResult:
        if not segments:
            return SummaryResult("", [], {"strategy": SummarizeStrategy.REFINE.value, "iterations": 0})
        chunks = self._token_chunk(segments, self.chunk_size, prompt_builder=self._prompt_initial, query=query)
        logger.info("Refine strategy: %d chunks", len(chunks))

        first_content = self._concat_contents(chunks[0])
        prompt0 = self._prompt_initial(first_content, query, chunks[0])
        current = self._gen_with_cache(prompt0)

        for i, chunk in enumerate(chunks[1:], start=1):
            content = self._concat_contents(chunk)
            refine_prompt = self._prompt_refine(current, content, query)
            try:
                current = self._gen_with_cache(refine_prompt)
            except Exception as exc:
                logger.warning("Refinement %d failed: %s; keep current", i, exc)
                continue

        meta = {
            "strategy": SummarizeStrategy.REFINE.value,
            "iterations": len(chunks),
            "chunk_count": len(chunks),
            "token_count": self._count_tokens(current),
            "language": self.language,
            "format": self.output_format,
        }
        meta["num_verify"] = self._numeric_verify(current, list(segments))
        return SummaryResult(summary=current, source_segments=list(segments), metadata=meta)


# =========================
# Facade
# =========================

class DocumentSummarizer:
    """Main document summarizer class (enhanced)."""

    def __init__(self, config: Dict[str, Any], llm_interface: LLMInterface):
        """
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration options, supports:
              - strategy: "stuff" | "map_reduce" | "refine" | "auto" (default "auto")
              - max_tokens: int (Stuff input cap)
              - chunk_size: int (Map-Reduce / Refine)
              - language: "en" | "nl" | "zh"
              - output_format: "plain" | "bullets" | "json"
              - max_summary_tokens: int (hard cap for outputs)
              - extractive_first: bool (Map extractive JSON bullets with citations)
              - parallel_map_workers: int
              - cache_dir: str (optional disk cache directory)
              - cache_max_items: int (memory cache size)
        llm_interface : LLMInterface
            LLM interface instance.
        """
        self.config: Dict[str, Any] = config
        self.llm: LLMInterface = llm_interface
        self.summarizer: BaseSummarizer = self._build_summarizer()

    # -------- public API --------

    def summarize_segments(
        self,
        segments: Sequence[DocumentSegment],
        query: Optional[str] = None,
    ) -> SummaryResult:
        chosen = self.config.get("strategy", SummarizeStrategy.AUTO.value)
        logger.info("Starting summarization, strategy: %s; segment count: %d", chosen, len(segments))
        result = self.summarizer.summarize(segments, query)
        logger.info("Summarization completed, length: %d characters.", len(result.summary))
        return result

    def summarize_retrieval_results(
        self,
        retrieval_results: Sequence[RetrievalResult],
        query: Optional[str] = None,
    ) -> SummaryResult:
        segments = [r.segment for r in retrieval_results]
        return self.summarize_segments(segments, query)

    def batch_summarize(
        self,
        segment_groups: Sequence[Sequence[DocumentSegment]],
        queries: Optional[Sequence[Optional[str]]] = None,
    ) -> List[SummaryResult]:
        queries = list(queries) if queries is not None else [None] * len(segment_groups)
        results: List[SummaryResult] = []
        for segs, q in zip(segment_groups, queries):
            try:
                results.append(self.summarize_segments(segs, q))
            except Exception as exc:
                logger.error("Batch summarization failed: %s", exc)
                results.append(SummaryResult("", list(segs), {"error": str(exc)}))
        return results

    @staticmethod
    def get_summary_statistics(summary_result: SummaryResult) -> Dict[str, Any]:
        total_src_len = sum(len(seg.content) for seg in summary_result.source_segments) or 1
        return {
            "summary_length": len(summary_result.summary),
            "source_segments_count": len(summary_result.source_segments),
            "total_source_length": total_src_len,
            "compression_ratio": len(summary_result.summary) / total_src_len,
            "metadata": summary_result.metadata,
        }

    # -------- internals --------

    def _auto_strategy(self, total_tokens: int) -> SummarizeStrategy:
        # Heuristics (tune for your model/context window)
        if total_tokens <= 3500:
            return SummarizeStrategy.STUFF
        if total_tokens <= 20000:
            return SummarizeStrategy.MAP_REDUCE
        return SummarizeStrategy.REFINE

    def _build_summarizer(self) -> BaseSummarizer:
        raw = str(self.config.get("strategy", SummarizeStrategy.AUTO.value)).lower()
        language = str(self.config.get("language", "en")).lower()
        output_format = str(self.config.get("output_format", "plain")).lower()
        max_summary_tokens = self.config.get("max_summary_tokens")
        extractive_first = bool(self.config.get("extractive_first", False))
        parallel_map_workers = int(self.config.get("parallel_map_workers", 1))

        cache_dir = self.config.get("cache_dir")
        cache_max = int(self.config.get("cache_max_items", 2048))
        cache = _SimpleCache(cache_dir, cache_max) if (cache_dir or cache_max) else None

        # If AUTO, pick based on token count of concatenated content estimate
        if raw == SummarizeStrategy.AUTO.value:
            # build a temporary Stuff to count tokens quickly
            tmp = StuffSummarizer(self.llm, max_tokens=int(self.config.get("max_tokens", 3000)),
                                  language=language, output_format=output_format,
                                  max_summary_tokens=max_summary_tokens,
                                  extractive_first=extractive_first,
                                  parallel_map_workers=parallel_map_workers, cache=cache)
            # Provide a callable that wraps selection at runtime
            class _AutoWrapper(BaseSummarizer):
                def __init__(self, outer: 'DocumentSummarizer'):
                    super().__init__(outer.llm, language=language, output_format=output_format,
                                     max_summary_tokens=max_summary_tokens,
                                     extractive_first=extractive_first,
                                     parallel_map_workers=parallel_map_workers, cache=cache)
                    self.outer = outer
                    self.tmp = tmp

                def summarize(self, segments: Sequence[DocumentSegment], query: Optional[str] = None) -> SummaryResult:
                    total_tokens = sum(self._count_tokens(s.content) for s in segments)
                    pick = self.outer._auto_strategy(total_tokens)
                    logger.info("AUTO selected strategy=%s (total_tokens=%d)", pick.value, total_tokens)
                    # delegate to a concrete summarizer
                    conf = dict(self.outer.config)
                    conf["strategy"] = pick.value
                    self.outer.config = conf
                    self.outer.summarizer = self.outer._build_summarizer()
                    return self.outer.summarizer.summarize(segments, query)

            return _AutoWrapper(self)

        # Concrete strategies
        if raw == SummarizeStrategy.STUFF.value:
            return StuffSummarizer(
                self.llm, max_tokens=int(self.config.get("max_tokens", 3000)),
                language=language, output_format=output_format,
                max_summary_tokens=max_summary_tokens,
                extractive_first=extractive_first,
                parallel_map_workers=parallel_map_workers,
                cache=cache,
            )
        if raw == SummarizeStrategy.MAP_REDUCE.value:
            return MapReduceSummarizer(
                self.llm, chunk_size=int(self.config.get("chunk_size", 2000)),
                language=language, output_format=output_format,
                max_summary_tokens=max_summary_tokens,
                extractive_first=extractive_first,
                parallel_map_workers=parallel_map_workers,
                cache=cache,
            )
        if raw == SummarizeStrategy.REFINE.value:
            return RefineSummarizer(
                self.llm, chunk_size=int(self.config.get("chunk_size", 2000)),
                language=language, output_format=output_format,
                max_summary_tokens=max_summary_tokens,
                extractive_first=extractive_first,
                parallel_map_workers=parallel_map_workers,
                cache=cache,
            )
        raise ValueError(f"Unsupported summarization strategy: {raw}")
