# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Information Extraction Module - 改进版

新增：
- precision_mode: "direct" | "shot"（数值精度增强）
- keyword_hints / fewshot_example（单例 few-shot + 关键词补全）
- 严格 JSON 解析兜底
- 数值/单位规范化 + 溯源校核，不改写原值，另给 normalized_value
- Hybrid 一致性融合与置信度重估
- RETA 指标函数（金融容错评测）

与原版兼容：
- 保留原 API：InformationExtractor / LLMExtractor / RegexExtractor / HybridExtractor
- ExtractionTarget/ExtractionResult 保持向后兼容（仅新增可选字段）
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Iterable

from .summarization import SummaryResult
from ..models.llm_interface import LLMInterface

logger = logging.getLogger(__name__)

# =========================
# 数据模型
# =========================

@dataclass
class ExtractionTarget:
    """Extraction target definition"""
    name: str
    description: str
    data_type: str  # "number", "text", "date", "boolean", "list", "integer", ...
    required: bool = False
    format_pattern: Optional[str] = None  # Regular expression pattern
    unit: Optional[str] = None  # Numerical unit

    def __str__(self) -> str:
        return f"ExtractionTarget({self.name}, {self.data_type})"


@dataclass
class ExtractionResult:
    """Extraction result data class"""
    target: ExtractionTarget
    value: Union[str, float, int, bool, List, None]
    confidence: float
    source_text: str
    metadata: Dict[str, Any]
    # ---- 新增（可选） ----
    normalized_value: Optional[float] = None    # 解析后的数值（不替代主 value）
    source_span: Optional[Dict[str, int]] = None  # {"start": int, "end": int}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        out = {
            "target_name": self.target.name,
            "value": self.value,
            "confidence": self.confidence,
            "source_text": self.source_text,
            "data_type": self.target.data_type,
            "unit": self.target.unit,
            "metadata": self.metadata,
        }
        if self.normalized_value is not None:
            out["normalized_value"] = self.normalized_value
        if self.source_span is not None:
            out["source_span"] = self.source_span
        return out


# =========================
# 基类
# =========================

class BaseExtractor:
    def extract(self, text: str, targets: List[ExtractionTarget]) -> List[ExtractionResult]:
        raise NotImplementedError


# =========================
# Regex 提取器
# =========================

class RegexExtractor(BaseExtractor):
    """Regex-based extractor（轻量、可并用作一致性校核）"""

    def __init__(self):
        self.patterns = {
            "number": r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?(?:%| percent)?",
            "percentage": r"\d+(?:\.\d+)?%",
            "currency": r"[$¥€£]?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?",
            "date": r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",
            "year": r"\b\d{4}\b",
            "text": r"[^\n]{3,}",
        }

    def _extract_all(self, text: str, pattern: str) -> List[str]:
        try:
            m = re.findall(pattern, text, flags=re.IGNORECASE)
            out = [s if isinstance(s, str) else "".join(s) for s in m]
            return [s.strip() for s in out if str(s).strip()]
        except Exception:
            return []

    def extract(self, text: str, targets: List[ExtractionTarget]) -> List[ExtractionResult]:
        results: List[ExtractionResult] = []
        for t in targets:
            pat = t.format_pattern or self.patterns.get(t.data_type) or self.patterns["text"]
            vals = self._extract_all(text, pat)
            value = vals[0] if vals else None
            conf = 1.0 if value else 0.0
            results.append(
                ExtractionResult(
                    target=t,
                    value=value,
                    confidence=conf,
                    source_text=value or "",
                    metadata={"method": "regex", "pattern": pat},
                )
            )
        return results


# =========================
# LLM 提取器（精度增强 + 关键词提示 + 严格 JSON）
# =========================

_NUM = re.compile(r"[\$€¥£]?\(?-?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?%?")

def _precision_rules(mode: str) -> str:
    if str(mode).lower() == "shot":
        return (
            "Example (do NOT convert units):\n"
            'Text: "Total net sales were $3.50 billion in 2022."\n'
            'Return: {"extractions":[{"target_name":"revenue_2022","value":"$3.50 billion","confidence":0.98,"source_sentence":"Total net sales were $3.50 billion in 2022."}]}\n'
            "Rules: keep original unit/precision; if context says 'in millions', keep that wording in source_sentence and DO NOT scale.\n"
        )
    # direct
    return (
        "- Copy values EXACTLY as in text, including signs, commas, parentheses, and units/currencies.\n"
        "- DO NOT round or convert units. DO NOT infer missing units.\n"
    )

def _strict_schema_block() -> str:
    return (
        "You MUST return STRICT JSON only, no extra text:\n"
        '{ "extractions": [\n'
        '  { "target_name": string, "value": string|null, "confidence": number, "source_sentence": string }\n'
        "]}\n"
        "If not found: value=null, confidence=0, source_sentence=\"\".\n"
        "Confidence range: [0,1]. Temperature=0 assumptions; rely on literals from text.\n"
    )

def _json_safely_extract(raw: str) -> Optional[Dict[str, Any]]:
    """多层兜底解析 ```json ...``` 或最外层 {...}"""
    try:
        m = re.search(r"```json\s*(.*?)```", raw, flags=re.S | re.I)
        if m:
            return json.loads(m.group(1).strip())
        m = re.search(r"\{.*\}\s*$", raw.strip(), flags=re.S)
        if m:
            return json.loads(m.group(0))
        return json.loads(raw)
    except Exception as e:
        logger.warning("JSON parse failed once: %s", e)
        return None

def _json_to_results(obj: Dict[str, Any], targets: List[ExtractionTarget]) -> List[ExtractionResult]:
    tmap = {t.name: t for t in targets}
    out: List[ExtractionResult] = []
    arr = obj.get("extractions", []) if isinstance(obj, dict) else []
    for item in arr:
        try:
            name = str(item.get("target_name", "")).strip()
            if name not in tmap:
                # 忽略未知 target
                continue
            t = tmap[name]
            val = item.get("value", None)
            st = str(item.get("source_sentence", "") or "")
            conf = float(item.get("confidence", 0.0) or 0.0)
            # 保持 value 原文字符串；类型转换仅做在 normalized_value
            res = ExtractionResult(
                target=t,
                value=val,
                confidence=conf,
                source_text=st,
                metadata={"method": "llm"},
            )
            out.append(res)
        except Exception as e:
            logger.warning("skip bad item: %s", e)
            continue

    # 未覆盖的目标补空
    covered = {r.target.name for r in out}
    for t in targets:
        if t.name not in covered:
            out.append(ExtractionResult(
                target=t, value=None, confidence=0.0, source_text="", metadata={"method": "llm", "status": "not_found"}
            ))
    return out

def _normalize_number_str(s: str, context: str) -> Optional[float]:
    """保守规范化：仅作为 normalized_value，不改写主 value."""
    try:
        raw = s.strip()
        neg = raw.startswith("-") or (raw.startswith("(") and raw.endswith(")"))
        clean = raw.replace("(", "").replace(")", "").strip()
        unit_mul = 1.0
        # 上下文提示优先
        if re.search(r"\bin\s+millions\b", context, re.I):
            unit_mul = 1e6
        if re.search(r"\bin\s+billions\b", context, re.I):
            unit_mul = 1e9
        # 百分比
        if clean.endswith("%"):
            v = float(clean.rstrip("%").replace(",", ""))
            return (-v if neg else v) / 100.0
        # 货币符号
        v = float(re.sub(r"^[\$€¥£]", "", clean).replace(",", ""))
        # 文本中的规模词（若未由 context 提示）
        if re.search(r"\bbn|billion\b", clean, re.I):
            unit_mul = max(unit_mul, 1e9)
        if re.search(r"\bmn|million\b", clean, re.I):
            unit_mul = max(unit_mul, 1e6)
        if re.search(r"\bthousand|k\b", clean, re.I):
            unit_mul = max(unit_mul, 1e3)
        out = v * unit_mul
        return -out if neg else out
    except Exception:
        return None

def _post_check_and_score(text: str, results: List[ExtractionResult]) -> List[ExtractionResult]:
    """字符级溯源 + 规范化 + 置信度重估"""
    for r in results:
        val_str = r.value if isinstance(r.value, str) else None
        if not val_str:
            continue

        # 逐字匹配
        idx = text.find(val_str)
        matched = idx >= 0
        if matched:
            if not r.source_text:
                window = text[max(0, idx - 120): idx + len(val_str) + 120]
                r.source_text = window.strip()
            r.source_span = {"start": idx, "end": idx + len(val_str)}

        # 规范化
        r.normalized_value = _normalize_number_str(val_str, r.source_text or "")

        # 置信度融合
        c = float(r.confidence or 0.0)
        if matched:
            c += 0.40
        if r.normalized_value is not None:
            c += 0.20
        r.confidence = max(0.0, min(1.0, c))
    return results


class LLMExtractor(BaseExtractor):
    """基于 LLM 的提取器（数值精度增强 + 严格 JSON）"""

    def __init__(self, llm_interface: LLMInterface):
        self.llm = llm_interface

    # ---- Prompt 生成 ----
    def _create_extraction_prompt(
        self,
        text: str,
        targets: List[ExtractionTarget],
        *,
        precision_mode: str = "direct",
        keyword_hints: Optional[List[str]] = None,
        fewshot_example: Optional[str] = None,
    ) -> str:
        hints = ""
        if keyword_hints:
            hints = "Keyword hints: " + "; ".join([str(h) for h in keyword_hints[:8]]) + "\n"

        targets_json = json.dumps(
            [
                {
                    "name": t.name,
                    "type": t.data_type,
                    "unit": t.unit,
                    "required": t.required,
                    "pattern": t.format_pattern or "",
                }
                for t in targets
            ],
            ensure_ascii=False,
        )

        return (
            "Extract the requested fields from the text.\n"
            + _precision_rules(precision_mode)
            + (fewshot_example or "")
            + hints
            + "Text:\n"
            + text
            + "\n\nTargets (name/type/unit/required):\n"
            + targets_json
            + "\n\n"
            + _strict_schema_block()
        )

    # ---- 主流程 ----
    def extract(
        self,
        text: str,
        targets: List[ExtractionTarget],
        *,
        precision_mode: str = "direct",
        keyword_hints: Optional[List[str]] = None,
        fewshot_example: Optional[str] = None,
    ) -> List[ExtractionResult]:
        prompt = self._create_extraction_prompt(
            text, targets,
            precision_mode=precision_mode,
            keyword_hints=keyword_hints,
            fewshot_example=fewshot_example,
        )
        try:
            raw = self.llm.generate_with_retry(prompt, max_retries=3)
            obj = _json_safely_extract(raw) or {"extractions": []}
            results = _json_to_results(obj, targets)
            return _post_check_and_score(text, results)
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return [
                ExtractionResult(
                    target=t, value=None, confidence=0.0, source_text="",
                    metadata={"method": "llm", "error": str(e)}
                )
                for t in targets
            ]


# =========================
# Hybrid 提取器（一致性融合）
# =========================

class HybridExtractor(BaseExtractor):
    """Hybrid extractor combining regex and LLM with consistency boosting"""

    def __init__(self, llm_interface: LLMInterface, regex_first: bool = True):
        self.regex = RegexExtractor()
        self.llm = LLMExtractor(llm_interface)
        self.regex_first = regex_first

    def extract(
        self,
        text: str,
        targets: List[ExtractionTarget],
        *,
        precision_mode: str = "direct",
        keyword_hints: Optional[List[str]] = None,
        fewshot_example: Optional[str] = None,
    ) -> List[ExtractionResult]:
        if self.regex_first:
            r1 = self.regex.extract(text, targets)
            # 对失败项再走 LLM
            failed = [x.target for x in r1 if not x.value or x.confidence < 0.5]
            r2_map: Dict[str, ExtractionResult] = {}
            if failed:
                r2 = self.llm.extract(
                    text, failed,
                    precision_mode=precision_mode,
                    keyword_hints=keyword_hints,
                    fewshot_example=fewshot_example,
                )
                r2_map = {x.target.name: x for x in r2}

            fused: List[ExtractionResult] = []
            for x in r1:
                y = r2_map.get(x.target.name)
                if y is None:
                    fused.append(x)
                    continue
                # 一致性加分
                if isinstance(x.value, str) and isinstance(y.value, str) and x.value == y.value:
                    y.confidence = min(1.0, max(x.confidence, y.confidence) + 0.20)
                    fused.append(y)
                else:
                    # 优先选择“能在原文匹配”的那一个
                    x_hit = isinstance(x.value, str) and (text.find(x.value) >= 0)
                    y_hit = isinstance(y.value, str) and (text.find(y.value) >= 0)
                    if y_hit and not x_hit:
                        fused.append(y)
                    elif x_hit and not y_hit:
                        fused.append(x)
                    else:
                        # 都能/都不能：取置信度高者
                        fused.append(y if y.confidence >= x.confidence else x)
            return _post_check_and_score(text, fused)
        else:
            # LLM 优先
            r = self.llm.extract(
                text, targets,
                precision_mode=precision_mode,
                keyword_hints=keyword_hints,
                fewshot_example=fewshot_example,
            )
            return r


# =========================
# Facade
# =========================

class InformationExtractor:
    """Main information extractor class"""

    def __init__(self, config: Dict[str, Any], llm_interface: Optional[LLMInterface] = None):
        self.config = config
        self.extractor = self._initialize_extractor(llm_interface)

    def _initialize_extractor(self, llm_interface: Optional[LLMInterface]) -> BaseExtractor:
        method = str(self.config.get("extraction_method", "llm")).lower()
        if method == "regex":
            return RegexExtractor()
        if method == "llm":
            if llm_interface is None:
                raise ValueError("LLM 提取需要提供 LLM 接口")
            return LLMExtractor(llm_interface)
        if method == "hybrid":
            if llm_interface is None:
                raise ValueError("混合提取需要提供 LLM 接口")
            regex_first = bool(self.config.get("regex_first", True))
            return HybridExtractor(llm_interface, regex_first)
        raise ValueError(f"不支持的提取方法: {method}")

    # ---- 对外 API ----

    def extract_from_text(
        self,
        text: str,
        targets: List[ExtractionTarget],
        *,
        precision_mode: str = "direct",
        keyword_hints: Optional[List[str]] = None,
        fewshot_example: Optional[str] = None,
    ) -> List[ExtractionResult]:
        logger.info("Starting information extraction, targets=%d", len(targets))
        if isinstance(self.extractor, LLMExtractor):
            results = self.extractor.extract(
                text, targets,
                precision_mode=precision_mode,
                keyword_hints=keyword_hints,
                fewshot_example=fewshot_example,
            )
        elif isinstance(self.extractor, HybridExtractor):
            results = self.extractor.extract(
                text, targets,
                precision_mode=precision_mode,
                keyword_hints=keyword_hints,
                fewshot_example=fewshot_example,
            )
        else:
            results = self.extractor.extract(text, targets)

        ok = sum(1 for r in results if r.value is not None)
        logger.info("Extraction completed: %d/%d", ok, len(targets))
        return results

    def extract_from_summary(
        self,
        summary_result: SummaryResult,
        targets: List[ExtractionTarget],
        *,
        precision_mode: str = "direct",
        keyword_hints: Optional[List[str]] = None,
        fewshot_example: Optional[str] = None,
    ) -> List[ExtractionResult]:
        return self.extract_from_text(
            summary_result.summary, targets,
            precision_mode=precision_mode,
            keyword_hints=keyword_hints,
            fewshot_example=fewshot_example,
        )

    def batch_extract(
        self,
        texts: List[str],
        targets: List[ExtractionTarget],
        *,
        precision_mode: str = "direct",
        keyword_hints_list: Optional[List[Optional[List[str]]]] = None,
        fewshot_example: Optional[str] = None,
    ) -> List[List[ExtractionResult]]:
        results_all: List[List[ExtractionResult]] = []
        for i, text in enumerate(texts):
            hints = (keyword_hints_list[i] if keyword_hints_list and i < len(keyword_hints_list) else None)
            try:
                res = self.extract_from_text(
                    text, targets,
                    precision_mode=precision_mode,
                    keyword_hints=hints,
                    fewshot_example=fewshot_example,
                )
                results_all.append(res)
            except Exception as e:
                logger.error("batch extract failed: %s", e)
                results_all.append([
                    ExtractionResult(
                        target=t, value=None, confidence=0.0, source_text="",
                        metadata={"error": str(e)}
                    )
                    for t in targets
                ])
        return results_all

    def create_extraction_targets_from_config(self, targets_config: List[Dict[str, Any]]) -> List[ExtractionTarget]:
        targets: List[ExtractionTarget] = []
        for cfg in targets_config:
            targets.append(ExtractionTarget(
                name=cfg["name"],
                description=cfg["description"],
                data_type=cfg.get("data_type", "text"),
                required=cfg.get("required", False),
                format_pattern=cfg.get("format_pattern"),
                unit=cfg.get("unit"),
            ))
        return targets

    def get_extraction_statistics(self, results: List[ExtractionResult]) -> Dict[str, Any]:
        if not results:
            return {}
        total = len(results)
        successful = sum(1 for r in results if r.value is not None)
        avg_conf = sum(float(r.confidence or 0.0) for r in results) / max(total, 1)
        by_type: Dict[str, Dict[str, int]] = {}
        for r in results:
            dt = r.target.data_type
            by_type.setdefault(dt, {"total": 0, "successful": 0})
            by_type[dt]["total"] += 1
            if r.value is not None:
                by_type[dt]["successful"] += 1
        return {
            "total_targets": total,
            "successful_extractions": successful,
            "success_rate": successful / max(total, 1),
            "average_confidence": avg_conf,
            "by_data_type": by_type,
        }


# =========================
# 评测：RETA 指标（相对误差容忍）
# =========================

def reta_accuracy(
    y_true: Iterable[Optional[float]],
    y_pred: Iterable[Optional[float]],
    tol: float = 0.03
) -> float:
    ok, n = 0, 0
    for gt, pr in zip(y_true, y_pred):
        if gt is None or pr is None:
            continue
        n += 1
        denom = max(abs(gt), 1e-12)
        if abs(pr - gt) / denom <= tol:
            ok += 1
    return ok / max(n, 1)
