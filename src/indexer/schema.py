from __future__ import annotations
from math import isnan
from datetime import datetime, UTC
from decimal import Decimal, InvalidOperation
from typing import Optional, List, Dict, Iterable, Type, TypeVar, Union
from pathlib import Path
from uuid import UUID, uuid4
from enum import Enum
import json
from pydantic import BaseModel, Field, ConfigDict, AliasChoices, field_validator, model_validator
import sys
import argparse
import pandas as pd
import re

HTML_TAG_RE = re.compile(r"<[^>]+>")
PERCENT_NUM_RE = re.compile(r"\d(?:\s*\.\s*\d+)?\s*%")   # 数字紧邻的百分号，例如 "3.5%"、"25 %"
TEXTBLOCK_RE = re.compile(r"TextBlock$", re.I)

def _strip_html(s: Optional[str]) -> str:
    if not s:
        return ""
    # 去掉标签，仅保留可见文本；再压缩空白
    txt = HTML_TAG_RE.sub(" ", str(s))
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt



def dump_parquet(path: Path, records: Iterable):
    if pd is None:
        raise RuntimeError("pandas 未安装，无法导出 parquet。请 `pip install pandas pyarrow`")
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.model_dump() for r in records])
    df.to_parquet(path, index=False)

def dump_csv(path: Path, records: Iterable):
    if pd is None:
        raise RuntimeError("pandas 未安装，无法导出 CSV。请 `pip install pandas`")
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.model_dump() for r in records])
    df.to_csv(path, index=False, encoding="utf-8")

SchemaVersion = str
FactScalar = Union[Decimal, int, float, str, bool, None]

# -----------------------------
# 基础枚举
# -----------------------------
class FilingForm(str, Enum):
    TEN_K = "10-K"
    TEN_Q = "10-Q"
    EIGHT_K = "8-K"
    OTHER = "OTHER"

class StatementHint(str, Enum):
    INCOME = "income"
    BALANCE = "balance"
    CASHFLOW = "cashflow"
    NOTES = "notes"
    OTHER = "other"

# -----------------------------
# 通用基类（元信息） —— CHANGED: 增加 fy/fq/doc_date/linkrole
# -----------------------------
class RecordBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: SchemaVersion = "0.3.0"

    # 源头追溯
    source_path: str

    # 归一/别名
    ticker: Optional[str] = Field(default=None, validation_alias=AliasChoices('ticker', 'symbol'))
    form: Optional[FilingForm] = Field(default=None, validation_alias=AliasChoices('form', 'filing_form'))
    year: Optional[int] = Field(default=None, validation_alias=AliasChoices('year', 'fiscal_year', 'report_year'))
    accno: Optional[str] = Field(default=None, validation_alias=AliasChoices('accno', 'accession', 'acc_no'))

    # NEW: 期别/文档日期/链接角色（利于回溯与过滤）
    fy: Optional[int] = None
    fq: Optional[str] = None
    doc_date: Optional[str] = None
    linkrole: Optional[str] = None

    # 文档定位
    page_no: Optional[int] = None
    page_anchor: Optional[str] = None
    xpath: Optional[str] = None

    # 统计 / 语言
    language: str = Field(default="en")
    tokens: Optional[int] = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


    @field_validator("language")
    @classmethod
    def _norm_lang(cls, v: Optional[str]) -> Optional[str]:
        return v.lower() if isinstance(v, str) else v

    @field_validator("ticker")
    @classmethod
    def _norm_ticker(cls, v: Optional[str]) -> Optional[str]:
        return v.upper().strip() if isinstance(v, str) else v

    @field_validator("year", "fy")
    @classmethod
    def _coerce_years(cls, v):
        if isinstance(v, str) and v.isdigit():
            return int(v)
        return v

    @field_validator("form")
    @classmethod
    def _coerce_form(cls, v):
        if isinstance(v, str):
            s = v.strip().upper()
            try:
                return FilingForm(s)
            except Exception:
                return FilingForm.OTHER
        return v

# -----------------------------
# 文本块（text.jsonl / text_corpus.jsonl）
# -----------------------------
class TextChunk(RecordBase):
    id: UUID = Field(default_factory=uuid4)
    section: Optional[str] = None
    heading: Optional[str] = None
    statement_hint: Optional[StatementHint] = None
    text: str

class TextChunkInput(TextChunk):
    model_config = ConfigDict(extra="ignore")
    @model_validator(mode="before")
    def _coalesce_and_alias(cls, data: dict) -> dict:
        for k, v in list(data.items()):
            if isinstance(v, float) and isnan(v):
                data[k] = None
            elif isinstance(v, str) and (v.strip() == ""):
                data[k] = None
        if data.get("accno") is None and data.get("accession"):
            data["accno"] = data.get("accession")
        sh = data.get("statement_hint")
        if isinstance(sh, str):
            s = sh.strip().lower()
            mapping = {
                "income": StatementHint.INCOME,
                "balance": StatementHint.BALANCE,
                "cashflow": StatementHint.CASHFLOW,
                "cash flow": StatementHint.CASHFLOW,
                "notes": StatementHint.NOTES,
                "other": StatementHint.OTHER,
            }
            data["statement_hint"] = mapping.get(s, data.get("statement_hint"))
        return data

# -----------------------------
# XBRL 上下文
# -----------------------------
class XbrlPeriod(BaseModel):
    model_config = ConfigDict(extra="ignore")
    start_date: Optional[str] = Field(default=None, validation_alias=AliasChoices('start_date','startDate','period_start','start'))
    end_date:   Optional[str] = Field(default=None, validation_alias=AliasChoices('end_date','endDate','period_end','end'))
    instant:    Optional[str] = Field(default=None, validation_alias=AliasChoices('instant','period_instant'))

class XbrlContext(BaseModel):
    model_config = ConfigDict(extra="ignore")
    entity: Optional[str] = None
    period: Optional[XbrlPeriod] = None
    dimensions: Dict[str, str] = {}

# -----------------------------
# Fact 输入（clean → schema）
# -----------------------------
class FactInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # ---- 溯源/元信息 ----
    source_path: Optional[str] = None
    ticker: Optional[str] = None
    form:   Optional[str] = None
    year:   Optional[int]  = None
    accno:  Optional[str]  = None
    doc_date: Optional[str] = None
    fy: Optional[int] = None
    fq: Optional[str] = None
    linkrole: Optional[str] = None

    # ---- 概念 & 取值 ----
    qname: str = Field(validation_alias=AliasChoices('qname','concept','name'))
    # “原始/清洗后/展示”的多路输入，全部接受以便兜底
    value: FactScalar = None
    value_num: Optional[float] = Field(default=None, validation_alias=AliasChoices('value_num', 'numeric_value'))
    value_raw: Optional[str]   = Field(default=None, validation_alias=AliasChoices('value_raw','raw_value'))
    value_num_clean: Optional[float] = None
    value_raw_clean: Optional[str]   = None
    value_display:   Optional[str]   = None

    # ---- 单位/小数位 ----
    unit: Optional[str] = Field(default=None, validation_alias=AliasChoices('unit','uom','unit_normalized'))
    unit_family: Optional[str] = None  # 用于判定百分比
    decimals: Optional[int] = None

    # ---- 上下文/标签 ----
    context_id: Optional[str] = Field(default=None, validation_alias=AliasChoices('context_id','contextRef'))
    context: Optional[XbrlContext] = None
    label_text: Optional[str] = None
    period_label: Optional[str] = None
    rag_text: Optional[str] = None
    statement_hint: Optional[StatementHint] = None

    # ---- 维度 ----
    dimensions_json: Optional[str] = None
    dimensions: Optional[Dict[str, str]] = None

    @field_validator("decimals", mode="before")
    def _coerce_decimals(cls, v):
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v) if v.is_integer() else None
        s = str(v).strip().lower()
        if s in ("inf", "+inf", "-inf", "infinite"):
            return None
        try:
            return int(s)
        except Exception:
            return None

    @model_validator(mode="before")
    def _coalesce_fields(cls, data: dict) -> dict:
        # ---------- NaN → None ----------
        from math import isnan
        for k, v in list(data.items()):
            if isinstance(v, float) and isnan(v):
                data[k] = None
            elif isinstance(v, str) and v.strip() == "":
                # 空串统一成 None（避免后续判断混乱）
                data[k] = None

        # ---------- year/fy 转 int ----------
        for key in ("year", "fy"):
            v = data.get(key)
            if isinstance(v, str) and v.isdigit():
                data[key] = int(v)

        # ---------- context 兜底（若只给了扁平 period_*） ----------
        if data.get("context") is None and any(k in data for k in ("period_start","period_end","instant")):
            data["context"] = {
                "period": {
                    "start_date": data.get("period_start"),
                    "end_date":   data.get("period_end"),
                    "instant":    data.get("instant"),
                }
            }

        # ---------- decimals 归一（字符串/inf） ----------
        dec = data.get("decimals")
        if isinstance(dec, str):
            s = dec.strip().lower()
            if s in ("inf", "infinite", "+inf", "-inf"):
                data["decimals"] = None
            else:
                try:
                    data["decimals"] = int(s)
                except ValueError:
                    data["decimals"] = None
        elif isinstance(dec, float):
            data["decimals"] = int(dec) if dec.is_integer() else None

        # ---------- statement_hint → Enum ----------
        sh = data.get("statement_hint")
        if isinstance(sh, str):
            m = {
                "income": StatementHint.INCOME,
                "balance": StatementHint.BALANCE,
                "cashflow": StatementHint.CASHFLOW,
                "cash flow": StatementHint.CASHFLOW,
                "notes": StatementHint.NOTES,
                "other": StatementHint.OTHER,
            }
            data["statement_hint"] = m.get(sh.strip().lower(), data.get("statement_hint"))

        # ---------- dimensions_json → dimensions ----------
        dj = data.get("dimensions_json")
        if isinstance(dj, str):
            try:
                data["dimensions"] = json.loads(dj)
            except Exception:
                pass

        # ---------- 统一求 value（多路兜底） ----------
        def parse_human_number(s: Optional[str]) -> Optional[float]:
            if s is None:
                return None
            txt = str(s).strip()
            if not txt:
                return None
            # 括号负号、去千分位、去$
            neg = txt.startswith("(") and txt.endswith(")")
            if neg:
                txt = txt[1:-1]
            txt = (txt.replace("\u00A0"," ")
                      .replace(",", "")
                      .replace("$", "")
                      .strip())
            m = re.match(r"^([+-]?\d+(?:\.\d+)?)(?:\s*([KkMmBbTt]))?$", txt)
            if not m:
                return None
            num = float(m.group(1))
            mul = {"K":1e3,"M":1e6,"B":1e9,"T":1e12}.get((m.group(2) or "").upper(), 1.0)
            val = num * mul
            return -val if neg else val

        # 优先级：value_num_clean → value_num → value → value_raw_clean → value_raw → value_display
        val: Optional[float] = None
        if data.get("value_num_clean") is not None:
            val = data["value_num_clean"]
        elif data.get("value_num") is not None:
            val = data["value_num"]
        elif isinstance(data.get("value"), (int, float)) and str(data["value"]).lower() not in ("true","false"):
            val = float(data["value"])
        if val is None and isinstance(data.get("value_raw_clean"), str):
            s = data["value_raw_clean"].strip()
            if s.lower() in ("true","false"):
                data["value"] = (s.lower() == "true")
                val = None
            else:
                try:
                    val = float(Decimal(s.replace(",", "")))
                except Exception:
                    val = parse_human_number(s)
        if val is None and isinstance(data.get("value_raw"), str):
            val = parse_human_number(data["value_raw"])
        if val is None and isinstance(data.get("value_display"), str):
            val = parse_human_number(data["value_display"])

        # 百分比自动 /100：unit_family=percent 或 文本带 % 或 概念名含 percent/percentage
        unit_family = str(data.get("unit_family") or "").lower()
        qname_l = str(data.get("qname") or "").lower()
        text_src = None
        for k in ("value_display","value_raw_clean","value_raw"):
            if isinstance(data.get(k), str) and data[k]:
                text_src = data[k]
                break
        is_percent = (
            unit_family == "percent"
            or (isinstance(text_src, str) and "%" in text_src)
            or ("percent" in qname_l or "percentage" in qname_l)
        )
        if val is not None and is_percent:
            val = val / 100.0

        # 若最终拿到了数值，写回到统一的 value
        if val is not None:
            data["value"] = val
        else:
            # 处理布尔（若还没处理）
            v_raw = data.get("value_raw_clean") or data.get("value_raw")
            if isinstance(v_raw, str) and v_raw.strip().lower() in ("true","false"):
                data["value"] = (v_raw.strip().lower() == "true")

        return data






# -----------------------------
# Fact（Silver 输出） —— CHANGED: 新增扁平 period_* 与回溯字段
# -----------------------------
# -----------------------------
# Fact（Silver 输出）—— 扩充缺失字段
# -----------------------------
class Fact(RecordBase):
    id: UUID = Field(default_factory=uuid4)

    # 概念与取值
    qname: str
    value: Decimal                           # 原始高精度值（保留）
    unit: Optional[str] = None               # 原始/清洗后的单位（见映射逻辑）
    decimals: Optional[int] = None

    # ✅ 新增：衍生/对齐列（全部可选，向后兼容）
    value_raw: Optional[str] = None          # 原始文本（如果能拿到）
    value_num: Optional[float] = None        # 可计算的 float（百分比已/100）
    value_display: Optional[str] = None      # 友好展示文本（K/M/B 等）
    unit_normalized: Optional[str] = None    # 归一化单位：USD / shares / % / pure / ...
    unit_family: Optional[str] = None        # currency / shares / percent / pure
    rag_text: Optional[str] = None           # label+value+period+meta

    # 上下文
    context_id: Optional[str] = None
    context: Optional[XbrlContext] = None

    # 扁平期间（便于 SQL 过滤）
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    instant: Optional[str] = None
    period_label: Optional[str] = None       # ✅ 新增：FY.. instant.. 或 FY.. a→b

    # 语义
    statement_hint: Optional[StatementHint] = None

    # ✅ 新增：维度展平签名
    dims_signature: Optional[str] = None

    # ✅ 新增：规范化期别（方便排序/过滤）
    fy_norm: Optional[int] = None
    fq_norm: Optional[int] = None

    @field_validator("decimals", mode="before")
    def _coerce_decimals_out(cls, v):
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v) if v.is_integer() else None
        s = str(v).strip().lower()
        if s in ("inf", "+inf", "-inf", "infinite"):
            return None
        try:
            return int(s)
        except Exception:
            return None

    @field_validator("qname")
    @classmethod
    def _must_have_colon(cls, v: str) -> str:
        if ":" not in v:
            raise ValueError("qname must include namespace, e.g., 'us-gaap:Revenues'")
        return v


# -----------------------------
# 其它模型（labels/def/cal）保持原状
# -----------------------------
# ---------- CalcEdge：补充 order/linkrole/追溯列 ----------
class CalcEdge(BaseModel):
    model_config = ConfigDict(extra="ignore")

    parent_concept: str
    child_concept: str
    weight: Optional[float] = 1.0
    order: Optional[float] = None
    linkrole: Optional[str] = None

    ticker: Optional[str] = None
    year: Optional[int] = None
    fy: Optional[int] = None
    fq: Optional[str] = None
    form: Optional[str] = None
    accno: Optional[str] = None
    doc_date: Optional[str] = None
    file_type: Optional[str] = None
    source_path: Optional[str] = None

    @field_validator("year", "fy", mode="before")
    def _to_int_optional(cls, v):
        if v is None: return None
        try:
            return int(v)
        except Exception:
            return None

    @field_validator("weight", "order", mode="before")
    def _to_float_optional(cls, v):
        if v is None: return None
        try:
            return float(v)
        except Exception:
            return None

# ---------- DefArc：补充 order/linkrole/追溯列 ----------
class DefArc(BaseModel):
    model_config = ConfigDict(extra="ignore")

    from_concept: str
    to_concept: str
    arcrole: Optional[str] = None
    order: Optional[float] = None
    linkrole: Optional[str] = None
    preferred_label: Optional[str] = None

    ticker: Optional[str] = None
    year: Optional[int] = None
    fy: Optional[int] = None
    fq: Optional[str] = None
    form: Optional[str] = None
    accno: Optional[str] = None
    doc_date: Optional[str] = None
    file_type: Optional[str] = None
    source_path: Optional[str] = None

    @field_validator("year", "fy", mode="before")
    def _to_int_optional(cls, v):
        if v is None: return None
        try:
            return int(v)
        except Exception:
            return None

    @field_validator("order", mode="before")
    def _to_float_optional(cls, v):
        if v is None: return None
        try:
            return float(v)
        except Exception:
            return None

class LabelItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    concept: str
    label_text: Optional[str] = Field(default=None, validation_alias=AliasChoices("label_text","label"))
    label_role: Optional[str] = Field(default=None, validation_alias=AliasChoices("label_role","role"))
    lang: Optional[str] = None

    # 这些是常见追溯字段，允许输入里没有
    ticker: Optional[str] = None
    fy: Optional[int] = None
    fq: Optional[str] = None
    doc_date: Optional[str] = None
    form: Optional[str] = None
    accno: Optional[str] = None
    source_path: Optional[str] = None

# ---------- Labels：best（你 clean 产物的主输入） ----------
class LabelBestItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ticker: Optional[str] = None
    fy: Optional[int] = None
    fq: Optional[str] = None
    form: Optional[str] = None
    accno: Optional[str] = None
    doc_date: Optional[str] = None
    concept: str

    label_best: Optional[str] = None
    label_best_role: Optional[str] = None
    label_best_lang: Optional[str] = None

    # 统一文档化长定义：可能来自 label_doc 或你示例里出现的 label_doc_x / label_doc_y
    label_doc: Optional[str] = None
    label_search_tokens: Optional[str] = None

    @model_validator(mode="before")
    def _coalesce_doc_and_tokens(cls, data: dict) -> dict:
        # 1) doc 合并：优先 label_doc，然后 x/y
        doc = data.get("label_doc")
        if not doc:
            doc = data.get("label_doc_x") or data.get("label_doc_y")
        data["label_doc"] = doc

        # 2) 如果没提供 label_search_tokens，则用 best + doc 自动生成
        if not data.get("label_search_tokens"):
            import re
            parts = []
            if data.get("label_best"): parts.append(str(data["label_best"]))
            if doc: parts.append(str(doc))
            if parts:
                txt = " ".join(parts).lower()
                txt = re.sub(r"[^\w\s\-/%]+", " ", txt).strip()
                txt = re.sub(r"\s+", " ", txt)
                data["label_search_tokens"] = txt or None
        return data


# ---------- Labels：wide（你 clean 产物的宽表） ----------
class LabelWideItem(BaseModel):
    """
    宽表含有大量动态列（label_{role}_{lang}），所以设为 extra='allow' 保留所有展开列
    """
    model_config = ConfigDict(extra="allow")

    ticker: Optional[str] = None
    fy: Optional[int] = None
    fq: Optional[str] = None
    form: Optional[str] = None
    accno: Optional[str] = None
    doc_date: Optional[str] = None
    concept: str
    # 其它所有 label_* 列将被完整保留（extra="allow"）



# -----------------------------
# 工具函数：FactInput -> Fact —— CHANGED: 填充 period_* 与回溯字段
# -----------------------------
# ===== 辅助函数（贴到文件任意位置，_map_factinput_to_fact 前即可） =====
import math

_PERCENT_RE = re.compile(r"percent|percentage", re.I)

def _norm_unit(unit: Optional[str], qname: str, value_display: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """归一化单位；返回 (unit_normalized, unit_family)"""

    # 1) TextBlock 一律不自动推断单位
    if qname and TEXTBLOCK_RE.search(qname):
        return None, None

    # 2) 有 unit 就优先用 unit
    if unit and str(unit).strip():
        s = str(unit).strip()
        m = re.search(r"(?:iso4217:)?([A-Z]{3})\b", s)
        if m:
            return m.group(1).upper(), "currency"
        if "%" in s or "percent" in s.lower():
            return "%", "percent"
        if re.search(r"\bshares?\b", s, re.I):
            return "shares", "shares"
        if re.search(r"\bpure\b", s, re.I):
            return "pure", "pure"
        return s, None

    # 3) 没有 unit 时，只在“可见文本”里出现“数字+%”才推断为百分比
    vis = _strip_html(value_display)
    if PERCENT_NUM_RE.search(vis) or _PERCENT_RE.search(qname or ""):
        return "%", "percent"

    return None, None


def _to_float_maybe(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        s = str(x).strip().replace(",", "")
        if s.endswith("%"):
            s = s[:-1]
        return float(s)
    except Exception:
        return None

def _fmt_value_short(v: Optional[float]) -> Optional[str]:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    av = abs(v)
    if av >= 1e9:  return f"{v/1e9:.3f} B"
    if av >= 1e6:  return f"{v/1e6:.3f} M"
    if av >= 1e3:  return f"{v/1e3:.3f} K"
    return f"{v:.6g}"

def _mk_period_label(fy: Optional[int], fq: Optional[str], start: Optional[str], end: Optional[str], instant: Optional[str]) -> Optional[str]:
    if instant:
        return f"FY{fy} instant {instant}" if fy else f"instant {instant}"
    if start or end:
        a = start or "?"
        b = end or "?"
        if fy and fq:
            return f"FY{fy} {a}→{b}"
        if fy:
            return f"FY{fy} {a}→{b}"
        return f"{a}→{b}"
    return None

def _build_dims_signature(dims: Optional[Dict[str, str]]) -> str:
    if not dims:
        return ""
    items = [f"{k}={v}" for k, v in sorted(dims.items())]
    return "|".join(items)

def _mk_rag_text(label: Optional[str], qname: str, value_display: Optional[str], period_label: Optional[str],
                 ticker: Optional[str], form: Optional[str], accno: Optional[str]) -> str:
    name = label or qname or "(no label)"
    val  = value_display if value_display is not None else ""
    meta = f"{ticker} {form}" + (f" accno={accno}" if accno else "")
    per  = period_label or ""
    return f"{name}: {val} ({per}; {meta})".strip()

def _fq_to_int(fq: Optional[str]) -> Optional[int]:
    if not fq:
        return None
    m = re.search(r"(\d+)", str(fq))
    return int(m.group(1)) if m else None

# ===== 替换你的 _map_factinput_to_fact =====
def _map_factinput_to_fact(fi: "FactInput") -> "Fact":
    # 1) 扁平期间
    p_start = p_end = p_inst = None
    if fi.context and fi.context.period:
        p_start = getattr(fi.context.period, "start_date", None)
        p_end   = getattr(fi.context.period, "end_date", None)
        p_inst  = getattr(fi.context.period, "instant", None)

    # 2) 数值：优先 value_num_clean → 其次 value_raw_clean → 再次 value_display / value
    value_raw = None
    if fi.value_raw_clean is not None:
        value_raw = str(fi.value_raw_clean)
    elif isinstance(fi.value, str):
        value_raw = fi.value

    vnum = None
    if fi.value_num_clean is not None:
        vnum = fi.value_num_clean
    else:
        # 从 value_raw_clean / value_display / value 尝试解析
        for cand in (fi.value_raw_clean, fi.value_display, fi.value):
            vnum = _to_float_maybe(cand)
            if vnum is not None:
                break

    # 3) 归一化单位、百分比 /100
    unit_norm, unit_family = _norm_unit(fi.unit, fi.qname, fi.value_display)
    is_textblock = bool(fi.qname and TEXTBLOCK_RE.search(fi.qname))
    if (not is_textblock) and unit_family == "percent" and vnum is not None:
        vnum = vnum / 100.0

    # 4) 友好展示
    is_textblock = bool(fi.qname and TEXTBLOCK_RE.search(fi.qname))
    if is_textblock:
        # 用简短提示 + 长度
        raw_len = len(str(fi.value_display or fi.value_raw_clean or "")) 
        vdisp = f"[HTML TextBlock ~{raw_len} chars]"
    else:
        vdisp = fi.value_display or _fmt_value_short(vnum)

    # 5) period_label / fy_norm / fq_norm
    period_label = _mk_period_label(fi.fy, fi.fq, p_start, p_end, p_inst)
    fy_norm = int(fi.fy) if isinstance(fi.fy, int) else None
    fq_norm = _fq_to_int(fi.fq)

    # 6) 维度签名
    dims = None
    if fi.dimensions:
        dims = fi.dimensions
    elif fi.context and fi.context.dimensions:
        dims = fi.context.dimensions
    dims_sig = _build_dims_signature(dims)

    # 7) rag_text
    rag = _mk_rag_text(fi.label_text, fi.qname, vdisp, period_label, fi.ticker, fi.form, fi.accno)

    # 8) Decimal 高精度 value（保底：由 vnum 或 0）
    if vnum is not None:
        dec_value = Decimal(str(vnum))
    else:
        # 若没解析出数值，保底 0；仍可从 value_raw 看文本
        dec_value = Decimal("0")

    return Fact(
        # —— 继承 RecordBase 回溯元信息 —— 
        source_path = fi.source_path or "",
        ticker      = fi.ticker,
        form        = fi.form,
        year        = fi.year,
        accno       = fi.accno,
        fy          = fi.fy,
        fq          = fi.fq,
        doc_date    = fi.doc_date,
        linkrole    = fi.linkrole,

        # —— fact 主体 —— 
        qname       = fi.qname,
        value       = dec_value,
        unit        = fi.unit or unit_norm,     # 若入参无 unit，用归一化后的
        decimals    = fi.decimals,
        context_id  = fi.context_id,
        context     = fi.context,
        statement_hint = fi.statement_hint,

        # —— 扁平期间 —— 
        period_start = p_start,
        period_end   = p_end,
        instant      = p_inst,
        period_label = period_label,

        # —— 新增/补齐的衍生字段 —— 
        value_raw   = value_raw,
        value_num   = vnum,
        value_display = vdisp,
        unit_normalized = unit_norm,
        unit_family = unit_family,
        rag_text    = rag,
        dims_signature = dims_sig,
        fy_norm     = fy_norm,
        fq_norm     = fq_norm,
    )


# -----------------------------
# 路由
# -----------------------------
def pick_model_for_file(p: Path) -> Type[BaseModel] | None:
    name = p.name.lower()

    if name.endswith("fact.jsonl"):
        return FactInput

    if name.endswith("text.jsonl") or "text_corpus" in name:
        return TextChunkInput

    if "calculation_edges" in name:
        return CalcEdge

    if "definition_arcs" in name:
        return DefArc

    # ---- labels 路由 ----
    if "labels_wide" in name:
        return LabelWideItem
    if "labels_best" in name:
        return LabelBestItem
    # 你的 clean/labels.jsonl == best，为兼容历史，这里也路由到 best
    if name == "labels.jsonl" or name.endswith("_labels.jsonl") or name.endswith("labels.parquet"):
        return LabelBestItem

    # 若还有 processed 的原始 labels.jsonl（长表）
    if "labels" in name:
        return LabelItem

    return None


# -----------------------------
# IO
# -----------------------------
T = TypeVar("T", bound=BaseModel)

def dump_jsonl(path: Path | str, records: Iterable[T]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")

def load_jsonl_recursive_auto(root: Path | str) -> dict[str, int]:
    root = Path(root)
    stats: dict[str, int] = {}
    for p in root.rglob("*.jsonl"):
        Model = pick_model_for_file(p)
        if Model is None:
            print(f"[skip] {p} (no model router)")
            continue
        try:
            recs = load_jsonl(p, Model)
            stats.setdefault(Model.__name__, 0)
            stats[Model.__name__] += len(recs)
            print(f"[ok] {p} -> {len(recs)} ({Model.__name__})")
        except Exception as e:
            print(f"[fail] {p}: {e}")
    return stats

def load_jsonl(path: Path | str, model: type[T]) -> List[T]:
    path = Path(path)
    out: List[T] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                out.append(model.model_validate(obj))
            except Exception as e:
                raise ValueError(f"{path}:{ln}: {e}") from e
    return out

def load_jsonl_recursive(root: Path | str, model: type[T]) -> List[T]:
    root = Path(root)
    acc: List[T] = []
    for p in root.rglob("*.jsonl"):
        acc.extend(load_jsonl(p, model))
    return acc

# -----------------------------
# 主流程 —— CHANGED: 对 fact.jsonl 做瘦身映射
# -----------------------------
def process_tree(
    in_root: Path,
    out_root: Path,
    *,
    export_format: str = "jsonl",
    overwrite: bool = False,
    verbose: bool = True,
) -> dict:
    assert export_format in {"jsonl", "parquet", "csv"}
    stats = {"ok": 0, "skip_no_model": 0, "skip_exists": 0, "fail": 0}

    for p in sorted(in_root.rglob("*.jsonl")):
        Model = pick_model_for_file(p)
        if Model is None:
            stats["skip_no_model"] += 1
            if verbose:
                print(f"[skip] {p} (no model router)")
            continue

        rel = p.relative_to(in_root)
        out_path = (out_root / rel)
        if export_format == "parquet":
            out_path = out_path.with_suffix(".parquet")
        elif export_format == "csv":
            out_path = out_path.with_suffix(".csv")

        if out_path.exists() and not overwrite:
            stats["skip_exists"] += 1
            if verbose:
                print(f"[skip] {out_path} (exists, use --overwrite to force)")
            continue

        try:
            recs = load_jsonl(p, Model)

            # 仅对 fact 做映射（瘦身+扁平期）
            if Model is FactInput:
                recs = [_map_factinput_to_fact(x) for x in recs]

            if export_format == "jsonl":
                dump_jsonl(out_path, recs)
            elif export_format == "parquet":
                dump_parquet(out_path, recs)
            else:
                dump_csv(out_path, recs)

            stats["ok"] += 1
            if verbose:
                print(f"[ok] {p} -> {out_path} ({Model.__name__}, {len(recs)} rows)")
        except Exception as e:
            stats["fail"] += 1
            print(f"[fail] {p}: {e}", file=sys.stderr)

    if verbose:
        print("[summary]", stats)
    return stats

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch parse & export clean → silver (mirror paths)")
    parser.add_argument("--in-root",  type=str, default="data/clean",  help="Input root directory (clean)")
    parser.add_argument("--out-root", type=str, default="data/silver", help="Output root directory (silver)")
    parser.add_argument("--format",   type=str, default="jsonl", choices=["jsonl", "parquet", "csv"],
                        help="Export format")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--quiet", action="store_true", help="Less logs")
    args = parser.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    verbose = not args.quiet

    if not in_root.exists():
        print(f"[error] input root not found: {in_root}", file=sys.stderr)
        sys.exit(1)

    process_tree(
        in_root,
        out_root,
        export_format=args.format,
        overwrite=args.overwrite,
        verbose=verbose,
    )
