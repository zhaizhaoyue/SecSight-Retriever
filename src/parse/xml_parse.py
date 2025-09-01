#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, os, re, sys
import numpy as np
from arelle import Cntlr, FileSource
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal, InvalidOperation
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
# ---------------- filename regex ----------------
DASH_CLASS = r"\-\u2010\u2011\u2012\u2013\u2014\u2212"  # -, ‐, -, ‒, –, —, −
ACCNO_SEARCH_RE = re.compile(rf"(?<!\d)(\d{{10}})[{DASH_CLASS}](\d{{2}})[{DASH_CLASS}](\d{{6}})(?!\d)")
FORM_RE  = re.compile(r"\b(10-K|10-Q|20-F|40-F|8-K)\b", re.I)
DATE8_RE = re.compile(r"(?<!\d)(\d{8})(?!\d)")

# ---------------- try import config ----------------
def _try_import_cfg():
    try:
        from config import cfg_get as _cfg_get, path_join as _path_join
        return _cfg_get, _path_join
    except Exception:
        try:
            import yaml
        except Exception:
            yaml = None

        def _project_root_guess() -> Path:
            p = Path(__file__).resolve()
            for parent in p.parents:
                if (parent / ".git").exists():
                    return parent
            return p.parent

        _PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", _project_root_guess()))
        _CFG_PATH = _PROJECT_ROOT / "configs" / "config.yaml"

        _CFG_RAW: Dict[str, Any] = {}
        if yaml and _CFG_PATH.exists():
            with open(_CFG_PATH, "r", encoding="utf-8") as f:
                _CFG_RAW = yaml.safe_load(f) or {}

        def _expand(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _expand(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_expand(x) for x in obj]
            if isinstance(obj, str):
                return obj.replace("${project_root}", str(_PROJECT_ROOT))
            return obj

        _CFG = _expand(_CFG_RAW)

        def _cfg_get(path: str, default: Any = None) -> Any:
            cur: Any = _CFG
            for key in path.split("."):
                if not isinstance(cur, dict) or key not in cur:
                    return default
                cur = cur[key]
            return cur

        def _path_join(*parts: str | os.PathLike) -> Path:
            return Path(*parts).expanduser().resolve()

        return _cfg_get, _path_join

_cfg_get, _path_join = _try_import_cfg()

# ---------------- helpers: file meta ----------------
ACCNO_RE = re.compile(r"\b\d{10}-\d{2}-\d{6}\b")
FORM_SET = {"10-K","10-Q","20-F","40-F","8-K"}


def normalize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """将 DataFrame 中 pyarrow 不友好的类型转为可写 Parquet 的类型。
    - Decimal -> str（保留精度；如需数值计算可改成 float）
    - dict/list -> JSON 字符串
    - 统一部分文本列为 string dtype
    """
    df = df.copy()

    # 1) Decimal 列 -> str（保精度）
    def dec_to_str(x):
        return str(x) if isinstance(x, Decimal) else x

    if "value" in df.columns:
        df["value"] = df["value"].apply(dec_to_str)

    # 若 value_num 里也可能混入 Decimal（通常不会），也转一下
    if "value_num" in df.columns:
        df["value_num"] = df["value_num"].apply(
            lambda x: float(x) if isinstance(x, Decimal) else x
        )

    # 2) dict/list（如 context）-> JSON 字符串
    def to_json_if_needed(x):
        return json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x

    if "context" in df.columns:
        df["context"] = df["context"].apply(to_json_if_needed)

    # 3) 统一一些文本列类型，避免混合类型触发推断问题
    textish = [
        "label_text","period_label","statement_hint","unitRef","decimals",
        "qname","ticker","form","year","accno","doc_date","source_path",
        "context_id","value_display","unit_normalized","unit_family",
        "period_type","start_date","end_date","instant","entity","dimensions_json"
    ]
    for col in textish:
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df

from datetime import date, timedelta

def derive_period_fq_fye(ps, pe, inst, fye):
    d = inst or pe or ps  # 'YYYY-MM-DD'
    if not d or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
        return None
    yyyy, mm, dd = map(int, (d[:4], d[5:7], d[8:10]))
    cur = date(yyyy, mm, dd)

    if fye and re.fullmatch(r"\d{2}-\d{2}", fye):
        fye_m, fye_d = map(int, (fye[:2], fye[3:5]))
        # 当年财年截止日
        fye_this = date(yyyy, fye_m, fye_d)
        # 如果 instant 恰好是 FYE 的次日（常见 XBRL 表达）
        if cur == fye_this + timedelta(days=1):
            return "Q4"  # 归回上一财年的 Q4
    # 其他日期按原先平移逻辑
    mm = cur.month
    fye_mm = int(fye[:2]) if fye and re.fullmatch(r"\d{2}-\d{2}", fye) else None
    if not fye_mm:
        return month_to_q(mm)
    shifted = (mm - (fye_mm % 12)) % 12
    if shifted in (1,2,3): return "Q1"
    if shifted in (4,5,6): return "Q2"
    if shifted in (7,8,9): return "Q3"
    if shifted in (10,11,0): return "Q4"
    return None


def build_dims_signature(d: Dict[str,str]) -> str:
    if not d: return ""
    items = [f"{k}={v}" for k,v in sorted(d.items())]
    return "|".join(items)

def to_parquet_with_decimal(df: pd.DataFrame, out_path: Path):
    # 推断小数精度，保险起见统一给个较大的 scale/precision
    # 常用：precision=38, scale=10（自行按你的数据调整）
    schema_fields = []
    for col in df.columns:
        if col == "value":
            schema_fields.append(pa.field(col, pa.decimal128(38, 10)))
        elif col == "context":
            schema_fields.append(pa.field(col, pa.string()))
        else:
            # 交给 from_pandas 推断
            pass

    # 先把 context 转成 json 字符串
    df2 = df.copy()
    if "context" in df2.columns:
        df2["context"] = df2["context"].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x)

    # 再构建 Arrow Table
    table = pa.Table.from_pandas(df2, preserve_index=False)
    # 可选：如果需要强制 cast
    # table = table.set_column(table.schema.get_field_index("value"),
    #                          "value",
    #                          pa.compute.cast(table.column("value"), pa.decimal128(38,10)))

    pq.write_table(table, out_path)


def sniff_meta(p: Path) -> Dict[str, Any]:
    name = p.name
    base = p.stem
    meta = {
        "ticker": None, "year": None, "form": None,
        "accno": None, "doc_date": None, "fy": None, "fq": None,
        "source_path": str(p),
    }

    m_acc = ACCNO_SEARCH_RE.search(name) or ACCNO_SEARCH_RE.search(str(p))
    if m_acc:
        meta["accno"] = f"{m_acc.group(1)}-{m_acc.group(2)}-{m_acc.group(3)}"

    m_form = FORM_RE.search(name)
    if m_form:
        meta["form"] = m_form.group(1).upper()

    dates = list(DATE8_RE.finditer(name))
    if dates:
        meta["doc_date"] = dates[-1].group(1)

    tokens = base.split("_")
    if tokens and tokens[0].upper() == "US":
        tokens = tokens[1:]

    def looks_ticker(tok: str) -> bool:
        return bool(re.fullmatch(r"[A-Z0-9.\-]{1,12}", tok))

    if len(tokens) >= 1 and looks_ticker(tokens[0].upper()):
        meta["ticker"] = tokens[0].upper()

    if len(tokens) >= 2 and tokens[1].isdigit() and len(tokens[1]) == 4:
        meta["year"] = tokens[1]

    if not meta.get("year") and meta.get("doc_date"):
        meta["year"] = meta["doc_date"][:4]

    if meta.get("year") and str(meta["year"]).isdigit():
        meta["fy"] = int(meta["year"])

    if meta.get("doc_date"):
        try:
            mm = int(meta["doc_date"][4:6])
            meta["fq"] = {3:"Q1",6:"Q2",9:"Q3",12:"Q4"}.get(mm)
        except Exception:
            pass

    return meta

def sniff_from_parents(p: Path) -> Dict[str, Any]:
    parts = list(p.parts)
    up = [s.upper() for s in parts]
    out = {"ticker": None, "form": None, "accno": None}
    for i, seg in enumerate(up):
        if seg in FORM_SET:
            if i > 0 and re.fullmatch(r"[A-Z0-9.\-]{1,12}", up[i-1]):
                out["ticker"] = up[i-1]
            if i + 1 < len(parts):
                seg_next = parts[i+1]
                m_acc = ACCNO_SEARCH_RE.fullmatch(seg_next) or ACCNO_SEARCH_RE.search(seg_next)
                if m_acc:
                    out["accno"] = f"{m_acc.group(1)}-{m_acc.group(2)}-{m_acc.group(3)}"
            out["form"] = seg
            break
    if not out["accno"]:
        for seg in parts:
            m_acc = ACCNO_SEARCH_RE.fullmatch(seg) or ACCNO_SEARCH_RE.search(seg)
            if m_acc:
                out["accno"] = f"{m_acc.group(1)}-{m_acc.group(2)}-{m_acc.group(3)}"
                break
    return out

def enrich_meta_from_dei(model_xbrl, meta: Dict[str, Any]) -> Dict[str, Any]:
    import re

    def qname_localname(qn) -> Optional[str]:
        if hasattr(qn, "localName") and qn.localName:
            return qn.localName
        if qn is not None:
            s = str(qn)
            if "}" in s and s.startswith("{"):
                return s.split("}", 1)[1]
            if ":" in s:
                return s.split(":", 1)[1]
            return s
        return None

    def get_dei(local: str) -> Optional[str]:
        for f in getattr(model_xbrl, "facts", []):
            qn = getattr(f, "qname", None) or getattr(getattr(f, "concept", None), "qname", None)
            ln = qname_localname(qn)
            if ln == local:
                v = getattr(f, "xValue", None)
                if v is None:
                    v = getattr(f, "value", None)
                if v is not None:
                    return str(v).strip()
        return None

    ticker = get_dei("TradingSymbol")
    form   = get_dei("DocumentType")
    fy_s   = get_dei("DocumentFiscalYearFocus")
    fq_s   = get_dei("DocumentFiscalPeriodFocus")
    dped   = get_dei("DocumentPeriodEndDate")  # YYYY-MM-DD
    fye_s  = get_dei("CurrentFiscalYearEndDate")  # --09-30

    if not meta.get("ticker") and ticker:
        meta["ticker"] = ticker.upper()

    if not meta.get("form") and form:
        fup = form.upper()
        m = re.search(r"(10-K|10-Q|20-F|40-F|8-K)", fup)   # 兼容 "10-Q/A"
        if m:
            meta["form"] = m.group(1)

    if not meta.get("doc_date") and dped and re.fullmatch(r"\d{4}-\d{2}-\d{2}", dped):
        meta["doc_date"] = dped.replace("-", "")

    if not meta.get("fy"):
        if fy_s and fy_s.isdigit():
            meta["fy"] = int(fy_s)
        elif meta.get("doc_date"):
            meta["fy"] = int(meta["doc_date"][:4])

    # ✅ 关键：保留 DEI 的 FQ 原值（Q1/Q2/Q3/Q4/FY），不要把 FY 置空
    if not meta.get("fq") and fq_s:
        q = fq_s.upper()
        if q in {"Q1","Q2","Q3","Q4","FY"}:
            meta["fq"] = q

    # 可选：保留公司 FYE（兜底推断时有用）
    if fye_s:
        m = re.search(r"(\d{2})-(\d{2})", fye_s)
        if m:
            meta["fye"] = f"{m.group(1)}-{m.group(2)}"

    if not meta.get("year") and meta.get("fy"):
        meta["year"] = str(meta["fy"])

    return meta


# ---------------- arelle helpers ----------------
from arelle import Cntlr, FileSource

def load_instance(ctrl: Cntlr.Cntlr, file_path: str):
    mm = ctrl.modelManager
    try:
        model_xbrl = mm.load(file=file_path)
        if model_xbrl: return model_xbrl
    except TypeError:
        pass
    fs = FileSource.openFileSource(file_path, ctrl)
    model_xbrl = mm.load(fs)
    fs.close()
    return model_xbrl

def derive_period_fy(ps: Optional[str], pe: Optional[str], inst: Optional[str], fy_fallback: Optional[int]) -> Optional[int]:
    d = inst or pe or ps
    if d and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
        return int(d[:4])
    return fy_fallback

def month_to_q(mm: int) -> Optional[str]:
    return {1:"Q1",2:"Q1",3:"Q1",4:"Q2",5:"Q2",6:"Q2",7:"Q3",8:"Q3",9:"Q3",10:"Q4",11:"Q4",12:"Q4"}.get(mm)

def derive_period_fq(ps: Optional[str], pe: Optional[str], inst: Optional[str]) -> Optional[str]:
    d = inst or pe or ps
    if d and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
        try:
            return month_to_q(int(d[5:7]))
        except Exception:
            return None
    return None


def fact_qname(f) -> str:
    qn = getattr(getattr(f,"concept",None),"qname",None) or getattr(f,"qname",None)
    return str(qn) if qn else ""

def fact_value_num(f):
    xv = getattr(f,"xValue",None)
    return xv if xv is not None and not isinstance(xv,str) else None

def to_iso(dt) -> Optional[str]:
    if not dt: return None
    if hasattr(dt, "isoformat"):
        return dt.isoformat()[:10]
    s = str(dt)
    return s[:10] if re.fullmatch(r"\d{4}\-\d{2}\-\d{2}.*", s) else s

def fact_period(f):
    ctx = getattr(f,"context",None)
    if ctx is None:
        return None,None,None
    if getattr(ctx,"startDatetime",None) or getattr(ctx,"endDatetime",None) or getattr(ctx,"instantDatetime",None):
        return to_iso(ctx.startDatetime), to_iso(ctx.endDatetime), to_iso(ctx.instantDatetime)
    return getattr(ctx,"startDate",None), getattr(ctx,"endDate",None), getattr(ctx,"instantDate",None)

def context_to_dict(ctx) -> Dict[str, Any]:
    if ctx is None:
        return {"entity": None, "period": {"start_date": None, "end_date": None, "instant": None}, "dimensions": {}}
    # entity
    ent = None
    try:
        scheme, ident = ctx.entityIdentifier
        ent = ident or None
    except Exception:
        ent = None
    # period
    ps, pe, inst = fact_period(type("F", (), {"context": ctx}))
    period = {"start_date": ps, "end_date": pe, "instant": inst}
    # dimensions
    dims: Dict[str,str] = {}
    try:
        qd = getattr(ctx, "qnameDims", {}) or {}
        for dim_qn, mem in qd.items():
            dim = str(dim_qn)

            mem_qn = getattr(mem, "memberQname", None)
            if mem_qn is None:
                mem_qn = getattr(mem, "typedMember", None)
            if mem_qn is None:
                mem_qn = getattr(mem, "qname", None)

            if hasattr(mem_qn, "qname"):            # arelle 的维度成员对象
                mem_str = str(getattr(mem_qn, "qname"))
            elif hasattr(mem_qn, "prefixedName"):   # 某些类型有 prefixedName
                mem_str = str(mem_qn.prefixedName)
            else:
                mem_str = str(mem_qn) if mem_qn is not None else ""

            dims[str(dim)] = mem_str
    except Exception:
        pass
    return {"entity": ent, "period": period, "dimensions": dims}

def unit_to_str(u) -> Optional[str]:
    if u is None:
        return None
    try:
        # arelle unit 可能是 measures[0] / measures[1] 分子分母
        measures = getattr(u, "measures", None)
        if measures:
            num = measures[0] if len(measures) > 0 else []
            den = measures[1] if len(measures) > 1 else []
            def _fmt(lst):
                return "*".join([str(x) for x in lst]) if lst else ""
            s = _fmt(num)
            if den:
                s = s + "/" + _fmt(den)
            return s or getattr(u, "id", None)
        # 退回 id
        return getattr(u, "id", None)
    except Exception:
        return getattr(u, "id", None)

UNIT_CURRENCY_PAT = re.compile(r"(iso4217:)?([A-Z]{3})(?:\b|[^A-Z])")
UNIT_SHARE_PAT    = re.compile(r"\b(shares?|share)\b", re.I)
UNIT_PURE_PAT     = re.compile(r"\bpure\b", re.I)
UNIT_PERCENT_PAT  = re.compile(r"\bpercent\b", re.I)
def normalize_unit(unit_str: Optional[str], qname_str: str, value_display: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    返回: (unit_normalized, unit_family)
    unit_family ∈ {"currency","shares","percent","pure", None}
    """
    if not unit_str:
        # 无 unit 时尝试从 qname / 文本里推断百分号
        if value_display and "%" in value_display:
            return "%", "percent"
        if "percent" in qname_str.lower() or "percentage" in qname_str.lower():
            return "%", "percent"
        return None, None

    us = unit_str.strip()

    # 货币：抓 ISO4217
    m = UNIT_CURRENCY_PAT.search(us)
    if m:
        iso = m.group(2)
        return iso.upper(), "currency"

    # 股数
    if UNIT_SHARE_PAT.search(us):
        return "shares", "shares"

    # 百分比（部分 taxonomy 会写 percent/pure）
    if UNIT_PERCENT_PAT.search(us):
        return "%", "percent"

    # pure（无单位）
    if UNIT_PURE_PAT.search(us):
        # value_display 带 % 的话仍按百分比处理
        if value_display and "%" in value_display:
            return "%", "percent"
        return "pure", "pure"

    # 没识别出来就原样返回
    return us, None

def to_float_maybe(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, Decimal):
        try:
            return float(x)
        except Exception:
            return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(",", "")
    # 去掉尾部百分号（先不除以100，这一步只做“能转 float”）
    if s.endswith("%"):
        s = s[:-1]
    try:
        return float(s)
    except Exception:
        return None

def concept_label_text(f, lang_priority=("en","en-US","en-GB")) -> Optional[str]:
    try:
        c = getattr(f, "concept", None)
        if c is None: return None
        # standard label role
        for lang in lang_priority:
            lbl = c.label(lang=lang)  # arelle 会按标准 role 查
            if lbl: return str(lbl).strip()
        # 退路：不指明语言
        lbl = c.label()
        return str(lbl).strip() if lbl else None
    except Exception:
        return None
    
BALANCE_PREFIXES = {
    "us-gaap:propertyplantandequipment",    # 任意 PPE 变体
    "us-gaap:propertyplantandequipmentgross",
    "us-gaap:accumulateddepreciation",
    "us-gaap:intangibleasset",
    "us-gaap:goodwill",
    "us-gaap:inventory",
    # 其他你想强制归类到 balance 的概念前缀……
}

def guess_statement_hint(qname_str: str) -> str:
    s = qname_str.lower()
    if any(s.startswith(p) for p in BALANCE_PREFIXES):
        return "balance"
    if any(k in s for k in ["revenue","income","earnings","profit","loss","eps"]):
        return "income"
    if any(k in s for k in ["cash","operatingactivities","investingactivities","financingactivities"]):
        return "cashflow"
    if any(k in s for k in ["disclosure","note","supplemental","schedule","textblock"]):
        return "notes"
    if any(k in s for k in ["asset","liabilit","equity","payable","receivable","goodwill","inventory"]):
        return "balance"
    return "other"


def period_label_builder(year: Optional[str|int], ps, pe, inst) -> Optional[str]:
    fy = f"FY{year}" if year else "FY?"
    if inst:
        return f"{fy} instant {inst}"
    if ps or pe:
        return f"{fy} {ps or '?'} — {pe or '?'}"
    return None

def iter_instance_files(indir: Path) -> List[Path]:
    cands = []
    for p in indir.rglob("*"):
        if not p.is_file(): continue
        name = p.name.lower()
        if any(name.endswith(s) for s in ["_lab.xml","_def.xml","_pre.xml","_cal.xml"]): continue
        if name.endswith((".xml",".htm",".html","_htm.xml")):
            cands.append(p)
    return cands

# ---------------- main parse (REWRITTEN) ----------------
def parse_one(file_path: Path) -> pd.DataFrame:
    ctrl = Cntlr.Cntlr(logFileName=None)
    mx = load_instance(ctrl, str(file_path))
    rows = []

    # 1) 文件名/路径推断 + 2) 父目录兜底 + 3) DEI 兜底
    base_meta = sniff_meta(file_path)
    pmeta = sniff_from_parents(file_path)
    for k, v in pmeta.items():
        if not base_meta.get(k) and v:
            base_meta[k] = v
    base_meta = enrich_meta_from_dei(mx, base_meta)

    if not base_meta.get("ticker") or not base_meta.get("form") or not base_meta.get("accno"):
        print(f"[dbg] meta incomplete -> {file_path.name} -> {base_meta}")

    # 4) 遍历事实，收集所需字段
    for f in mx.facts:
        if getattr(f,"isNil",False):
            continue

        # 基本
        qname = fact_qname(f)
        ctx   = getattr(f, "context", None)
        unit  = getattr(f, "unit", None)

        # 值
        value_raw = getattr(f, "value", None)
        value_num = fact_value_num(f)
        # value_display：始终字符串（便于 UI 显示）
        value_display = None if value_raw is None else str(value_raw).strip()

        # value：优先数值（Decimal），否则原串
        value: Any = None
        if value_num is not None:
            try:
                value = Decimal(str(value_num))
            except (InvalidOperation, ValueError):
                value = value_num
        else:
            if value_raw is not None:
                s = str(value_raw).strip()
                # 尝试把纯数字字符串转 Decimal
                try:
                    value = Decimal(s.replace(",", ""))
                except (InvalidOperation, ValueError):
                    value = s

        # period
        ps, pe, inst = fact_period(f)

        # 组织 context 字段
        ctx_dict = context_to_dict(ctx)
        context_id = getattr(ctx, "id", None) if ctx is not None else None

        # 其他
        unit_ref  = unit_to_str(unit)
        unit_norm, unit_family = normalize_unit(unit_ref, qname, value_display)
        decimals_raw = getattr(f, "decimals", None)
        decimals = None
        if decimals_raw is not None:
            s = str(decimals_raw).strip().lower()
            if s not in ("inf", "infinite"):
                try:
                    decimals = int(decimals_raw)
                except Exception:
                    decimals = None
        label_txt = concept_label_text(f)
        stmt_hint = guess_statement_hint(qname)

        # === 期间（基于 FYE 计算季度） ===
        period_fy = derive_period_fy(ps, pe, inst, base_meta.get("fy"))
        period_fq = derive_period_fq_fye(ps, pe, inst, base_meta.get("fye")) or base_meta.get("fq")

        # === 申报批次（来自 DEI），与事实期间区分 ===
        filing_fy = base_meta.get("fy")
        filing_fq = base_meta.get("fq")

        # 用“期间财年”生成更准确的 label
        per_label = period_label_builder(period_fy, ps, pe, inst)

        # === 构造 row（一次性写全，避免先 update 再覆盖） ===
        row = dict(
            # —— 事实所属期间 —— 
            period_fy=period_fy,
            period_fq=period_fq,
            fy=period_fy,     # 行级 fy/fq 用事实期间
            fq=period_fq,

            # —— 申报批次（方便筛选/分组） ——
            filing_fy=filing_fy,
            filing_fq=filing_fq,

            # —— 概念与取值 ——
            qname=qname,
            value=value,                 # 若后续转为 value_num，可在 main() 里统一处理
            value_raw=value_raw,
            value_num=value_num,         # 这里是 arelle 的 xValue（可能为 None），main() 再标准化
            value_display=value_display,

            # —— 单位/小数 —— 
            unitRef=unit_ref,
            unit_normalized=unit_norm,
            unit_family=unit_family,
            decimals=decimals,

            # —— 上下文 —— 
            context_id=context_id,
            context=ctx_dict,

            # —— 文本/标签/提示 —— 
            label_text=label_txt,
            period_label=per_label,
            statement_hint=stmt_hint,

            # —— 期间细节 —— 
            period_type=("instant" if inst else ("duration" if (ps or pe) else None)),
            start_date=ps,
            end_date=pe,
            instant=inst,

            # —— 从 context 摊平 —— 
            entity=ctx_dict.get("entity"),
            dimensions=ctx_dict.get("dimensions", {}),

            # —— 元数据 —— 
            ticker=(base_meta.get("ticker") or None),
            form=(base_meta.get("form") or None),
            year=(str(base_meta.get("year")) if base_meta.get("year") else None),
            accno=(base_meta.get("accno") or None),
            doc_date=(base_meta.get("doc_date") or None),
            source_path=str(file_path),
        )

        # === 维度签名（只做一次） ===
        row["dims_signature"] = build_dims_signature(row["dimensions"])

        rows.append(row)


    mx.close()
    return pd.DataFrame(rows)

# ---------------- CLI & export ----------------
def main():
    ap = argparse.ArgumentParser(description="Parse XBRL with arelle → Silver-ready facts")
    ap.add_argument("--in", dest="indir", default=None, help="Input root (raw_reports/standard)")
    ap.add_argument("--out", dest="outdir", default=None, help="Output root (processed)")
    ap.add_argument("--to", dest="fmt", nargs="+", choices=["parquet","csv","jsonl"],
                    default=["parquet","jsonl"])
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    raw_in  = _cfg_get("data.raw_reports.standard", "${project_root}/data/raw_reports/standard")
    raw_out = _cfg_get("data.processed", "${project_root}/data/processed")

    prj_root = Path(__file__).resolve().parents[2]
    indir  = Path(args.indir).resolve() if args.indir else Path(raw_in.replace("${project_root}", str(prj_root))).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else Path(raw_out.replace("${project_root}", str(prj_root))).resolve()

    files = iter_instance_files(indir)
    if not files:
        raise SystemExit(f"No instance files found in {indir}")

    for fp in files:
        try:
            df = parse_one(fp)
        except Exception as e:
            print(f"[fail] {fp.name}: {e}", file=sys.stderr)
            continue
        if df.empty:
            continue
        # ========== Silver → Gold 轻加工：数值与维度归档 ==========
        df["value_num"] = df.apply(lambda r: to_float_maybe(r.get("value")), axis=1)

        percent_mask = (
            (df["unit_family"].fillna("") == "percent")
            | df["value_display"].fillna("").str.contains("%", regex=False)
            | df["qname"].fillna("").str.lower().str.contains("percent|percentage")
        )
        df.loc[percent_mask & df["value_num"].notna(), "value_num"] = \
            df.loc[percent_mask & df["value_num"].notna(), "value_num"] / 100.0

        if "value" in df.columns:
            df = df.drop(columns=["value"])
        if "value_num_clean" in df.columns:
            df = df.drop(columns=["value_num_clean"])

        if "decimals" in df.columns:
            dec = pd.to_numeric(df["decimals"], errors="coerce")
            dec = dec.replace([np.inf, -np.inf], np.nan)
            df["decimals"] = pd.array(dec, dtype="Int64")

        if "dims_signature" not in df.columns and "dimensions" in df.columns:
            df["dims_signature"] = df["dimensions"].apply(build_dims_signature)
        # 以解析后的 DataFrame 拿元数据（已 DEI 兜底）
        def first_nonnull(df, col, default=None):
            if col in df.columns:
                s = df[col].dropna()
                if not s.empty:
                    return str(s.iloc[0])
            return default

        ticker = first_nonnull(df, "ticker", "UNKNOWN").upper()
        year = first_nonnull(df, "year", None)
        if not year:
            fy = first_nonnull(df, "doc_date", None)
            year = fy[:4] if fy else "NA"

        form = first_nonnull(df, "form", None)
        if not form:
            m_form = FORM_RE.search(Path(fp).name)
            form = m_form.group(1).upper() if m_form else "NA"
        else:
            form = form.upper()

        accno = first_nonnull(df, "accno", None) or "ACCNO_NA"

        out_dir = outdir / ticker / year / f"{form}_{accno}"
        out_dir.mkdir(parents=True, exist_ok=True)

        fmts = [f.lower() for f in (args.fmt or ["parquet","jsonl"])]
        for fmt in fmts:
            df_to_save = df
            if fmt in ("parquet", "csv"):
                df_to_save = normalize_for_parquet(df)

            if fmt == "parquet":
                out_path = out_dir / "facts.parquet"
                df_to_save.to_parquet(out_path, index=False)
            elif fmt == "csv":
                out_path = out_dir / "facts.csv"
                df_to_save.to_csv(out_path, index=False, encoding="utf-8")
            elif fmt == "jsonl":
                out_path = out_dir / "facts.jsonl"
                # JSONL 可以保留原 df，这样 Decimal 会被自动序列化为字符串；如果你想统一也可用 df_to_save
                df.to_json(out_path, orient="records", lines=True, force_ascii=False)
            else:
                raise ValueError(f"Unknown format: {fmt}")

            if not args.quiet:
                print(f"[ok] saved {len(df)} facts -> {out_path}")

if __name__ == "__main__":
    main()
