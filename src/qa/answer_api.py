# src/qa/answer_api.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os
import json
import time
import logging
import re

from .utils_citation import ensure_citations, citation_to_display

# ======================================================================
# SECTION 0) 配置与常量
# ======================================================================
USE_LLM = os.getenv("RAG_USE_LLM", "true").lower() in {"1", "true", "yes", "y"}
PROMPT_VERSION = os.getenv("RAG_PROMPT_VER", "2025-09-02-v2")
LLM_MODEL_NAME = os.getenv("RAG_LLM_MODEL", "your-llm-name")  # 仅用于记录
FACTS_NUMERIC_PATH = os.getenv("RAG_FACTS_NUMERIC", "data/clean/facts_numeric.parquet")
# 禁止 numeric 失败后走文本兜底；想临时恢复，把环境变量设为 1
DISABLE_TEXT_FALLBACK = os.getenv("RAG_DISABLE_TEXT_FALLBACK", "1") in {"1","true","yes","y"}
_REVENUE_ALIASES = [
    "us-gaap:SalesRevenueNet",
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap:SalesRevenueGoodsNet",
    "us-gaap:SalesRevenueServicesNet",
    "us-gaap:Revenues",
]
def _pick_best_revenue_concept(df, ticker: Optional[str], form: Optional[str], fy: Optional[int]) -> Optional[str]:
    """
    在 facts_numeric 里为给定标的挑一个“有数据”的营收概念。
    优先：匹配 ticker/form 的行数（若 form 下无数据，则放宽 form）。
    """
    try:
        sub = df.copy()
        if ticker:
            sub = sub[sub["ticker"] == ticker]
        if fy is not None and "fy_norm" in sub.columns:
            sub = sub[sub["fy_norm"] == int(fy)]
        # 先带 form 统计，再放宽
        def _score(use_form: bool):
            s = sub
            if use_form and form:
                s = s[s["form"] == form]
            s = s[s["concept"].isin(_REVENUE_ALIASES)]
            if s.empty:
                return None
            # 统计每个概念的可用 value_std 数量
            counts = s.groupby("concept")["value_std"].apply(lambda x: x.notna().sum()).sort_values(ascending=False)
            return counts.index[0] if len(counts) else None

        best = _score(True) or _score(False)
        return str(best) if best else None
    except Exception:
        return None
    
# GAAP 概念映射
GAAP = {
    "revenue": "us-gaap:SalesRevenueNet",
    "gross_profit": "us-gaap:GrossProfit",
    "operating_income": "us-gaap:OperatingIncomeLoss",
    "net_income": "us-gaap:NetIncomeLoss",
    "r_and_d": "us-gaap:ResearchAndDevelopmentExpense",
    "sga": "us-gaap:SellingGeneralAndAdministrativeExpense",
    "cfo": "us-gaap:NetCashProvidedByUsedInOperatingActivities",
    "capex": "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
    "eps_diluted": "us-gaap:EarningsPerShareDiluted",
    "eps_basic": "us-gaap:EarningsPerShareBasic",
    "total_assets": "us-gaap:Assets",
    "total_liabilities": "us-gaap:Liabilities",
    "current_assets": "us-gaap:AssetsCurrent",
    "current_liabilities": "us-gaap:LiabilitiesCurrent",
    "total_debt": "us-gaap:LongTermDebtAndCapitalLeaseObligations",
    "cash": "us-gaap:CashAndCashEquivalentsAtCarryingValue",
}

# 触发词
TRIG = {
    "ttm": ["ttm", "trailing twelve months", "过去四个季度", "近四季"],
    "cagr": ["cagr", "复合增长", "复合年增长率", "compound annual growth"],
    "gross_margin": ["gross margin", "毛利率"],
    "op_margin": ["operating margin", "经营利润率", "营业利润率"],
    "net_margin": ["net margin", "净利率"],
    "rd_ratio": ["r&d ratio", "r&d as % of", "研发占比", "研发费用率"],
    "sga_ratio": ["sga ratio", "sga as % of", "销售管理费用占比", "期间费用率"],
    "fcf": ["free cash flow", "fcf", "自由现金流"],
    "eps": ["eps", "每股收益", "earnings per share"],
    "diluted": ["diluted", "摊薄"],
    "basic": ["basic", "基本"],
    "yoy": ["yoy", "同比", "year over year", "year-over-year"],
    "qoq": ["qoq", "环比", "quarter over quarter", "quarter-over-quarter"],
    "diff": ["difference", "变化", "变动", "差额", "增减"],
    "revenue": ["revenue", "revenues", "sales", "net sales", "营收", "营业收入", "净销售额"],
    "net_income": ["net income", "净利润", "profit", "净收益"],
    "cash": ["cash", "现金"],
}

# 强制 numeric 的触发（用于 detect 降级失败时）
Y_NUM_TRIG = ["yoy","同比","year-over-year","比去年","同比增长","同比下降","growth","increase","decrease","变动","变化"]
Y_REV_TRIG = ["revenue","net sales","营收","净销售额"]


# ======================================================================
# SECTION 1) 依赖导入 & 日志
# ======================================================================
try:
    from src.rag.retriever import hybrid_search
except Exception as e:
    raise ImportError(f"[answer_api] 无法导入 hybrid_search：{e}")

from .utils_detect import detect_query_type
from .hybrid_qa import answer_textual_or_mixed

logger = logging.getLogger("answer_api")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# ======================================================================
# SECTION 2) facts_numeric 懒加载
# ======================================================================
# ======================================================================
# SECTION 2) facts_numeric 懒加载（兼容缺列）
# ======================================================================
_FACTS_NUMERIC_DF = None  # pandas.DataFrame | None
_REQUIRED_FACT_COLS = [
    "ticker","form","accno","concept",
    "fy_norm","fq_norm","period_type",
    "value_std","unit_std",
    "period_start","period_end","instant",
    "page_no","source_path"
]

def _facts_df():
    """延迟加载 facts_numeric.parquet；若缺列则补 None，类型尽量规范化，确保 numeric 计算不炸。"""
    global _FACTS_NUMERIC_DF
    if _FACTS_NUMERIC_DF is not None:
        return _FACTS_NUMERIC_DF
    try:
        import pandas as pd
        if not os.path.exists(FACTS_NUMERIC_PATH):
            logger.warning(f"[answer_api] 未找到 {FACTS_NUMERIC_PATH}，数值计算将降级。")
            return None
        df = pd.read_parquet(FACTS_NUMERIC_PATH)

        # 补齐缺列（老数据也可用）
        for col in _REQUIRED_FACT_COLS:
            if col not in df.columns:
                df[col] = None

        # 规范 dtype（尽量不抛错）
        for c in ("fy_norm",):
            try: df[c] = df[c].astype("Int64")
            except Exception: pass
        for c in ("fq_norm","period_type","unit_std"):
            try: df[c] = df[c].astype("string")
            except Exception: pass
        # value_std 转浮点
        try:
            df["value_std"] = pd.to_numeric(df["value_std"], errors="coerce")
        except Exception:
            pass

        _FACTS_NUMERIC_DF = df
        return _FACTS_NUMERIC_DF
    except Exception as e:
        logger.warning(f"[answer_api] 加载/规范化 facts_numeric 失败：{e}")
        return None

def _dedupe_and_cap_hits(hits: List[Dict[str, Any]], topk: int = 8) -> List[Dict[str, Any]]:
    """对 hybrid_search 的命中做：按 chunk_id 去重 + 不同 file_type 分桶上限。"""
    seen = set()
    buckets = {}
    CAP = {"text": 3, "fact": 5, "cal": 2, "def": 2}
    out: List[Dict[str, Any]] = []
    for h in hits or []:
        cid = h.get("chunk_id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        meta = h.get("meta") or {}
        ft = (meta.get("file_type") or "text").lower()
        if buckets.get(ft, 0) >= CAP.get(ft, 99):
            continue
        buckets[ft] = buckets.get(ft, 0) + 1
        out.append(h)
        if len(out) >= topk:
            break
    return out


# ======================================================================
# SECTION 3) 检索辅助：GAAP 概念提取
# ======================================================================
GAAP_RE = re.compile(r"\bus-gaap:[A-Za-z0-9]+", re.I)

def _extract_gaap_candidates(hits: List[Dict[str, Any]], topn: int = 5) -> List[str]:
    cands: List[str] = []
    for h in hits or []:
        meta = (h.get("meta") or {})
        c = meta.get("concept")
        if isinstance(c, str) and c.lower().startswith("us-gaap:"):
            cands.append(c)
        txt = " ".join([h.get("text") or "", h.get("snippet") or ""])
        cands.extend(GAAP_RE.findall(txt))
        for _, v in meta.items():
            if isinstance(v, str) and v.lower().startswith("us-gaap:"):
                cands.append(v)
        if len(cands) >= topn:
            break
    # 去重保序
    seen, out = set(), []
    for x in cands:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out[:topn]


# ======================================================================
# SECTION 4) 工具：字符串/格式化/引用
# ======================================================================
def force_numeric(query: str) -> bool:
    q = (query or "").lower()
    return any(k in q for k in Y_NUM_TRIG) or any(k in q for k in Y_REV_TRIG)

def _norm_fq_str(fq) -> Optional[str]:
    if not fq: return None
    s = str(fq).upper().strip()
    if s in {"Q1","Q2","Q3","Q4"}: return s
    m = re.match(r".*Q([1-4]).*", s)
    return f"Q{m.group(1)}" if m else None

def _format_pct(x: float) -> str:
    return f"{x*100:.2f}%"

def _format_money(x: float) -> str:
    return f"${x:,.0f}"

def _citation_from_fact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_path": row.get("source_path"),
        "accno": row.get("accno"),
        "ticker": row.get("ticker"),
        "form": row.get("form"),
        "fy": row.get("fy_norm") or row.get("fy"),
        "fq": _norm_fq_str(row.get("fq_norm") or row.get("fq")),
        "section": None,
        "page": row.get("page_no"),
        "chunk_id": None,
        "lines": None,
        "concept": row.get("concept"),
    }

def _sort_citations_chronologically(cites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(c):
        fy = int(c.get("fy") or 0)
        fq = _norm_fq_str(c.get("fq"))
        qmap = {"Q1":1,"Q2":2,"Q3":3,"Q4":4,"FY":5,None:5}
        return (fy, qmap.get(fq, 5), int(c.get("page") or 10**9))
    return sorted(cites, key=key)

def _rows_to_citations(rows: List[Dict[str, Any]], hits: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    cits = [_citation_from_fact_row(r) for r in rows if r]
    cits = ensure_citations(cits, hits) or cits
    cits = _sort_citations_chronologically(cits)
    return cits, [citation_to_display(c) for c in cits]


# ======================================================================
# SECTION 5) 取数：单点/序列
# ======================================================================
def _select_fact(df, ticker: Optional[str], form: Optional[str], concept: str, fy: Optional[int], fq: Optional[str] = None):
    import pandas as pd  # noqa: F401
    q = (df["concept"] == concept)
    if ticker: q &= (df["ticker"] == ticker)
    if form:   q &= (df["form"] == form)
    if fy is not None:
        q &= (df["fy_norm"] == int(fy))
    if fq is not None:
        q &= (df["fq_norm"].astype(str).str.upper() == _norm_fq_str(fq))
    cand = df[q]
    if cand.empty: return None
    _pt = cand["period_type"].fillna("instant").astype(str).str.lower()
    rank = _pt.apply(lambda s: 0 if s == "duration" else 1)
    end_key = cand["period_end"].fillna(cand["instant"].fillna(""))
    cand = cand.assign(_rank=rank, _end=end_key).sort_values(by=["_rank","_end"], ascending=[True, True])
    return cand.tail(1).iloc[0].to_dict()

def _get_fact(df, ticker, form, concept, fy=None, fq=None):
    return _select_fact(df, ticker, form, concept, fy, fq)

def _get_quarter_series(df, ticker, form, concept, fy: int, fq: str, n: int=4) -> List[Dict[str, Any]]:
    order = ["Q1","Q2","Q3","Q4"]
    fq = _norm_fq_str(fq) or "Q4"
    idx = order.index(fq)
    out: List[Dict[str, Any]] = []
    y, i = int(fy), idx
    while len(out) < n:
        fqi = order[i]
        row = _get_fact(df, ticker, form, concept, y, fqi)
        if row: out.append(row)
        if i == 0: y, i = y-1, 3
        else: i -= 1
        if y < (fy-5): break  # 防炸
    return list(reversed(out))


# ======================================================================
# SECTION 6) 计算：YoY / QoQ / diff / raw / TTM / TTM YoY / CAGR / ratio / FCF / EPS
# ======================================================================
def _compute_yoy_rows(df, ticker, form, concept, fy: int):
    cur = _get_fact(df, ticker, form, concept, fy)
    prv = _get_fact(df, ticker, form, concept, fy-1)
    if not cur or not prv: return None, [], "数据不足"
    v_now = float(cur["value_std"]); v_prev = float(prv["value_std"])
    if v_prev == 0: return None, [cur,prv], "分母为 0"
    return (v_now - v_prev) / v_prev, [prv, cur], None

def _compute_qoq_rows(df, ticker, form, concept, fy: int, fq: str):
    order = ["Q1","Q2","Q3","Q4"]
    fq = _norm_fq_str(fq) or "Q4"
    i = order.index(fq)
    fy_prev, fq_prev = (fy-1, "Q4") if i == 0 else (fy, order[i-1])
    cur = _get_fact(df, ticker, form, concept, fy, fq)
    prv = _get_fact(df, ticker, form, concept, fy_prev, fq_prev)
    if not cur or not prv: return None, [], "数据不足"
    v_now = float(cur["value_std"]); v_prev = float(prv["value_std"])
    if v_prev == 0: return None, [cur,prv], "分母为 0"
    return (v_now - v_prev) / v_prev, [prv, cur], None

def _compute_diff_rows(df, ticker, form, concept, fy: int, fy_prev: Optional[int] = None):
    fy_prev = fy-1 if fy_prev is None else fy_prev
    cur = _get_fact(df, ticker, form, concept, fy)
    prv = _get_fact(df, ticker, form, concept, fy_prev)
    if not cur or not prv: return None, [], "数据不足"
    return float(cur["value_std"]) - float(prv["value_std"]), [prv, cur], None

def _compute_raw_row(df, ticker, form, concept, fy: Optional[int], fq: Optional[str]):
    row = _get_fact(df, ticker, form, concept, fy, fq)
    if not row: return None, [], "数据不足"
    return float(row["value_std"]), [row], None

def _compute_ttm(df, ticker, form, concept, fy: int, fq: str):
    qs = _get_quarter_series(df, ticker, form, concept, fy, fq, n=4)
    if len(qs) < 4: return None, [], "TTM 需要连续 4 个季度"
    return sum(float(x["value_std"]) for x in qs), qs, None

def _compute_ttm_yoy(df, ticker, form, concept, fy: int, fq: str):
    v_now, rows_now, err = _compute_ttm(df, ticker, form, concept, fy, fq)
    fq_prev = _norm_fq_str(fq) or "Q4"
    v_prev, rows_prev, err2 = _compute_ttm(df, ticker, form, concept, fy-1, fq_prev)
    if err or err2 or v_now is None or v_prev in (None, 0): return None, rows_now+rows_prev, err or err2 or "分母为 0"
    return (v_now - v_prev) / v_prev, rows_now+rows_prev, None

def _cagr(v_start: float, v_end: float, years: int):
    if v_start is None or v_end is None or v_start <= 0: return None
    try: return (v_end / v_start) ** (1/years) - 1
    except Exception: return None

def _compute_cagr(df, ticker, form, concept, fy_end: int, years: int):
    row_end = _get_fact(df, ticker, form, concept, fy_end)
    row_start = _get_fact(df, ticker, form, concept, fy_end - years)
    if not row_end or not row_start: return None, [], "数据不足"
    val = _cagr(float(row_start["value_std"]), float(row_end["value_std"]), years)
    if val is None: return None, [row_start, row_end], "无效分母或数据"
    return val, [row_start, row_end], None

def _compute_ratio(df, ticker, form, num_concept, den_concept, fy=None, fq=None, ttm=False):
    if ttm and fy is not None and fq:
        vn, rowsN, err1 = _compute_ttm(df, ticker, form, num_concept, fy, fq)
        vd, rowsD, err2 = _compute_ttm(df, ticker, form, den_concept, fy, fq)
        rows = rowsN + rowsD
    else:
        rowN = _get_fact(df, ticker, form, num_concept, fy, fq)
        rowD = _get_fact(df, ticker, form, den_concept, fy, fq)
        rows = [r for r in (rowN, rowD) if r]
        vn = float(rowN["value_std"]) if rowN else None
        vd = float(rowD["value_std"]) if rowD else None
        err1 = err2 = None
    if err1 or err2: return None, rows, err1 or err2
    if vn is None or vd in (None, 0): return None, rows, "数据不足或分母为 0"
    return vn / vd, rows, None

def _compute_fcf(df, ticker, form, fy=None, fq=None, ttm=False):
    if ttm and fy is not None and fq:
        vCFO, rowsCFO, err1 = _compute_ttm(df, ticker, form, GAAP["cfo"], fy, fq)
        vCap, rowsCap, err2 = _compute_ttm(df, ticker, form, GAAP["capex"], fy, fq)
        rows = rowsCFO + rowsCap
    else:
        rowCFO = _get_fact(df, ticker, form, GAAP["cfo"], fy, fq)
        rowCap = _get_fact(df, ticker, form, GAAP["capex"], fy, fq)
        rows = [r for r in (rowCFO, rowCap) if r]
        vCFO = float(rowCFO["value_std"]) if rowCFO else None
        vCap = float(rowCap["value_std"]) if rowCap else None
        err1 = err2 = None
    if err1 or err2: return None, rows, err1 or err2
    if vCFO is None or vCap is None: return None, rows, "数据不足"
    return vCFO - vCap, rows, None

def _compute_eps(df, ticker, form, diluted: bool, fy=None, fq=None, yoy=False):
    concept = GAAP["eps_diluted"] if diluted else GAAP["eps_basic"]
    if yoy and fy is not None:
        cur = _get_fact(df, ticker, form, concept, fy, fq)
        prv = _get_fact(df, ticker, form, concept, fy-1, fq)
        if not cur or not prv: return None, [], "数据不足"
        vC, vP = float(cur["value_std"]), float(prv["value_std"])
        if vP == 0: return None, [prv, cur], "分母为 0"
        return (vC - vP)/vP, [prv, cur], None
    else:
        row = _get_fact(df, ticker, form, concept, fy, fq)
        if not row: return None, [], "数据不足"
        return float(row["value_std"]), [row], None

# 可能遇到“revenue”概念多种命名，给一些别名候选
_REVENUE_ALIASES = [
    "us-gaap:SalesRevenueNet",
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap:SalesRevenueGoodsNet",
    "us-gaap:SalesRevenueServicesNet",
    "us-gaap:Revenues",
]

def _latest_fy_fq(df, ticker: Optional[str], form: Optional[str]):
    """在 facts_numeric 里找该 ticker/form 的最新财年与季度（兜底）"""
    try:
        q = (df["ticker"] == ticker) if ticker else (df["ticker"].notna())
        if form:
            q &= (df["form"] == form)
        sub = df[q][["fy_norm","fq_norm","period_end","instant"]].dropna(how="all")
        if sub.empty:
            return None, None
        sub = sub.sort_values(["fy_norm","fq_norm","period_end","instant"])
        r = sub.tail(1).iloc[0]
        return int(r.get("fy_norm")), (str(r.get("fq_norm")) if r.get("fq_norm") else None)
    except Exception:
        return None, None

def _relax_numeric_attempts(df, ticker, form, concept, fy, fq, hits):
    """
    给 numeric 计算做逐级放宽的重试：
    1) 原条件；
    2) 放宽 form（10-K/10-Q 任意）；
    3) 放宽概念到别名列表（仅 revenue 类）；
    4) 若 fy 缺失，则使用最新 fy/fq；
    5) 若请求 YoY/QoQ 但缺对比期，则自动推前一年/上一季。
    返回: (best_rows, best_reason_note)
    """
    notes = []

    def try_get(op: str, cpt: str, y, q):
        # 只尝试 raw 取数，后续由上层决定算 yoy/qoq/ttm 等
        row = _get_fact(df, ticker, form, cpt, y, q)
        if row:
            return [row], f"hit:{op}|concept={cpt}|fy={y}|fq={q}"
        return None, f"miss:{op}|concept={cpt}|fy={y}|fq={q}"

    # 0) 入参兜底 fy/fq
    if fy is None:
        _fy,_fq = _latest_fy_fq(df, ticker, form)
        if _fy:
            fy = _fy; fq = fq or _fq
            notes.append(f"fill-latest fy/fq → {fy}/{fq or 'NA'}")

    # 1) 原条件
    rows, note = try_get("orig", concept, fy, fq)
    notes.append(note)
    if rows: return rows, "; ".join(notes)

    # 2) 放宽 form
    if form:
        rows, note = try_get("relax-form", concept, fy, fq)
        notes.append(note)
        if rows: return rows, "; ".join(notes)

    # 3) 概念别名（只对 revenue 有意义）
    if concept in _REVENUE_ALIASES or concept == "us-gaap:SalesRevenueNet":
        for c in _REVENUE_ALIASES:
            rows, note = try_get("alias", c, fy, fq)
            notes.append(note)
            if rows: return rows, "; ".join(notes)

    # 4) 若 fq 季度缺，试 FY；若 FY 缺，试 hit 推断
    if fq:
        rows, note = try_get("drop-fq", concept, fy, None)
        notes.append(note)
        if rows: return rows, "; ".join(notes)

    # 5) 从 hits 抽 GAAP 候选再试
    for c in _extract_gaap_candidates(hits, topn=5):
        rows, note = try_get("gaap-from-hits", c, fy, fq)
        notes.append(note)
        if rows: return rows, "; ".join(notes)

    return None, "; ".join(notes)  # 全部失败

# ======================================================================
# SECTION 7) 规则解析：numeric 意图
# ======================================================================
def _contains(q: str, keys: List[str]) -> bool:
    ql = q.lower()
    return any(k in ql for k in keys)

def _infer_numeric_case(query: str, hits: List[Dict[str, Any]], filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    返回结构化意图：
    { "op": "yoy|qoq|diff|raw|ttm|ttm_yoy|cagr|margin|expense_ratio|fcf|fcf_ttm|eps|eps_yoy",
      "args": {...} }
    """
    q = (query or "").lower()
    fy     = filters.get("year") or filters.get("fy")
    fq     = filters.get("fq")

    # 从 hits 兜底 fy/fq
    if fy is None:
        for h in hits or []:
            m = (h.get("meta") or {})
            fy = m.get("fy") or m.get("fy_norm") or fy
            if fy: break
    if fq is None:
        for h in hits or []:
            m = (h.get("meta") or {})
            fq = m.get("fq") or m.get("fq_norm") or fq
            if fq: break
    if fq: fq = _norm_fq_str(fq)

    # 概念默认 revenue；命中 net_income/cash 优先；hits 里如有 us-gaap 更优先
    concept = GAAP["revenue"]
    if _contains(q, TRIG["net_income"]): concept = GAAP["net_income"]
    elif _contains(q, TRIG["cash"]):     concept = GAAP["cash"]
    elif _contains(q, TRIG["revenue"]):  concept = GAAP["revenue"]
    for h in hits or []:
        c = (h.get("meta") or {}).get("concept")
        if isinstance(c, str) and c.lower().startswith("us-gaap"):
            concept = c; break

    is_ttm   = _contains(q, TRIG["ttm"])
    is_cagr  = _contains(q, TRIG["cagr"])
    is_yoy   = _contains(q, TRIG["yoy"])
    is_qoq   = _contains(q, TRIG["qoq"])
    is_diff  = _contains(q, TRIG["diff"])
    is_eps   = _contains(q, TRIG["eps"])
    eps_dil  = _contains(q, TRIG["diluted"])
    eps_bas  = _contains(q, TRIG["basic"])

    if _contains(q, TRIG["gross_margin"]):
        return {"op":"margin", "args":{"num": GAAP["gross_profit"], "den": GAAP["revenue"], "fy": fy, "fq": fq, "ttm": is_ttm}}
    if _contains(q, TRIG["op_margin"]):
        return {"op":"margin", "args":{"num": GAAP["operating_income"], "den": GAAP["revenue"], "fy": fy, "fq": fq, "ttm": is_ttm}}
    if _contains(q, TRIG["net_margin"]):
        return {"op":"margin", "args":{"num": GAAP["net_income"], "den": GAAP["revenue"], "fy": fy, "fq": fq, "ttm": is_ttm}}
    if _contains(q, TRIG["rd_ratio"]):
        return {"op":"expense_ratio", "args":{"num": GAAP["r_and_d"], "den": GAAP["revenue"], "fy": fy, "fq": fq, "ttm": is_ttm}}
    if _contains(q, TRIG["sga_ratio"]):
        return {"op":"expense_ratio", "args":{"num": GAAP["sga"], "den": GAAP["revenue"], "fy": fy, "fq": fq, "ttm": is_ttm}}
    if _contains(q, TRIG["fcf"]):
        return {"op":"fcf_ttm" if is_ttm else "fcf", "args":{"fy": fy, "fq": fq}}

    if is_eps:
        diluted = True if eps_dil or not eps_bas else False
        if is_yoy:
            return {"op":"eps_yoy", "args":{"diluted": diluted, "fy": fy, "fq": fq}}
        else:
            return {"op":"eps", "args":{"diluted": diluted, "fy": fy, "fq": fq}}

    if is_cagr:
        m = re.search(r"(\d+)[-\s]*year", q)
        years = int(m.group(1)) if m else 3
        return {"op":"cagr", "args":{"concept": concept, "fy": fy, "years": years}}

    if is_ttm and fy and fq:
        return {"op":"ttm", "args":{"concept": concept, "fy": fy, "fq": fq}}
    if is_yoy and fy:
        return {"op":"yoy", "args":{"concept": concept, "fy": fy}}
    if is_qoq and fy and fq:
        return {"op":"qoq", "args":{"concept": concept, "fy": fy, "fq": fq}}
    if is_diff and fy:
        return {"op":"diff", "args":{"concept": concept, "fy": fy}}

    return {"op":"raw", "args":{"concept": concept, "fy": fy, "fq": fq}}


# ======================================================================
# SECTION 8) Numeric 作答主流程（统一调度）
# ======================================================================
def _numeric_answer_pipeline(
    query: str,
    hits: List[Dict[str, Any]],
    filters: Dict[str, Any],
) -> Tuple[Optional[str], str, List[Dict[str, Any]], List[str]]:

    df = _facts_df()
    notes = []
    if df is None:
        return None, "数值视图缺失，无法计算。（facts_numeric 未加载）", [], []

    # 关键列检查
    need_cols = {"ticker","form","concept","value_std","fy_norm"}
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        notes.append(f"facts_numeric 缺列: {miss}")

    ticker = filters.get("ticker")
    form   = filters.get("form")
    intent = _infer_numeric_case(query, hits, filters)
    op     = intent["op"]
    args   = intent["args"]

    # 如果问题涉及“营收”，尝试用数据驱动选最合适的营收概念
    ql = (query or "").lower()
    is_revenue_q = ("revenue" in ql) or ("net sales" in ql) or ("营收" in ql) or ("净销售额" in ql)
    concept = args.get("concept")
    if is_revenue_q or concept in _REVENUE_ALIASES:
        best = _pick_best_revenue_concept(df, ticker, form, args.get("fy"))
        if best and best != concept:
            notes.append(f"revenue concept → {best}")
            args["concept"] = best

    try:
        if op == "yoy":
            concept, fy = args["concept"], int(args["fy"])
            v, rows, err = _compute_yoy_rows(df, ticker, form, concept, fy)
            if v is None: 
                 # 走自救链：先拿到“当前期”的行，再自己拼上一年前的行去算同比
                rows_now, note = _relax_numeric_attempts(df, ticker, form, concept, fy, args.get("fq"), hits)
                if rows_now:
                    fy_now = int(rows_now[-1].get("fy_norm") or rows_now[-1].get("fy"))
                    cur = rows_now[-1]
                    prv = _get_fact(df, ticker, form, concept, fy_now-1, _norm_fq_str(cur.get("fq_norm") or cur.get("fq")))
                    if prv and cur and float(prv.get("value_std") or 0) != 0:
                        v = (float(cur["value_std"]) - float(prv["value_std"])) / float(prv["value_std"])
                        rows = [prv, cur]; err = None
                        # 继续正常返回
                        cits, disp = _rows_to_citations(rows, hits)
                        unit = cur.get("unit_std") or "money"
                        ans = f"YoY：{_format_pct(v)}（{fy_now-1}→{fy_now}：{_format_money(prv['value_std']) if unit=='money' else f'{prv['value_std']:,}'} → {_format_money(cur['value_std']) if unit=='money' else f'{cur['value_std']:,}'}）。"
                        rsn = f"自救链成功：{note}"
                        return ans, rsn, cits, disp
                # 自救失败
                return None, f"信息不足：{err}" + (f" | {'; '.join(notes)}" if notes else ""), [], []
            cits, disp = _rows_to_citations(rows, hits)
            cur, prv = rows[-1], rows[0]
            unit = cur.get("unit_std") or "money"
            ans = f"YoY：{_format_pct(v)}（{fy-1}→{fy}：{_format_money(prv['value_std']) if unit=='money' else prv['value_std']:,} → {_format_money(cur['value_std']) if unit=='money' else cur['value_std']:,}）。"
            rsn = f"概念={concept}；ticker={ticker or 'NA'}；form={form or 'NA'}；fy={fy}。"
            return ans, rsn, cits, disp

        if op == "qoq":
            concept, fy, fq = args["concept"], int(args["fy"]), args["fq"]
            v, rows, err = _compute_qoq_rows(df, ticker, form, concept, fy, fq)
            if v is None: 
                 # 走自救链：先拿到“当前期”的行，再自己拼上一年前的行去算同比
                rows_now, note = _relax_numeric_attempts(df, ticker, form, concept, fy, args.get("fq"), hits)
                if rows_now:
                    fy_now = int(rows_now[-1].get("fy_norm") or rows_now[-1].get("fy"))
                    cur = rows_now[-1]
                    prv = _get_fact(df, ticker, form, concept, fy_now-1, _norm_fq_str(cur.get("fq_norm") or cur.get("fq")))
                    if prv and cur and float(prv.get("value_std") or 0) != 0:
                        v = (float(cur["value_std"]) - float(prv["value_std"])) / float(prv["value_std"])
                        rows = [prv, cur]; err = None
                        # 继续正常返回
                        cits, disp = _rows_to_citations(rows, hits)
                        unit = cur.get("unit_std") or "money"
                        ans = f"YoY：{_format_pct(v)}（{fy_now-1}→{fy_now}：{_format_money(prv['value_std']) if unit=='money' else f'{prv['value_std']:,}'} → {_format_money(cur['value_std']) if unit=='money' else f'{cur['value_std']:,}'}）。"
                        rsn = f"自救链成功：{note}"
                        return ans, rsn, cits, disp
                # 自救失败
                return None, f"信息不足：{err}" + (f" | {'; '.join(notes)}" if notes else ""), [], []
            cits, disp = _rows_to_citations(rows, hits)
            ans = f"QoQ：{_format_pct(v)}。"
            rsn = f"概念={concept}；ticker={ticker or 'NA'}；form={form or 'NA'}；fy={fy}, fq={fq}。"
            return ans, rsn, cits, disp

        if op == "diff":
            concept, fy = args["concept"], int(args["fy"])
            v, rows, err = _compute_diff_rows(df, ticker, form, concept, fy)
            if v is None: 
                 # 走自救链：先拿到“当前期”的行，再自己拼上一年前的行去算同比
                rows_now, note = _relax_numeric_attempts(df, ticker, form, concept, fy, args.get("fq"), hits)
                if rows_now:
                    fy_now = int(rows_now[-1].get("fy_norm") or rows_now[-1].get("fy"))
                    cur = rows_now[-1]
                    prv = _get_fact(df, ticker, form, concept, fy_now-1, _norm_fq_str(cur.get("fq_norm") or cur.get("fq")))
                    if prv and cur and float(prv.get("value_std") or 0) != 0:
                        v = (float(cur["value_std"]) - float(prv["value_std"])) / float(prv["value_std"])
                        rows = [prv, cur]; err = None
                        # 继续正常返回
                        cits, disp = _rows_to_citations(rows, hits)
                        unit = cur.get("unit_std") or "money"
                        ans = f"YoY：{_format_pct(v)}（{fy_now-1}→{fy_now}：{_format_money(prv['value_std']) if unit=='money' else f'{prv['value_std']:,}'} → {_format_money(cur['value_std']) if unit=='money' else f'{cur['value_std']:,}'}）。"
                        rsn = f"自救链成功：{note}"
                        return ans, rsn, cits, disp
                # 自救失败
                return None, f"信息不足：{err}" + (f" | {'; '.join(notes)}" if notes else ""), [], []
            unit = rows[-1].get("unit_std") or "money"
            cits, disp = _rows_to_citations(rows, hits)
            ans = f"差额：{_format_money(v) if unit=='money' else f'{v:,.4g}'}。"
            rsn = f"概念={concept}；ticker={ticker or 'NA'}；form={form or 'NA'}；FY 对 FY-1。"
            return ans, rsn, cits, disp

        if op == "raw":
            concept, fy, fq = args["concept"], args.get("fy"), args.get("fq")
            v, rows, err = _compute_raw_row(df, ticker, form, concept, fy, fq)
            if v is None: 
                 # 走自救链：先拿到“当前期”的行，再自己拼上一年前的行去算同比
                rows_now, note = _relax_numeric_attempts(df, ticker, form, concept, fy, args.get("fq"), hits)
                if rows_now:
                    fy_now = int(rows_now[-1].get("fy_norm") or rows_now[-1].get("fy"))
                    cur = rows_now[-1]
                    prv = _get_fact(df, ticker, form, concept, fy_now-1, _norm_fq_str(cur.get("fq_norm") or cur.get("fq")))
                    if prv and cur and float(prv.get("value_std") or 0) != 0:
                        v = (float(cur["value_std"]) - float(prv["value_std"])) / float(prv["value_std"])
                        rows = [prv, cur]; err = None
                        # 继续正常返回
                        cits, disp = _rows_to_citations(rows, hits)
                        unit = cur.get("unit_std") or "money"
                        ans = f"YoY：{_format_pct(v)}（{fy_now-1}→{fy_now}：{_format_money(prv['value_std']) if unit=='money' else f'{prv['value_std']:,}'} → {_format_money(cur['value_std']) if unit=='money' else f'{cur['value_std']:,}'}）。"
                        rsn = f"自救链成功：{note}"
                        return ans, rsn, cits, disp
                # 自救失败
                return None, f"信息不足：{err}" + (f" | {'; '.join(notes)}" if notes else ""), [], []
            unit = rows[-1].get("unit_std") or "money"
            cits, disp = _rows_to_citations(rows, hits)
            period = rows[-1].get("period_end") or rows[-1].get("instant")
            ans = f"{concept} = {_format_money(v) if unit=='money' else f'{v:,.4g}'}（期末/日期：{period}）。"
            rsn = f"概念={concept}；ticker={ticker or 'NA'}；form={form or 'NA'}。"
            return ans, rsn, cits, disp

        if op == "ttm":
            concept, fy, fq = args["concept"], int(args["fy"]), args["fq"]
            v, rows, err = _compute_ttm(df, ticker, form, concept, fy, fq)
            if v is None: 
                 # 走自救链：先拿到“当前期”的行，再自己拼上一年前的行去算同比
                rows_now, note = _relax_numeric_attempts(df, ticker, form, concept, fy, args.get("fq"), hits)
                if rows_now:
                    fy_now = int(rows_now[-1].get("fy_norm") or rows_now[-1].get("fy"))
                    cur = rows_now[-1]
                    prv = _get_fact(df, ticker, form, concept, fy_now-1, _norm_fq_str(cur.get("fq_norm") or cur.get("fq")))
                    if prv and cur and float(prv.get("value_std") or 0) != 0:
                        v = (float(cur["value_std"]) - float(prv["value_std"])) / float(prv["value_std"])
                        rows = [prv, cur]; err = None
                        # 继续正常返回
                        cits, disp = _rows_to_citations(rows, hits)
                        unit = cur.get("unit_std") or "money"
                        ans = f"YoY：{_format_pct(v)}（{fy_now-1}→{fy_now}：{_format_money(prv['value_std']) if unit=='money' else f'{prv['value_std']:,}'} → {_format_money(cur['value_std']) if unit=='money' else f'{cur['value_std']:,}'}）。"
                        rsn = f"自救链成功：{note}"
                        return ans, rsn, cits, disp
                # 自救失败
                return None, f"信息不足：{err}" + (f" | {'; '.join(notes)}" if notes else ""), [], []
            unit = rows[-1].get("unit_std") or "money"
            cits, disp = _rows_to_citations(rows, hits)
            ans = f"TTM（{concept}）= {_format_money(v) if unit=='money' else f'{v:,.4g}'}。"
            rsn = f"最近 4 季合计；ticker={ticker or 'NA'}；form={form or 'NA'}；截至 {fy} {fq}。"
            return ans, rsn, cits, disp

        if op == "ttm_yoy":
            concept, fy, fq = args["concept"], int(args["fy"]), args["fq"]
            v, rows, err = _compute_ttm_yoy(df, ticker, form, concept, fy, fq)
            if v is None: 
                 # 走自救链：先拿到“当前期”的行，再自己拼上一年前的行去算同比
                rows_now, note = _relax_numeric_attempts(df, ticker, form, concept, fy, args.get("fq"), hits)
                if rows_now:
                    fy_now = int(rows_now[-1].get("fy_norm") or rows_now[-1].get("fy"))
                    cur = rows_now[-1]
                    prv = _get_fact(df, ticker, form, concept, fy_now-1, _norm_fq_str(cur.get("fq_norm") or cur.get("fq")))
                    if prv and cur and float(prv.get("value_std") or 0) != 0:
                        v = (float(cur["value_std"]) - float(prv["value_std"])) / float(prv["value_std"])
                        rows = [prv, cur]; err = None
                        # 继续正常返回
                        cits, disp = _rows_to_citations(rows, hits)
                        unit = cur.get("unit_std") or "money"
                        ans = f"YoY：{_format_pct(v)}（{fy_now-1}→{fy_now}：{_format_money(prv['value_std']) if unit=='money' else f'{prv['value_std']:,}'} → {_format_money(cur['value_std']) if unit=='money' else f'{cur['value_std']:,}'}）。"
                        rsn = f"自救链成功：{note}"
                        return ans, rsn, cits, disp
                # 自救失败
                return None, f"信息不足：{err}" + (f" | {'; '.join(notes)}" if notes else ""), [], []
            cits, disp = _rows_to_citations(rows, hits)
            ans = f"TTM YoY：{_format_pct(v)}。"
            rsn = f"最近 4 季 vs 去年同期 4 季；ticker={ticker or 'NA'}；form={form or 'NA'}；截至 {fy} {fq}。"
            return ans, rsn, cits, disp

        if op == "cagr":
            concept, fy, years = args["concept"], int(args["fy"]), int(args["years"])
            v, rows, err = _compute_cagr(df, ticker, form, concept, fy, years)
            if v is None: 
                 # 走自救链：先拿到“当前期”的行，再自己拼上一年前的行去算同比
                rows_now, note = _relax_numeric_attempts(df, ticker, form, concept, fy, args.get("fq"), hits)
                if rows_now:
                    fy_now = int(rows_now[-1].get("fy_norm") or rows_now[-1].get("fy"))
                    cur = rows_now[-1]
                    prv = _get_fact(df, ticker, form, concept, fy_now-1, _norm_fq_str(cur.get("fq_norm") or cur.get("fq")))
                    if prv and cur and float(prv.get("value_std") or 0) != 0:
                        v = (float(cur["value_std"]) - float(prv["value_std"])) / float(prv["value_std"])
                        rows = [prv, cur]; err = None
                        # 继续正常返回
                        cits, disp = _rows_to_citations(rows, hits)
                        unit = cur.get("unit_std") or "money"
                        ans = f"YoY：{_format_pct(v)}（{fy_now-1}→{fy_now}：{_format_money(prv['value_std']) if unit=='money' else f'{prv['value_std']:,}'} → {_format_money(cur['value_std']) if unit=='money' else f'{cur['value_std']:,}'}）。"
                        rsn = f"自救链成功：{note}"
                        return ans, rsn, cits, disp
                # 自救失败
                return None, f"信息不足：{err}" + (f" | {'; '.join(notes)}" if notes else ""), [], []
            cits, disp = _rows_to_citations(rows, hits)
            ans = f"{years} 年 CAGR：{_format_pct(v)}。"
            rsn = f"{fy-years}→{fy}；概念={concept}；ticker={ticker or 'NA'}；form={form or 'NA'}。"
            return ans, rsn, cits, disp

        if op == "margin":
            num, den, fy, fq, ttm = args["num"], args["den"], args.get("fy"), args.get("fq"), bool(args.get("ttm"))
            v, rows, err = _compute_ratio(df, ticker, form, num, den, fy=fy, fq=fq, ttm=ttm)
            if v is None: 
                 # 走自救链：先拿到“当前期”的行，再自己拼上一年前的行去算同比
                rows_now, note = _relax_numeric_attempts(df, ticker, form, concept, fy, args.get("fq"), hits)
                if rows_now:
                    fy_now = int(rows_now[-1].get("fy_norm") or rows_now[-1].get("fy"))
                    cur = rows_now[-1]
                    prv = _get_fact(df, ticker, form, concept, fy_now-1, _norm_fq_str(cur.get("fq_norm") or cur.get("fq")))
                    if prv and cur and float(prv.get("value_std") or 0) != 0:
                        v = (float(cur["value_std"]) - float(prv["value_std"])) / float(prv["value_std"])
                        rows = [prv, cur]; err = None
                        # 继续正常返回
                        cits, disp = _rows_to_citations(rows, hits)
                        unit = cur.get("unit_std") or "money"
                        ans = f"YoY：{_format_pct(v)}（{fy_now-1}→{fy_now}：{_format_money(prv['value_std']) if unit=='money' else f'{prv['value_std']:,}'} → {_format_money(cur['value_std']) if unit=='money' else f'{cur['value_std']:,}'}）。"
                        rsn = f"自救链成功：{note}"
                        return ans, rsn, cits, disp
                # 自救失败
                return None, f"信息不足：{err}" + (f" | {'; '.join(notes)}" if notes else ""), [], []
            cits, disp = _rows_to_citations(rows, hits)
            ans = f"利润率：{_format_pct(v)}。"
            rsn = f"{num}/{den}；{'TTM' if ttm else '单期'}；ticker={ticker or 'NA'}；form={form or 'NA'}。"
            return ans, rsn, cits, disp

        if op == "expense_ratio":
            num, den, fy, fq, ttm = args["num"], args["den"], args.get("fy"), args.get("fq"), bool(args.get("ttm"))
            v, rows, err = _compute_ratio(df, ticker, form, num, den, fy=fy, fq=fq, ttm=ttm)
            if v is None: 
                 # 走自救链：先拿到“当前期”的行，再自己拼上一年前的行去算同比
                rows_now, note = _relax_numeric_attempts(df, ticker, form, concept, fy, args.get("fq"), hits)
                if rows_now:
                    fy_now = int(rows_now[-1].get("fy_norm") or rows_now[-1].get("fy"))
                    cur = rows_now[-1]
                    prv = _get_fact(df, ticker, form, concept, fy_now-1, _norm_fq_str(cur.get("fq_norm") or cur.get("fq")))
                    if prv and cur and float(prv.get("value_std") or 0) != 0:
                        v = (float(cur["value_std"]) - float(prv["value_std"])) / float(prv["value_std"])
                        rows = [prv, cur]; err = None
                        # 继续正常返回
                        cits, disp = _rows_to_citations(rows, hits)
                        unit = cur.get("unit_std") or "money"
                        ans = f"YoY：{_format_pct(v)}（{fy_now-1}→{fy_now}：{_format_money(prv['value_std']) if unit=='money' else f'{prv['value_std']:,}'} → {_format_money(cur['value_std']) if unit=='money' else f'{cur['value_std']:,}'}）。"
                        rsn = f"自救链成功：{note}"
                        return ans, rsn, cits, disp
                # 自救失败
                return None, f"信息不足：{err}" + (f" | {'; '.join(notes)}" if notes else ""), [], []
            cits, disp = _rows_to_citations(rows, hits)
            ans = f"费用占比：{_format_pct(v)}。"
            rsn = f"{num}/{den}；{'TTM' if ttm else '单期'}；ticker={ticker or 'NA'}；form={form or 'NA'}。"
            return ans, rsn, cits, disp

        if op == "fcf":
            fy, fq = args.get("fy"), args.get("fq")
            v, rows, err = _compute_fcf(df, ticker, form, fy=fy, fq=fq, ttm=False)
            if v is None: 
                 # 走自救链：先拿到“当前期”的行，再自己拼上一年前的行去算同比
                rows_now, note = _relax_numeric_attempts(df, ticker, form, concept, fy, args.get("fq"), hits)
                if rows_now:
                    fy_now = int(rows_now[-1].get("fy_norm") or rows_now[-1].get("fy"))
                    cur = rows_now[-1]
                    prv = _get_fact(df, ticker, form, concept, fy_now-1, _norm_fq_str(cur.get("fq_norm") or cur.get("fq")))
                    if prv and cur and float(prv.get("value_std") or 0) != 0:
                        v = (float(cur["value_std"]) - float(prv["value_std"])) / float(prv["value_std"])
                        rows = [prv, cur]; err = None
                        # 继续正常返回
                        cits, disp = _rows_to_citations(rows, hits)
                        unit = cur.get("unit_std") or "money"
                        ans = f"YoY：{_format_pct(v)}（{fy_now-1}→{fy_now}：{_format_money(prv['value_std']) if unit=='money' else f'{prv['value_std']:,}'} → {_format_money(cur['value_std']) if unit=='money' else f'{cur['value_std']:,}'}）。"
                        rsn = f"自救链成功：{note}"
                        return ans, rsn, cits, disp
                # 自救失败
                return None, f"信息不足：{err}" + (f" | {'; '.join(notes)}" if notes else ""), [], []
            cits, disp = _rows_to_citations(rows, hits)
            ans = f"FCF = {_format_money(v)}。"
            rsn = f"CFO - Capex；ticker={ticker or 'NA'}；form={form or 'NA'}。"
            return ans, rsn, cits, disp

        if op == "fcf_ttm":
            fy, fq = int(args["fy"]), args["fq"]
            v, rows, err = _compute_fcf(df, ticker, form, fy=fy, fq=fq, ttm=True)
            if v is None: 
                 # 走自救链：先拿到“当前期”的行，再自己拼上一年前的行去算同比
                rows_now, note = _relax_numeric_attempts(df, ticker, form, concept, fy, args.get("fq"), hits)
                if rows_now:
                    fy_now = int(rows_now[-1].get("fy_norm") or rows_now[-1].get("fy"))
                    cur = rows_now[-1]
                    prv = _get_fact(df, ticker, form, concept, fy_now-1, _norm_fq_str(cur.get("fq_norm") or cur.get("fq")))
                    if prv and cur and float(prv.get("value_std") or 0) != 0:
                        v = (float(cur["value_std"]) - float(prv["value_std"])) / float(prv["value_std"])
                        rows = [prv, cur]; err = None
                        # 继续正常返回
                        cits, disp = _rows_to_citations(rows, hits)
                        unit = cur.get("unit_std") or "money"
                        ans = f"YoY：{_format_pct(v)}（{fy_now-1}→{fy_now}：{_format_money(prv['value_std']) if unit=='money' else f'{prv['value_std']:,}'} → {_format_money(cur['value_std']) if unit=='money' else f'{cur['value_std']:,}'}）。"
                        rsn = f"自救链成功：{note}"
                        return ans, rsn, cits, disp
                # 自救失败
                return None, f"信息不足：{err}" + (f" | {'; '.join(notes)}" if notes else ""), [], []
            cits, disp = _rows_to_citations(rows, hits)
            ans = f"FCF（TTM）= {_format_money(v)}。"
            rsn = f"TTM CFO - TTM Capex；ticker={ticker or 'NA'}；form={form or 'NA'}；截至 {fy} {fq}。"
            return ans, rsn, cits, disp

        if op == "eps":
            diluted, fy, fq = bool(args["diluted"]), args.get("fy"), args.get("fq")
            v, rows, err = _compute_eps(df, ticker, form, diluted=diluted, fy=fy, fq=fq, yoy=False)
            if v is None: 
                 # 走自救链：先拿到“当前期”的行，再自己拼上一年前的行去算同比
                rows_now, note = _relax_numeric_attempts(df, ticker, form, concept, fy, args.get("fq"), hits)
                if rows_now:
                    fy_now = int(rows_now[-1].get("fy_norm") or rows_now[-1].get("fy"))
                    cur = rows_now[-1]
                    prv = _get_fact(df, ticker, form, concept, fy_now-1, _norm_fq_str(cur.get("fq_norm") or cur.get("fq")))
                    if prv and cur and float(prv.get("value_std") or 0) != 0:
                        v = (float(cur["value_std"]) - float(prv["value_std"])) / float(prv["value_std"])
                        rows = [prv, cur]; err = None
                        # 继续正常返回
                        cits, disp = _rows_to_citations(rows, hits)
                        unit = cur.get("unit_std") or "money"
                        ans = f"YoY：{_format_pct(v)}（{fy_now-1}→{fy_now}：{_format_money(prv['value_std']) if unit=='money' else f'{prv['value_std']:,}'} → {_format_money(cur['value_std']) if unit=='money' else f'{cur['value_std']:,}'}）。"
                        rsn = f"自救链成功：{note}"
                        return ans, rsn, cits, disp
                # 自救失败
                return None, f"信息不足：{err}" + (f" | {'; '.join(notes)}" if notes else ""), [], []
            cits, disp = _rows_to_citations(rows, hits)
            flavor = "Diluted" if diluted else "Basic"
            ans = f"EPS ({flavor}) = {v:.2f}。"
            rsn = f"{flavor} EPS；ticker={ticker or 'NA'}；form={form or 'NA'}。"
            return ans, rsn, cits, disp

        if op == "eps_yoy":
            diluted, fy, fq = bool(args["diluted"]), int(args["fy"]), args.get("fq")
            v, rows, err = _compute_eps(df, ticker, form, diluted=diluted, fy=fy, fq=fq, yoy=True)
            if v is None: 
                 # 走自救链：先拿到“当前期”的行，再自己拼上一年前的行去算同比
                rows_now, note = _relax_numeric_attempts(df, ticker, form, concept, fy, args.get("fq"), hits)
                if rows_now:
                    fy_now = int(rows_now[-1].get("fy_norm") or rows_now[-1].get("fy"))
                    cur = rows_now[-1]
                    prv = _get_fact(df, ticker, form, concept, fy_now-1, _norm_fq_str(cur.get("fq_norm") or cur.get("fq")))
                    if prv and cur and float(prv.get("value_std") or 0) != 0:
                        v = (float(cur["value_std"]) - float(prv["value_std"])) / float(prv["value_std"])
                        rows = [prv, cur]; err = None
                        # 继续正常返回
                        cits, disp = _rows_to_citations(rows, hits)
                        unit = cur.get("unit_std") or "money"
                        ans = f"YoY：{_format_pct(v)}（{fy_now-1}→{fy_now}：{_format_money(prv['value_std']) if unit=='money' else f'{prv['value_std']:,}'} → {_format_money(cur['value_std']) if unit=='money' else f'{cur['value_std']:,}'}）。"
                        rsn = f"自救链成功：{note}"
                        return ans, rsn, cits, disp
                # 自救失败
                return None, f"信息不足：{err}" + (f" | {'; '.join(notes)}" if notes else ""), [], []
            cits, disp = _rows_to_citations(rows, hits)
            flavor = "Diluted" if diluted else "Basic"
            ans = f"EPS YoY ({flavor})：{_format_pct(v)}。"
            rsn = f"{flavor} EPS，{fy-1}→{fy}；ticker={ticker or 'NA'}；form={form or 'NA'}。"
            return ans, rsn, cits, disp

    except Exception as ex:
        if notes:
            logger.warning("numeric pipeline 异常：%s | notes: %s", ex, "; ".join(notes))
        else:
            logger.warning("numeric pipeline 异常：%s", ex)

    # 全部失败
    return None, "数据不足" + (f" | {'; '.join(notes)}" if notes else ""), [], []


# ======================================================================
# SECTION 9) 对外主入口（统一问答）
# ======================================================================
def answer_question(query: str, filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    统一问答入口：
      1) 调检索（hybrid_search）
      2) 判定问题类型：numeric / textual / mixed
      3) 数字题优先 facts_numeric 直算；文本/混合题走 hybrid_qa
      4) 引用兜底、计时、结构化返回
    """
    t0 = time.perf_counter()
    used_method = "hybrid_search"
    q = (query or "").strip()
    if not q:
        return _error_result("问题为空", used_method, t0)

    DEBUG = os.getenv("RAG_DEBUG", "0") in {"1","true","yes"}

    # 1) 检索
    try:
        hits = hybrid_search(query=q, filters=filters or {}, topk=8)
        hits = _dedupe_and_cap_hits(hits, topk=8)

        if DEBUG:
            print("\n[DEBUG] top hits (trimmed):")
            for h in hits[:5]:
                m = (h.get("meta") or {})
                snippet = (h.get("snippet") or h.get("text") or "")
                snippet = re.sub(r"\s+", " ", snippet).strip()[:120]
                print({
                    "chunk_id": h.get("chunk_id"),
                    "file_type": (m.get("file_type") or "text"),
                    "fy": (m.get("fy") or m.get("fy_norm")),
                    "fq": (m.get("fq") or m.get("fq_norm")),
                    "period_end": m.get("period_end"),
                    "concept": m.get("concept"),
                    "snippet": snippet
                })

    except Exception as e:
        logger.exception("检索阶段异常")
        return _error_result(f"检索失败：{e}", used_method, t0)

    # 2) 判定类型
    try:
        qtype = detect_query_type(q, hits)
        if force_numeric(q):
            qtype = "numeric"
    except Exception as e:
        logger.warning(f"detect_query_type 异常：{e}，将按 textual 处理")
        qtype = "textual"

    # 3) 分发作答
    try:
        if qtype == "numeric":
            fa, rsn, cits, cits_disp = _numeric_answer_pipeline(q, hits, filters)
            if fa is not None:
                citations = ensure_citations(cits, hits) or cits or []
                citations_display = cits_disp or [citation_to_display(c) for c in citations]
                latency_ms = int((time.perf_counter() - t0) * 1000)
                return {
                    "final_answer": fa,
                    "reasoning": rsn,
                    "citations": citations,
                    "citations_display": citations_display,
                    "used_method": used_method + "::numeric",
                    "latency_ms": latency_ms,
                    "prompt_version": PROMPT_VERSION,
                    "llm_model": None,
                    "ctx_tokens": None,
                }
            # numeric 失败：根据开关决定是否允许文本兜底
            if DISABLE_TEXT_FALLBACK:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                return {
                    "final_answer": "信息不足",
                    "reasoning": f"数值意图识别但无法安全计算：{rsn}",
                    "citations": [],
                    "citations_display": [],
                    "used_method": used_method + "::numeric_failed_no_text_fallback",
                    "latency_ms": latency_ms,
                    "prompt_version": PROMPT_VERSION,
                    "llm_model": None,
                    "ctx_tokens": None,
                }
            # 允许文本兜底（仅当你手动打开环境变量）
            final_answer, reasoning, citations = answer_textual_or_mixed(q, hits, filters, use_llm=USE_LLM)
            used = used_method + "::numeric_fallback_text"

        else:
            final_answer, reasoning, citations = answer_textual_or_mixed(q, hits, filters, use_llm=USE_LLM)
            used = used_method + f"::{qtype}"
    except Exception as e:
        logger.exception("作答阶段异常")
        return _error_result(f"作答失败：{e}", used_method + f"::{qtype}", t0)

    # 4) 引用兜底
    try:
        citations = ensure_citations(citations, hits)
        citations_display = [citation_to_display(c) for c in citations]
    except Exception:
        citations = citations or []
        citations_display = []

    latency_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "final_answer": final_answer,
        "reasoning": reasoning,
        "citations": citations,
        "citations_display": citations_display,
        "used_method": used,
        "latency_ms": latency_ms,
        "prompt_version": PROMPT_VERSION,
        "llm_model": (LLM_MODEL_NAME if USE_LLM and "text" in used else None),
        "ctx_tokens": None,
    }


# ======================================================================
# SECTION 10) 错误兜底
# ======================================================================
def _error_result(msg: str, used_method: str, t0: float) -> Dict[str, Any]:
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "final_answer": f"信息不足：{msg}",
        "reasoning": "在检索/作答阶段发生错误或输入不完整。",
        "citations": [],
        "citations_display": [],
        "used_method": used_method + "::error",
        "latency_ms": latency_ms,
        "prompt_version": PROMPT_VERSION,
        "llm_model": None,
        "ctx_tokens": None,
    }


# ======================================================================
# SECTION 11) CLI
# ======================================================================
def _build_filters(args) -> Dict[str, Any]:
    f: Dict[str, Any] = {}
    if args.ticker:
        f["ticker"] = args.ticker
    if args.form:
        f["form"] = args.form
    if args.year is not None:
        f["year"] = args.year
    if getattr(args, "fq", None):
        f["fq"] = args.fq
    return f

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Answer API - 财报 RAG 问答入口（含数值直算）")
    ap.add_argument("--q", required=True, help="问题（中文或英文）")
    ap.add_argument("--ticker", help="股票代码，如 AAPL")
    ap.add_argument("--form", help="报表类型，如 10-K/10-Q")
    ap.add_argument("--year", type=int, help="财年，如 2023")
    ap.add_argument("--fq", help="季度，如 Q4（可选）")
    ap.add_argument("--no-llm", action="store_true", help="禁用 LLM（仅文本题有效）")
    args = ap.parse_args()

    if args.no_llm:
        USE_LLM = False

    filters = _build_filters(args)
    res = answer_question(args.q, filters)
    print(json.dumps(res, ensure_ascii=False, indent=2))


'''
python -m src.qa.answer_api --q "What is the year-over-year revenue in 2023?" --ticker AAPL --form 10-K --year 2023

'''