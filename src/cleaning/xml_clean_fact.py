#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional
import re

import numpy as np
import pandas as pd

# -------------------------
# Project paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR   = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR  = PROJECT_ROOT / "data" / "clean"

# -------------------------
# Helpers
# -------------------------
_QNUM = re.compile(r"\d+")

def _parse_human_number(s: Any) -> float:
    if s is None:
        return np.nan
    txt = str(s).strip()
    if not txt:
        return np.nan
    # 负号：括号 or 前缀 -
    neg = False
    if txt.startswith("(") and txt.endswith(")"):
        neg = True
        txt = txt[1:-1]
    txt = txt.replace("\u00A0", " ").replace(",", "").replace("$", "").strip()  # nbsp/逗号/美元符
    m = re.match(r"^([+-]?\d+(?:\.\d+)?)(?:\s*([KkMmBbTt]))?$", txt)
    if not m:
        return np.nan
    num = float(m.group(1))
    suf = (m.group(2) or "").upper()
    mul = {"K":1e3,"M":1e6,"B":1e9,"T":1e12}.get(suf, 1.0)
    val = num * mul
    return -val if neg else val


def _to_int_or_none(x) -> Optional[int]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    m = _QNUM.search(s)
    return int(m.group()) if m else None

def _to_float(x: Any) -> float:
    try:
        if x is None:
            return np.nan
        if isinstance(x, str):
            xs = x.strip()
            if xs.lower() in ("true", "false", ""):
                return np.nan
            # 去掉千分位逗号与末尾 %
            xs2 = xs.replace(",", "")
            if xs2.endswith("%"):
                xs2 = xs2[:-1]
            return float(xs2)
        return float(x)
    except Exception:
        return np.nan

def _norm_text(x: Any) -> Any:
    if isinstance(x, str):
        xl = x.strip()
        if xl.lower() in ("true", "false"):
            return xl.lower()
        return xl
    return x

def _mk_period_label(row: pd.Series) -> str:
    fy_i = _to_int_or_none(row.get("fy", row.get("period_fy")))
    fq_i = _to_int_or_none(row.get("fq", row.get("period_fq")))
    inst = row.get("instant")

    if pd.notna(inst) and str(inst).strip():
        return f"FY{fy_i} instant {inst}" if fy_i is not None else f"instant {inst}"

    a = row.get("period_start")
    b = row.get("period_end")
    a = str(a) if pd.notna(a) and str(a).strip() else None
    b = str(b) if pd.notna(b) and str(b).strip() else None

    if a or b:
        a_ = a or "?"
        b_ = b or "?"
        if (fy_i is not None) and (fq_i is not None):
            return f"FY{fy_i} Q{fq_i} {a_}→{b_}"
        if fy_i is not None:
            return f"FY{fy_i} {a_}→{b_}"
        return f"{a_}→{b_}"

    return "period:unknown"

def _fmt_value(r: pd.Series) -> str:
    # 优先数值渲染
    v = r.get("value_num")
    unit_family = (r.get("unit_family") or "").strip().lower()
    if pd.notna(v):
        # 如果是百分比（数值是小数），显示成人读格式
        if unit_family == "percent":
            return f"{v*100:.6g}%"
        # 货币或其他
        av = abs(v)
        if av >= 1e12: return f"{v/1e12:.3f} T"
        if av >= 1e9:  return f"{v/1e9:.3f} B"
        if av >= 1e6:  return f"{v/1e6:.3f} M"
        if av >= 1e3:  return f"{v/1e3:.3f} K"
        return f"{v:.6g}"

    # 否则使用原始文本
    raw = r.get("value_raw")
    if pd.notna(raw) and str(raw).strip():
        return str(raw).strip()
    raw2 = r.get("value_raw_clean")
    if pd.notna(raw2) and str(raw2).strip():
        return str(raw2).strip()
    return ""

def _mk_rag_text(r: pd.Series) -> str:
    label = r.get("label_text")
    if pd.isna(label) or not str(label).strip():
        label = r.get("concept") or r.get("qname") or "(no label)"
    val   = r.get("value_display")
    per   = r.get("period_label")
    tick  = r.get("ticker") or ""
    form  = r.get("form") or ""
    accno = r.get("accno")
    meta  = f"{tick} {form}".strip()
    if pd.notna(accno) and str(accno).strip():
        meta = f"{meta} accno={accno}".strip()
    return f"{label}: {val} ({per}; {meta})"

def _parse_dimensions(df: pd.DataFrame) -> pd.Series:
    """
    从 dimensions_json / dimensions 列生成 dims_signature。
    """
    if "dimensions_json" in df.columns:
        def _from_json(x):
            try:
                if isinstance(x, str) and x.strip():
                    obj = json.loads(x)
                    if isinstance(obj, dict) and obj:
                        return "|".join(f"{k}={v}" for k, v in sorted(obj.items()))
            except Exception:
                pass
            return ""
        return df["dimensions_json"].map(_from_json)

    if "dimensions" in df.columns:
        def _from_dict(x):
            if isinstance(x, dict) and x:
                return "|".join(f"{k}={v}" for k, v in sorted(x.items()))
            # 有些解析器落了字符串
            try:
                if isinstance(x, str) and x.strip().startswith("{"):
                    obj = json.loads(x)
                    if isinstance(obj, dict) and obj:
                        return "|".join(f"{k}={v}" for k, v in sorted(obj.items()))
            except Exception:
                pass
            return ""
        return df["dimensions"].map(_from_dict)

    return pd.Series([""] * len(df), index=df.index)

# -------------------------
# IO helpers
# -------------------------
def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".jsonl":
        recs: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except Exception as e:
                    print(f"    [WARN] {path.name} 第{i}行 JSON 解析失败：{e}")
        return pd.DataFrame(recs)
    if suf == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input type: {path.name}")

def try_write_parquet(df: pd.DataFrame, out_path: Path) -> bool:
    for engine in ("pyarrow", "fastparquet"):
        try:
            df.to_parquet(out_path, index=False, engine=engine)
            return True
        except Exception:
            continue
    print(f"    [WARN] 未安装 pyarrow/fastparquet，跳过写入 {out_path.name}")
    return False

def save_jsonl(df: pd.DataFrame, out_path: Path) -> None:
    df_safe = df.where(pd.notna(df), None)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in df_safe.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# -------------------------
# Cleaning core
# -------------------------
def clean_one_facts_table(facts_df: pd.DataFrame) -> pd.DataFrame:
    if facts_df is None or facts_df.empty:
        print("    [SKIP] 空的 facts，跳过")
        return pd.DataFrame()

    df = facts_df.copy()

    # ---------- 小工具 ----------
    def _parse_human_number(s: Any) -> float:
        # 解析 "442.000 M"、"(565)"、"$2,825"、"12 K" 等
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return np.nan
        txt = str(s).strip()
        if not txt:
            return np.nan
        neg = False
        if txt.startswith("(") and txt.endswith(")"):
            neg = True
            txt = txt[1:-1]
        txt = (txt.replace("\u00A0", " ")  # nbsp
                   .replace(",", "")
                   .replace("$", "")
                   .strip())
        m = re.match(r"^([+-]?\d+(?:\.\d+)?)(?:\s*([KkMmBbTt]))?$", txt)
        if not m:
            return np.nan
        num = float(m.group(1))
        mul = {"K":1e3, "M":1e6, "B":1e9, "T":1e12}.get((m.group(2) or "").upper(), 1.0)
        val = num * mul
        return -val if neg else val

    def _dims_signature_from(row: pd.Series) -> str:
        # 维度字典按 key 排序，拼 "key=value|..."；无维度返回空串
        dims = None
        if "dimensions_json" in row and pd.notna(row["dimensions_json"]):
            try:
                dims = json.loads(row["dimensions_json"])
            except Exception:
                dims = None
        if dims is None and "dimensions" in row:
            d = row["dimensions"]
            if isinstance(d, dict):
                dims = d
        if not isinstance(dims, dict) or not dims:
            return ""
        items = [f"{k}={dims[k]}" for k in sorted(dims.keys())]
        return "|".join(items)

    def _fmt_value_from_num(v: Any) -> str:
        if pd.isna(v):
            return ""
        v = float(v)
        if abs(v) >= 1e12: return f"{v/1e12:.3f} T"
        if abs(v) >= 1e9:  return f"{v/1e9:.3f} B"
        if abs(v) >= 1e6:  return f"{v/1e6:.3f} M"
        if abs(v) >= 1e3:  return f"{v/1e3:.3f} K"
        return f"{v:.6g}"

    def _mk_rag_text(r: pd.Series) -> str:
        label = r.get("label_text")
        if pd.isna(label) or not str(label).strip():
            label = r.get("concept") or r.get("qname") or "(no label)"
        val   = r.get("value_display", "")
        per   = r.get("period_label")
        tick  = r.get("ticker")
        form  = r.get("form")
        accno = r.get("accno")
        meta  = f"{tick} {form} accno={accno}" if pd.notna(accno) else f"{tick} {form}"
        return f"{label}: {val} ({per}; {meta})"

    def _to_int_or_none(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        s = str(x).strip()
        m = re.search(r"\d+", s)
        return int(m.group()) if m else None

    # ---------- 字段对齐 / 兜底 ----------
    if "concept" not in df.columns and "qname" in df.columns:
        df["concept"] = df["qname"]

    if "period_start" not in df.columns and "start_date" in df.columns:
        df["period_start"] = df["start_date"]
    if "period_end" not in df.columns and "end_date" in df.columns:
        df["period_end"] = df["end_date"]

    if "fy" not in df.columns and "period_fy" in df.columns:
        df["fy"] = df["period_fy"]
    if "fq" not in df.columns and "period_fq" in df.columns:
        df["fq"] = df["period_fq"]

    # value_raw / value_num 保持统一存在
    if "value_raw" not in df.columns and "value_raw_clean" in df.columns:
        df["value_raw"] = df["value_raw_clean"]
    if "value_num" not in df.columns:
        df["value_num"] = np.nan
    # 若 value 存在，先补到 value_num（有些管道把数值放在 value）
    if "value" in df.columns:
        mask_nan = df["value_num"].isna()
        df.loc[mask_nan, "value_num"] = df.loc[mask_nan, "value"]

    # ---------- 数值归一（多重来源） ----------
    # 1) value_num 自身 → 浮点
    df["value_num"] = df["value_num"].map(lambda x: float(x) if pd.notna(x) and str(x).strip().lower() not in ("true","false") else np.nan)

    # 2) value 兜底（若存在）
    if "value" in df.columns:
        mask_nan = df["value_num"].isna()
        df.loc[mask_nan, "value_num"] = df.loc[mask_nan, "value"].map(lambda x: float(x) if pd.notna(x) and str(x).strip().lower() not in ("true","false") else np.nan)

    # 3) value_raw 兜底（字符串数字）
    if "value_raw" in df.columns:
        mask_nan = df["value_num"].isna()
        df.loc[mask_nan, "value_num"] = df.loc[mask_nan, "value_raw"].map(lambda x: _parse_human_number(x) if x is not None else np.nan)

    # 4) value_display 最后兜底（"442.000 M" 等）
    if "value_display" in df.columns:
        mask_nan = df["value_num"].isna()
        df.loc[mask_nan, "value_num"] = df.loc[mask_nan, "value_display"].map(_parse_human_number)

    # 百分比归一（/100）
    unit_family = (df["unit_family"].astype(str).str.lower() if "unit_family" in df.columns
                   else pd.Series([""] * len(df), index=df.index))
    concept_series = df["concept"].astype(str).str.lower() if "concept" in df.columns else pd.Series([""] * len(df), index=df.index)
    value_text_src = None
    if "value_display" in df.columns:
        value_text_src = df["value_display"]
    elif "value_raw" in df.columns:
        value_text_src = df["value_raw"].astype(str)
    else:
        value_text_src = pd.Series([""] * len(df), index=df.index)

    percent_mask = (
        unit_family.eq("percent")
        | value_text_src.fillna("").str.contains("%", regex=False)
        | concept_series.str.contains("percent|percentage")
    )
    df.loc[percent_mask & df["value_num"].notna(), "value_num"] = df.loc[percent_mask & df["value_num"].notna(), "value_num"] / 100.0

    # ---------- period / label / rag ----------
    if "period_label" not in df.columns:
        # 用 _mk_period_label（若你有全局版本，也可以直接调用；这里就地实现）
        def _mk_period_label_local(r: pd.Series) -> str:
            fy_i = _to_int_or_none(r.get("fy", r.get("period_fy")))
            inst = r.get("instant")
            if pd.notna(inst):
                return f"FY{fy_i} instant {inst}" if fy_i is not None else f"instant {inst}"
            a = r.get("period_start"); b = r.get("period_end")
            if pd.notna(a) or pd.notna(b):
                a = str(a) if pd.notna(a) else "?"
                b = str(b) if pd.notna(b) else "?"
                return f"FY{fy_i} {a} — {b}" if fy_i is not None else f"{a} — {b}"
            return "period:unknown"
        df["period_label"] = df.apply(_mk_period_label_local, axis=1)

    # 归一化 FY/FQ（数字）
    df["fy_norm"] = df["fy"].map(_to_int_or_none) if "fy" in df.columns else None
    df["fq_norm"] = df["fq"].map(_to_int_or_none) if "fq" in df.columns else None

    # dims_signature
    if "dims_signature" not in df.columns:
        df["dims_signature"] = df.apply(_dims_signature_from, axis=1)

    # value_display（用归一后的 value_num 重新格式化）
    df["value_display"] = df["value_num"].map(_fmt_value_from_num)

    # rag_text
    df["rag_text"] = df.apply(_mk_rag_text, axis=1)

    # ---------- decimals 归一 ----------
    if "decimals" in df.columns:
        dec = pd.to_numeric(df["decimals"], errors="coerce")
        dec = dec.replace([np.inf, -np.inf], np.nan)
        try:
            df["decimals"] = pd.array(dec, dtype="Int64")
        except Exception:
            df["decimals"] = dec  # 兜底

    


    # ---------- 输出列 ----------
    base_keep = [
        "concept", "label_text",
        "value_raw", "value_num", "value_display", "rag_text",
        "period_start", "period_end", "instant", "period_label",
        "ticker", "year", "fy", "fq", "fy_norm", "fq_norm",
        "form", "accno", "doc_date", "source_path",
        "unit_normalized", "unit_family", "statement_hint", "decimals",
        "context_id", "context", "dimensions", "dimensions_json", "dims_signature",
    ]
    # 如果只有 unit 没有 unit_normalized，也保留 unit 方便追溯
    if "unit_normalized" not in df.columns and "unit" in df.columns:
        base_keep.append("unit")

    is_textblock = df["concept"].astype(str).str.endswith("TextBlock")
    for col in ("unit_normalized","unit_family","decimals"):
        if col in df.columns:
            df.loc[is_textblock, col] = None
    if "unit" in df.columns:
        df.loc[is_textblock, "unit"] = None
    # 占位的 value_display 已有就保留；若没有可以补：
    if "value_display" in df.columns:
        missing_disp = is_textblock & df["value_display"].isna()
        df.loc[missing_disp, "value_display"] = "[HTML TextBlock]"

    # —— EPS/股本相关概念：统一为 income —— 
    income_eps_mask = df["concept"].astype(str).str.contains(
        r"(?:EarningsPerShare|DilutedShares|BasicShares|AntidilutiveSecurities|WeightedAverageNumber)",
        case=False, regex=True, na=False
    )

    if "statement_hint" in df.columns:
        df.loc[income_eps_mask, "statement_hint"] = "income"
    
    concept_s = df["concept"].astype(str)

    # instant 行
    inst_mask = df["instant"].notna() if "instant" in df.columns else pd.Series(False, index=df.index)

    # 概念包含 Accumulated + Unrealized + (Gain|Loss)
    accum_unrealized_mask = concept_s.str.contains(
        r"Accumulated.*Unrealized.*(Gain|Loss)", case=False, regex=True, na=False
    )

    # 维度包含 FV 层级轴（Balance 常见切片）
    fv_hier_mask = (
        df["dims_signature"].astype(str).str.contains("FairValueByFairValueHierarchyLevelAxis", na=False)
        if "dims_signature" in df.columns else pd.Series(False, index=df.index)
    )

    # 覆盖为 balance（instant 且满足累计/层级条件任一）
    if "statement_hint" in df.columns:
        df.loc[inst_mask & (accum_unrealized_mask | fv_hier_mask), "statement_hint"] = "balance"


    # —— 空维度签名统一为 None —— 
    if "dims_signature" in df.columns:
        df["dims_signature"] = df["dims_signature"].replace({"": None})

    # —— period 字段以 context 为准（避免与 context_id 内嵌日期不一致）
    def _pick_period(row: pd.Series) -> pd.Series:
        ctx = row.get("context")
        per = {}

        if isinstance(ctx, dict):
            per = ctx.get("period") or {}
        elif isinstance(ctx, str):
            # 可能是从 parquet 里读出来的 JSON 字符串
            try:
                obj = json.loads(ctx)
                if isinstance(obj, dict):
                    per = obj.get("period") or {}
                    # 顺便把 context 规范回 dict，后续就安全了
                    row["context"] = obj
            except Exception:
                per = {}
        else:
            per = {}

        for k, dst in (("start_date", "period_start"),
                    ("end_date", "period_end"),
                    ("instant", "instant")):
            v = per.get(k)
            if v is not None:
                row[dst] = v
        return row
    df = df.apply(_pick_period, axis=1)

    # --- 清理“无维度”科目的维度信息（避免误挂 Axis 影响 RAG） ---
    if "concept" in df.columns:
        _DIMENSIONLESS = [
            "us-gaap:CashAndCashEquivalentsAtCarryingValue",
            "us-gaap:MarketableSecuritiesCurrent",
            "us-gaap:MarketableSecuritiesNoncurrent",
            # 如需可继续扩充：总资产/总负债/股东权益等汇总科目
            # "us-gaap:Assets", "us-gaap:Liabilities", "us-gaap:StockholdersEquity",
        ]
        m_dimless = df["concept"].isin(_DIMENSIONLESS)

        # 清空维度与签名
        if "dimensions" in df.columns:
            df.loc[m_dimless, "dimensions"] = None
        if "dimensions_json" in df.columns:
            df.loc[m_dimless, "dimensions_json"] = None
        if "dims_signature" in df.columns:
            df.loc[m_dimless, "dims_signature"] = None

        # 这些概念通常是资产负债表端（instant）；顺手统一 hint
        if "instant" in df.columns and "statement_hint" in df.columns:
            df.loc[m_dimless & df["instant"].notna(), "statement_hint"] = "balance"

    
    # —— 资产负债表优先级：对典型 Balance 概念做覆盖 —— 
    concept_s = df["concept"].astype(str)

    balance_like_mask = concept_s.str.contains(
        r"(?:CashAndCashEquivalents(?:AtCarryingValue)?"
        r"|MarketableSecurities(?:Current|Noncurrent)?"
        r"|Assets|Liabilities|StockholdersEquity|ShareholdersEquity"
        r"|RetainedEarnings|AccumulatedOtherComprehensiveIncome"
        r"|ShortTerm|LongTerm|Debt|Inventory|Receivables"
        r"|PropertyPlantAndEquipment|PPE)"
        , case=False, regex=True, na=False
    )

    # 安全拿到 instant 的非空掩码
    instant_mask = df["instant"].notna() if "instant" in df.columns else pd.Series([False]*len(df), index=df.index)
    if isinstance(instant_mask, pd.Series):
        df.loc[instant_mask & balance_like_mask, "statement_hint"] = "balance"


    # —— 明确覆盖已知科目（更保险） —— 
    known_balance = concept_s.isin([
        "us-gaap:CashAndCashEquivalentsAtCarryingValue",
        "us-gaap:MarketableSecuritiesCurrent",
        "us-gaap:MarketableSecuritiesNoncurrent",
    ])
    df.loc[known_balance, "statement_hint"] = "balance"

    cols = [c for c in base_keep if c in df.columns]
    out = df.loc[:, cols].copy()
    # 去重（概念+期间+维度+accno）
    dedup_keys = [k for k in ["concept", "period_start", "period_end", "instant", "dims_signature", "accno"] if k in out.columns]
    if dedup_keys:
        out = out.drop_duplicates(subset=dedup_keys, keep="first")

    return out


# -------------------------
# Scan inputs
# -------------------------
def iter_inputs(root: Path) -> Iterable[Path]:
    # 优先读取 parquet（更快），再读 jsonl，二者都保留
    yield from root.rglob("facts.parquet")
    yield from root.rglob("facts.jsonl")

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Batch clean facts (processed → clean, same structure)")
    ap.add_argument("--root", type=str, default=None, help="输入根目录（默认 data/processed）")
    ap.add_argument("--out", type=str, default=None, help="输出根目录（默认 data/clean）")
    ap.add_argument("--dry-run", action="store_true", help="只扫描不写文件")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else DEFAULT_INPUT_DIR.resolve()
    out_root = Path(args.out).expanduser().resolve() if args.out else DEFAULT_OUTPUT_DIR.resolve()

    if not root.exists():
        print(f"[INFO] 输入根目录不存在：{root}")
        return

    inputs = list(iter_inputs(root))
    if not inputs:
        print(f"[INFO] 在 {root} 下未找到 facts.jsonl / facts.parquet")
        return

    print(f"[INFO] 输入根目录：{root}")
    print(f"[INFO] 输出根目录：{out_root}（与 processed 结构一致）")
    print(f"[INFO] 待处理文件数：{len(inputs)}")

    for i, facts_path in enumerate(inputs, 1):
        try:
            print(f"[{i}/{len(inputs)}] 清洗：{facts_path}")
            facts_df  = read_table(facts_path)
            cleaned   = clean_one_facts_table(facts_df)
            if cleaned.empty:
                print("    [SKIP] 空结果，跳过")
                continue

            rel = facts_path.parent.relative_to(root)
            out_dir = out_root / rel

            if args.dry_run:
                print(f"    -> dry-run：rows={len(cleaned)}  out_dir={out_dir}")
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            pq = out_dir / "fact.parquet"
            jl = out_dir / "fact.jsonl"

            ok = try_write_parquet(cleaned, pq)
            save_jsonl(cleaned, jl)
            print(f"    -> 输出：{'fact.parquet, ' if ok else ''}fact.jsonl  rows={len(cleaned)}  dir={out_dir}")

        except Exception as e:
            print(f"[WARN] 处理失败：{facts_path}\n{e}")

    print("[DONE] 全部完成。")

if __name__ == "__main__":
    main()
