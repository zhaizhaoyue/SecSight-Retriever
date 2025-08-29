#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Iterable, Dict, Any, List
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
def _to_int_or_none(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    m = re.search(r"\d+", s)
    return int(m.group()) if m else None

def _to_float(x: Any) -> float:
    try:
        if isinstance(x, str) and x.strip().lower() in ("true", "false"):
            return np.nan
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

    if pd.notna(inst):
        return f"FY{fy_i} instant {inst}" if fy_i is not None else f"instant {inst}"

    a = row.get("period_start")
    b = row.get("period_end")
    if pd.notna(a) or pd.notna(b):
        a = str(a) if pd.notna(a) else "?"
        b = str(b) if pd.notna(b) else "?"
        if (fy_i is not None) and (fq_i is not None):
            return f"FY{fy_i} Q{fq_i} {a}→{b}"
        if fy_i is not None:
            return f"FY{fy_i} {a}→{b}"
        return f"{a}→{b}"

    return "period:unknown"

def _fmt_value(r: pd.Series) -> str:
    v = r.get("value_num_clean")
    if pd.notna(v):
        if abs(v) >= 1e9: return f"{v/1e9:.3f} B"
        if abs(v) >= 1e6: return f"{v/1e6:.3f} M"
        if abs(v) >= 1e3: return f"{v/1e3:.3f} K"
        return f"{v:.6g}"
    raw = r.get("value_raw_clean")
    return str(raw) if pd.notna(raw) and str(raw).strip() else ""

def _mk_rag_text(r: pd.Series) -> str:
    label = r.get("label_text")
    if pd.isna(label) or not str(label).strip():
        label = r.get("concept") or r.get("qname") or "(no label)"
    val   = r.get("value_display")
    per   = r.get("period_label")
    tick  = r.get("ticker")
    form  = r.get("form")
    accno = r.get("accno")
    meta  = f"{tick} {form} accno={accno}" if pd.notna(accno) else f"{tick} {form}"
    return f"{label}: {val} ({per}; {meta})"

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

    facts_df = facts_df.copy()

    # 字段对齐
    if "concept" not in facts_df.columns and "qname" in facts_df.columns:
        facts_df["concept"] = facts_df["qname"]
    if "period_start" not in facts_df.columns and "start_date" in facts_df.columns:
        facts_df["period_start"] = facts_df["start_date"]
    if "period_end" not in facts_df.columns and "end_date" in facts_df.columns:
        facts_df["period_end"] = facts_df["end_date"]
    if "fy" not in facts_df.columns and "period_fy" in facts_df.columns:
        facts_df["fy"] = facts_df["period_fy"]
    if "fq" not in facts_df.columns and "period_fq" in facts_df.columns:
        facts_df["fq"] = facts_df["period_fq"]

    if "value_num" not in facts_df.columns:
        facts_df["value_num"] = np.nan
    if "value" in facts_df.columns:
        mask_nan = facts_df["value_num"].isna()
        facts_df.loc[mask_nan, "value_num"] = facts_df.loc[mask_nan, "value"]

    # 清洗
    facts_df["value_num_clean"] = facts_df.apply(
        lambda r: _to_float(r.get("value_num")) if pd.notna(r.get("value_num")) else _to_float(r.get("value")), axis=1
    )
    facts_df["value_raw_clean"] = (
        facts_df["value_raw"].map(_norm_text)
        if "value_raw" in facts_df.columns
        else pd.Series([None] * len(facts_df), index=facts_df.index)
    )

    if "period_label" not in facts_df.columns:
        facts_df["period_label"] = facts_df.apply(_mk_period_label, axis=1)

    facts_df["fy_norm"] = (
        facts_df["fy"].map(_to_int_or_none)
        if "fy" in facts_df.columns
        else pd.Series([None] * len(facts_df), index=facts_df.index)
    )

    facts_df["fq_norm"] = (
        facts_df["fq"].map(_to_int_or_none)
        if "fq" in facts_df.columns
        else pd.Series([None] * len(facts_df), index=facts_df.index)
    )


    # 输出列
    facts_df["value_display"] = facts_df.apply(_fmt_value, axis=1)
    facts_df["rag_text"]      = facts_df.apply(_mk_rag_text, axis=1)
    for c in ("period_start", "period_end", "instant", "fy", "fq"):
        if c not in facts_df.columns:
            facts_df[c] = pd.NA

    keep = [
        "concept","label_text",
        "value_raw_clean","value_num_clean","value_display","rag_text",
        "period_start","period_end","instant","period_label",
        "ticker","year","fy","fq","form","accno","doc_date","source_path",
    ]
    extra_cols = [c for c in ["dimensions_json","unit_normalized","unit_family","statement_hint","decimals"]
                  if c in facts_df.columns]
    cols = keep + extra_cols
    cols = [c for c in cols if c in facts_df.columns]
    return facts_df.loc[:, cols].drop_duplicates()

# -------------------------
# Scan inputs
# -------------------------
def iter_inputs(root: Path) -> Iterable[Path]:
    yield from root.rglob("facts.jsonl")
    yield from root.rglob("facts.parquet")

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Batch clean facts (processed → clean, same structure)")
    ap.add_argument("--root", type=str, default=None,
                    help="输入根目录（默认 data/processed）")
    ap.add_argument("--dry-run", action="store_true", help="只扫描不写文件")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else DEFAULT_INPUT_DIR.resolve()
    out_root = DEFAULT_OUTPUT_DIR.resolve()

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
