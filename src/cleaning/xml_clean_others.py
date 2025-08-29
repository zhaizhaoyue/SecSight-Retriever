#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

# -----------------------
# 项目路径
# -----------------------
PROJECT_ROOT        = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR   = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR  = PROJECT_ROOT / "data" / "clean"

# -----------------------
# 基础 IO
# -----------------------
def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                s = line.strip()
                if not s:
                    continue
                try:
                    rows.append(json.loads(s))
                except Exception as e:
                    print(f"    [WARN] {path.name} 第{i}行解析失败：{e}")
        return pd.DataFrame(rows)
    elif suf == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported input type: {path}")

def try_write_parquet(df: pd.DataFrame, out_path: Path) -> bool:
    for engine in ("pyarrow", "fastparquet"):
        try:
            df.to_parquet(out_path, index=False, engine=engine)
            return True
        except Exception:
            continue
    print(f"    [WARN] 未安装 pyarrow/fastparquet，跳过写入 {out_path.name}。")
    return False

def save_jsonl(df: pd.DataFrame, out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def mirror_out_dir(in_file: Path, in_root: Path, out_root: Path) -> Path:
    """把 processed 的相对路径镜像到 clean 下（去掉文件名，保留其父目录层级）"""
    rel = in_file.parent.relative_to(in_root)  # e.g. AAPL/2023/10-Q_xxx
    out_dir = (out_root / rel).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

# -----------------------
# labels 优选（英文优先；standard > terse > verbose）
# -----------------------
ROLE_PRI = [
    "http://www.xbrl.org/2003/role/label",        # standard
    "http://www.xbrl.org/2003/role/terseLabel",   # terse
    "http://www.xbrl.org/2003/role/verboseLabel", # verbose
]

def build_preferred_labels(labels_df: pd.DataFrame) -> pd.DataFrame:
    if labels_df.empty or "concept" not in labels_df.columns:
        return pd.DataFrame(columns=["concept","label_text"])
    def _role_rank(x):
        try: return ROLE_PRI.index(x)
        except Exception: return len(ROLE_PRI)
    labels_df = labels_df.copy()
    labels_df["role_rank"] = labels_df.get("label_role", "").apply(_role_rank)
    labels_df["lang_rank"] = labels_df.get("lang", "").apply(
        lambda x: 0 if str(x).lower().startswith("en") else 1
    )
    pref = (labels_df
            .sort_values(["concept","role_rank","lang_rank"])
            .groupby("concept", as_index=False)
            .first()[["concept","label_text"]])
    return pref

# -----------------------
# 清洗：calculation_edges
# -----------------------
def clean_calculation_edges(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # 统一列存在
    for c in ["parent_concept","child_concept","weight","order","linkrole",
              "ticker","year","fy","fq","form","accno","doc_date","file_type","source_path"]:
        if c not in df.columns:
            df[c] = pd.NA
    # 类型规整
    def _to_float(x):
        try: return float(x)
        except Exception: return np.nan
    df["weight"] = df["weight"].apply(_to_float)
    # 去重
    keep = ["parent_concept","child_concept","weight","order","linkrole",
            "ticker","year","fy","fq","form","accno","doc_date","file_type","source_path"]
    df = df[keep].drop_duplicates().reset_index(drop=True)
    return df

# -----------------------
# 清洗：definition_arcs
# -----------------------
def clean_definition_arcs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for c in ["from_concept","to_concept","arcrole","order","linkrole",
              "ticker","year","fy","fq","form","accno","doc_date","file_type","source_path"]:
        if c not in df.columns:
            df[c] = pd.NA
    keep = ["from_concept","to_concept","arcrole","order","linkrole",
            "ticker","year","fy","fq","form","accno","doc_date","file_type","source_path"]
    df = df[keep].drop_duplicates().reset_index(drop=True)
    return df

# -----------------------
# 清洗：labels（输入 labels.parquet / labels.jsonl → 输出优选映射）
# -----------------------
def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["concept","label_text"])
    pref = build_preferred_labels(df)
    return pref

# -----------------------
# 主逻辑
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Clean calculation_edges / definition_arcs / labels with processed→clean mirroring")
    ap.add_argument("--root", type=str, default=None, help="输入根目录（默认 PROJECT_ROOT/data/processed）")
    ap.add_argument("--dry-run", action="store_true", help="只扫描不写文件")
    args = ap.parse_args()

    in_root  = Path(args.root).expanduser().resolve() if args.root else DEFAULT_INPUT_DIR.resolve()
    out_root = DEFAULT_OUTPUT_DIR.resolve()

    if not in_root.exists():
        print(f"[INFO] 输入根目录不存在：{in_root}")
        return

    # 搜索三类文件
    files_calc = list(in_root.rglob("calculation_edges.jsonl")) + list(in_root.rglob("calculation_edges.parquet"))
    files_def  = list(in_root.rglob("definition_arcs.jsonl"))  + list(in_root.rglob("definition_arcs.parquet"))
    files_lab  = list(in_root.rglob("labels.jsonl"))           + list(in_root.rglob("labels.parquet"))

    total = len(files_calc) + len(files_def) + len(files_lab)
    if total == 0:
        print(f"[INFO] 在 {in_root} 未找到 calculation_edges/definition_arcs/labels")
        return

    print(f"[INFO] 输入根：{in_root}")
    print(f"[INFO] 输出根：{out_root}（镜像 processed 结构）")
    print(f"[INFO] 发现：calc={len(files_calc)}, def={len(files_def)}, labels={len(files_lab)}")

    # 1) calculation_edges
    for i, p in enumerate(files_calc, 1):
        try:
            print(f"[calc {i}/{len(files_calc)}] {p}")
            df = read_table(p)
            cleaned = clean_calculation_edges(df)
            out_dir = mirror_out_dir(p, in_root, out_root)
            if args.dry_run:
                print(f"    -> dry-run rows={len(cleaned)} dir={out_dir}")
                continue
            # 输出文件名固定
            ok = try_write_parquet(cleaned, out_dir / "calculation_edges.parquet")
            save_jsonl(cleaned, out_dir / "calculation_edges.jsonl")
            print(f"    -> 输出：{'calculation_edges.parquet, ' if ok else ''}calculation_edges.jsonl  rows={len(cleaned)}  dir={out_dir}")
        except Exception as e:
            print(f"    [WARN] 失败：{p}\n{e}")

    # 2) definition_arcs
    for i, p in enumerate(files_def, 1):
        try:
            print(f"[def  {i}/{len(files_def)}] {p}")
            df = read_table(p)
            cleaned = clean_definition_arcs(df)
            out_dir = mirror_out_dir(p, in_root, out_root)
            if args.dry_run:
                print(f"    -> dry-run rows={len(cleaned)} dir={out_dir}")
                continue
            ok = try_write_parquet(cleaned, out_dir / "definition_arcs.parquet")
            save_jsonl(cleaned, out_dir / "definition_arcs.jsonl")
            print(f"    -> 输出：{'definition_arcs.parquet, ' if ok else ''}definition_arcs.jsonl  rows={len(cleaned)}  dir={out_dir}")
        except Exception as e:
            print(f"    [WARN] 失败：{p}\n{e}")

    # 3) labels → 优选映射（概念→标签）
    for i, p in enumerate(files_lab, 1):
        try:
            print(f"[labs {i}/{len(files_lab)}] {p}")
            raw = read_table(p)
            cleaned = clean_labels(raw)  # 只保留 concept,label_text
            out_dir = mirror_out_dir(p, in_root, out_root)
            if args.dry_run:
                print(f"    -> dry-run rows={len(cleaned)} dir={out_dir}")
                continue
            ok = try_write_parquet(cleaned, out_dir / "labels.parquet")
            save_jsonl(cleaned, out_dir / "labels.jsonl")
            print(f"    -> 输出：{'labels.parquet, ' if ok else ''}labels.jsonl  rows={len(cleaned)}  dir={out_dir}")
        except Exception as e:
            print(f"    [WARN] 失败：{p}\n{e}")

    print("[DONE] 全部完成。")

if __name__ == "__main__":
    main()
