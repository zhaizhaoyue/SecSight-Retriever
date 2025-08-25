import argparse, re, json, sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

try:
    import yaml
except Exception:
    yaml = None

def _is_nan(x):
    return isinstance(x, float) and x != x

def _collapse_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def _strip_colon(s: str) -> str:
    return re.sub(r"[:：]\s*$", "", s.strip())

def _normalize_label(s) -> str:
    if s is None or _is_nan(s):
        s = ""
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = _collapse_spaces(s)
    s = _strip_colon(s)
    return s

def load_metric_map(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("pyyaml is required to read YAML. Please pip install pyyaml.")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    flat: Dict[str, Dict[str, Any]] = {}
    for section_key, section_val in data.items():
        if section_key.startswith("__"):
            continue
        if isinstance(section_val, dict):
            for canon, meta in section_val.items():
                meta = meta or {}
                flat_key = f"{section_key}.{canon}"
                flat[flat_key] = {
                    "section": section_key,
                    "canonical": canon,
                    "aliases": [a for a in meta.get("aliases", []) if a],
                    "regex": [r for r in meta.get("regex", []) if r],
                    "sign_convention": meta.get("sign_convention"),
                    "compute_from": meta.get("compute_from"),
                    "description": meta.get("description"),
                }
    return flat

def build_match_index(metric_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    alias_index = []  # (section, canonical, alias_norm)
    regex_index = []
    for key, meta in metric_map.items():
        section = meta["section"]
        canon = meta["canonical"]
        for alias in meta.get("aliases", []):
            alias_norm = _normalize_label(alias)
            alias_index.append((section, canon, alias_norm))
        for rx in meta.get("regex", []):
            try:
                regex_index.append((section, canon, re.compile(rx, re.IGNORECASE)))
            except re.error:
                continue
    return {"aliases": alias_index, "regex": regex_index}

def map_label(raw_label, stmt_hint: Optional[str], match_index: Dict[str, Any]):
    raw_norm = _normalize_label(raw_label)
    stmt_hint_norm = _normalize_label(stmt_hint or "")
    best = None

    for sec, canon, alias_norm in match_index["aliases"]:
        if raw_norm == alias_norm:
            if stmt_hint and sec in stmt_hint_norm:
                return sec, canon
            best = best or (sec, canon)
    if best:
        return best

    for sec, canon, alias_norm in match_index["aliases"]:
        if alias_norm and alias_norm in raw_norm:
            if stmt_hint and sec in stmt_hint_norm:
                return sec, canon
            best = best or (sec, canon)
    if best:
        return best

    for sec, canon, rx in match_index["regex"]:
        if rx.search(str(raw_label) if raw_label is not None else ""):
            if stmt_hint and sec in stmt_hint_norm:
                return sec, canon
            best = best or (sec, canon)
    if best:
        return best

    return None, None

def normalize_tables(df: pd.DataFrame, metric_map_path: Path, row_col: str = "row_label_raw",
                     stmt_col: Optional[str] = "statement_type", value_cols_hint: Optional[List[str]] = None):
    metric_map = load_metric_map(metric_map_path)
    match_index = build_match_index(metric_map)

    if row_col not in df.columns:
        raise ValueError(f"Row label column '{row_col}' not found in input columns: {list(df.columns)[:20]} ...")

    sections, canonicals = [], []
    raw_list = df[row_col].tolist()
    for i, raw in enumerate(raw_list):
        stmt_hint = df[stmt_col].iloc[i] if (stmt_col and stmt_col in df.columns) else None
        sec, can = map_label(raw, stmt_hint, match_index)
        sections.append(sec)
        canonicals.append(can)

    df_out = df.copy()
    df_out["section_mapped"] = sections
    df_out["canonical_metric"] = canonicals

    unmapped_mask = df_out["canonical_metric"].isna()
    cols = [row_col]
    if stmt_col and stmt_col in df_out.columns:
        cols.append(stmt_col)
    base = df_out.loc[unmapped_mask, cols].copy()
    base["row_norm"] = base[row_col].map(_normalize_label)
    unmapped_df = (base.value_counts(subset=["row_norm"])
                        .rename("count")
                        .reset_index()
                        .sort_values("count", ascending=False))

    if value_cols_hint:
        val_cols = [c for c in value_cols_hint if c in df_out.columns]
    else:
        likely_idx = {row_col, "row_label_norm", "table_id", "source_page", "company", "report", "statement_type"}
        val_cols = [c for c in df_out.columns if c not in likely_idx and pd.api.types.is_numeric_dtype(df_out[c])]

    return {"normalized": df_out, "unmapped": unmapped_df, "value_columns_used": val_cols}

def main():
    ap = argparse.ArgumentParser(description="Normalize financial tables and report unmapped row labels.")
    ap.add_argument("--input", required=True, help="Input tables CSV/Parquet")
    ap.add_argument("--metric_map", required=True, help="metric_map.yaml path")
    ap.add_argument("--row_col", default="row_label_raw", help="Column with raw row label")
    ap.add_argument("--stmt_col", default="statement_type", help="Optional statement type column")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--output_prefix", default="normalized", help="Filename prefix for outputs")
    ap.add_argument("--sep", default=",", help="CSV separator (if reading CSV)")
    ap.add_argument("--value_cols", nargs="*", help="Optional value column hints")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path, sep=args.sep)

    res = normalize_tables(df, metric_map_path=Path(args.metric_map),
                           row_col=args.row_col, stmt_col=args.stmt_col if args.stmt_col else None,
                           value_cols_hint=args.value_cols)

    norm_csv = out_dir / f"{args.output_prefix}_table_facts.csv"
    res["normalized"].to_csv(norm_csv, index=False)

    unmapped_csv = out_dir / f"{args.output_prefix}_unmapped_report.csv"
    res["unmapped"].to_csv(unmapped_csv, index=False)

    summary = {
        "input_rows": int(len(df)),
        "normalized_rows": int(len(res["normalized"])),
        "unmapped_unique_labels": int(len(res["unmapped"])),
        "normalized_out": str(norm_csv),
        "unmapped_out": str(unmapped_csv),
        "value_columns_used": res["value_columns_used"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
