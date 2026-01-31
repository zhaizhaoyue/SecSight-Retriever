#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

# -----------------------
# [TRANSLATED]
# -----------------------
PROJECT_ROOT        = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR   = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR  = PROJECT_ROOT / "data" / "clean"

# -----------------------
# [TRANSLATED] IO
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
                    print(f"    [WARN] {path.name} [TRANSLATED]{i}[TRANSLATED]：{e}")
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
    print(f"    [WARN] [TRANSLATED] pyarrow/fastparquet，[TRANSLATED] {out_path.name}。")
    return False

def save_jsonl(df: pd.DataFrame, out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def mirror_out_dir(in_file: Path, in_root: Path, out_root: Path) -> Path:
    """[TRANSLATED] processed [TRANSLATED] clean [TRANSLATED]（[TRANSLATED]，[TRANSLATED]）"""
    rel = in_file.parent.relative_to(in_root)  # e.g. AAPL/2023/10-Q_xxx
    out_dir = (out_root / rel).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

# -----------------------
# labels [TRANSLATED]（[TRANSLATED]；standard > terse > verbose）
# -----------------------
# -----------------------
# labels [TRANSLATED]（[TRANSLATED]/[TRANSLATED] + [TRANSLATED]）
# -----------------------

# -----------------------
# labels [TRANSLATED]（[TRANSLATED]/[TRANSLATED] + [TRANSLATED]）
# -----------------------
ROLE_PRIORITY = ["standard", "terse", "total", "documentation", "verbose"]  # UI [TRANSLATED]
ROLE_MAP = {
    "http://www.xbrl.org/2003/role/label": "standard",
    "http://www.xbrl.org/2003/role/terseLabel": "terse",
    "http://www.xbrl.org/2003/role/verboseLabel": "verbose",
    "http://www.xbrl.org/2003/role/documentation": "documentation",
    "http://www.xbrl.org/2003/role/totalLabel": "total",
}
LANG_PRIORITY = ["en-US", "en", "en-GB", "zh", "nl"]
LANG_NORM = {
    "en": "en", "en-us": "en-US", "en-gb": "en-GB",
    "zh": "zh", "zh-cn": "zh", "nl": "nl",
}

def norm_role(role_uri: Optional[str]) -> str:
    if not role_uri:
        return "standard"
    return ROLE_MAP.get(role_uri, role_uri.rsplit("/", 1)[-1])

def norm_lang(lang: Optional[str]) -> Optional[str]:
    if not lang:
        return None
    t = str(lang).strip().lower()
    return LANG_NORM.get(t, lang)

def to_tokens(*txts: Optional[str]) -> Optional[str]:
    """[TRANSLATED] token，NaN [TRANSLATED]；[TRANSLATED] None"""
    parts: List[str] = []
    for txt in txts:
        if txt is None:
            continue
        if isinstance(txt, float) and np.isnan(txt):
            continue
        s = re.sub(r"[^\w\s\-/%]+", " ", str(txt).lower()).strip()
        if s:
            parts.append(s)
    if not parts:
        return None
    # [TRANSLATED] + [TRANSLATED]
    s = " ".join(parts)
    s = re.sub(r"\s+", " ", s)
    return s or None



def build_preferred_labels(labels_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    [TRANSLATED]：
      A) [TRANSLATED] best：[TRANSLATED] label_best/label_best_role/label_best_lang
      B) [TRANSLATED] wide：[TRANSLATED] label_{role}_{lang}（[TRANSLATED] label_standard_en-US）
      C) [TRANSLATED]：concept + (label_text/label, label_role/role, lang)
    [TRANSLATED]：
      best_df：[TRANSLATED] (ticker, fy, form, accno, concept) [TRANSLATED]，[TRANSLATED] label_best/label_best_role/label_best_lang/label_doc/label_search_tokens
      wide_df：[TRANSLATED] wide [TRANSLATED]；[TRANSLATED] best [TRANSLATED]，[TRANSLATED]
    """
    if labels_df is None or labels_df.empty:
        cols_best = ["ticker","fy","form","accno","concept","label_best","label_best_role","label_best_lang","label_doc","label_search_tokens"]
        cols_wide = ["ticker","fy","form","accno","concept"]
        return pd.DataFrame(columns=cols_best), pd.DataFrame(columns=cols_wide)

    df = labels_df.copy()
    keycols = [c for c in ["ticker","fy","form","accno","concept"] if c in df.columns]

    ROLE_PRIORITY = ["standard", "terse", "total", "documentation", "verbose"]
    LANG_PRIORITY = ["en-US", "en", "en-GB", "zh", "nl"]
    role_order = {r:i for i,r in enumerate(ROLE_PRIORITY)}
    lang_order = {l:i for i,l in enumerate(LANG_PRIORITY)}

    def _tokens(*txts):
        parts=[]
        for t in txts:
            if t is None: continue
            if isinstance(t, float) and pd.isna(t): continue
            s = re.sub(r"[^\w\s\-/%]+", " ", str(t).lower()).strip()
            if s: parts.append(s)
        if not parts: return None
        return re.sub(r"\s+", " ", " ".join(parts)) or None

    # ---------- A) [TRANSLATED] best ----------
    if {"label_best","label_best_role","label_best_lang"}.issubset(df.columns):
        best_df = df.copy()
        if "label_doc" not in best_df.columns:
            best_df["label_doc"] = None
        if "label_search_tokens" not in best_df.columns:
            best_df["label_search_tokens"] = best_df.apply(lambda r: _tokens(r.get("label_best"), r.get("label_doc")), axis=1)
        keep = [c for c in ["ticker","fy","form","accno","concept","label_best","label_best_role","label_best_lang","label_doc","label_search_tokens"] if c in best_df.columns]
        for col in ["year","fy"]:
            if col in best_df.columns:
                best_df[col] = pd.to_numeric(best_df[col], errors="coerce").astype("Int64")
        wide_df = pd.DataFrame(columns=[c for c in ["ticker","fy","form","accno","concept"] if c in df.columns])
        return best_df[keep].drop_duplicates().reset_index(drop=True), wide_df

    # ---------- B) [TRANSLATED] wide ----------
    wide_like_cols = [c for c in df.columns if c.startswith("label_")]
    def _parse_col(c: str):
        m = re.match(r"^label_([A-Za-z]+)_(.+)$", c)
        if not m: return None
        return m.group(1).lower(), m.group(2)

    parsed = [(_parse_col(c), c) for c in wide_like_cols]
    if any(p[0] is not None for p in parsed):
        def _pick_best_from_wide(row: pd.Series):
            items=[]
            for (tpl, colname) in parsed:
                if tpl is None: continue
                role, lang = tpl
                val = row.get(colname)
                if pd.isna(val) or str(val).strip()=="":
                    continue
                items.append((role, lang, str(val)))
            if not items:
                return pd.Series({"label_best": None, "label_best_role": None, "label_best_lang": None,
                                  "label_doc": None, "label_search_tokens": None})
            doc_texts = [t for (r,l,t) in items if r=="documentation"]
            label_doc = max(doc_texts, key=len) if doc_texts else None
            items_sorted = sorted(items, key=lambda it: (role_order.get(it[0],999), lang_order.get(it[1],999)))
            role_b, lang_b, text_b = items_sorted[0]
            return pd.Series({
                "label_best": text_b,
                "label_best_role": role_b,
                "label_best_lang": lang_b,
                "label_doc": label_doc,
                "label_search_tokens": _tokens(text_b, label_doc),
            })

        picks = df.apply(_pick_best_from_wide, axis=1)
        best_df = pd.concat([df[keycols], picks], axis=1) if keycols else picks
        for col in ["year","fy"]:
            if col in best_df.columns:
                best_df[col] = pd.to_numeric(best_df[col], errors="coerce").astype("Int64")
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        return best_df.drop_duplicates().reset_index(drop=True), df  # wide [TRANSLATED]

    # ---------- C) [TRANSLATED] ----------
    # [TRANSLATED]
    if "label_text" not in df.columns and "label" in df.columns:
        df["label_text"] = df["label"]
    if "label_role" not in df.columns and "role" in df.columns:
        df["label_role"] = df["role"]
    if "lang" not in df.columns:
        df["lang"] = pd.NA

    # [TRANSLATED] role/lang
    def _norm_role(uri: Optional[str]) -> str:
        if not uri: return "standard"
        mp = {
            "http://www.xbrl.org/2003/role/label": "standard",
            "http://www.xbrl.org/2003/role/terseLabel": "terse",
            "http://www.xbrl.org/2003/role/verboseLabel": "verbose",
            "http://www.xbrl.org/2003/role/documentation": "documentation",
            "http://www.xbrl.org/2003/role/totalLabel": "total",
        }
        return mp.get(uri, uri.rsplit("/", 1)[-1])

    def _norm_lang(lang: Optional[str]) -> Optional[str]:
        if not lang: return None
        t = str(lang).strip().lower()
        mp = {"en":"en","en-us":"en-US","en-gb":"en-GB","zh":"zh","zh-cn":"zh","nl":"nl"}
        return mp.get(t, lang)

    df["label_role"] = df["label_role"].map(_norm_role)
    df["lang"] = df["lang"].map(_norm_lang)

    df["__role_rk"] = df["label_role"].map(lambda x: role_order.get(x, 999))
    df["__lang_rk"] = df["lang"].map(lambda x: lang_order.get(x, 999) if pd.notna(x) else 500)

    gcols = keycols or ["concept"]

    def _pick_best_long(g: pd.DataFrame) -> pd.Series:
        g2 = g.sort_values(["__role_rk","__lang_rk"])
        top = g2.iloc[0]
        # [TRANSLATED]：[TRANSLATED] documentation
        doc_texts = g[g["label_role"]=="documentation"]["label_text"].dropna().astype(str)
        label_doc = max(doc_texts, key=len) if not doc_texts.empty else None
        return pd.Series({
            "label_best": top["label_text"],
            "label_best_role": top.get("label_role"),
            "label_best_lang": top.get("lang"),
            "label_doc": label_doc,
            "label_search_tokens": to_tokens(top["label_text"]) if not label_doc else to_tokens(top["label_text"] + " " + label_doc)
        })
        return pd.Series({"label_best": top["label_text"], "label_best_role": top.get("label_role"), "label_best_lang": top.get("lang")})

    def _pick_doc_long(g: pd.DataFrame) -> Optional[str]:
        s = g[g["label_role"] == "documentation"]["label_text"].dropna().astype(str)
        return (max(s, key=len) if not s.empty else None)

    best_df = (
        df.groupby(gcols, dropna=False)
          .apply(_pick_best_long, include_groups=False)
          .reset_index()
    )
    doc_df = (
        df.groupby(gcols, dropna=False)
          .apply(lambda g: pd.Series({"label_doc": _pick_doc_long(g)}), include_groups=False)
          .reset_index()
    )
    best_df = best_df.merge(doc_df, on=gcols, how="left")
    best_df["label_search_tokens"] = best_df.apply(lambda r: _tokens(r.get("label_best"), r.get("label_doc")), axis=1)

    # wide：[TRANSLATED]
    df["lang_safe"] = df["lang"].fillna("none")
    df["colkey"] = "label_" + df["label_role"].astype(str) + "_" + df["lang_safe"].astype(str)
    wide_df = df.pivot_table(
        index=gcols,
        columns="colkey",
        values="label_text",
        aggfunc=lambda x: sorted(set([t for t in x if pd.notna(t) and str(t).strip()]))[0] if len(x)>0 else None
    ).reset_index()

    for col in ["year","fy"]:
        if col in best_df.columns:
            best_df[col] = pd.to_numeric(best_df[col], errors="coerce").astype("Int64")
        if col in wide_df.columns:
            wide_df[col] = pd.to_numeric(wide_df[col], errors="coerce").astype("Int64")

    return best_df.drop_duplicates().reset_index(drop=True), wide_df


# -----------------------
# [TRANSLATED]：calculation_edges
# -----------------------
def clean_calculation_edges(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # [TRANSLATED]
    for c in ["parent_concept","child_concept","weight","order","linkrole",
              "ticker","year","fy","fq","form","accno","doc_date","file_type","source_path"]:
        if c not in df.columns:
            df[c] = pd.NA
    # [TRANSLATED]
    def _to_float(x):
        try: return float(x)
        except Exception: return np.nan
    df["weight"] = df["weight"].apply(_to_float)
    df["order"]  = df["order"].apply(_to_float)

    for col in ["year", "fy"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    # [TRANSLATED]
    keep = ["parent_concept","child_concept","weight","order","linkrole",
            "ticker","year","fy","fq","form","accno","doc_date","file_type","source_path"]
    df = df[keep].drop_duplicates().reset_index(drop=True)
    return df

# -----------------------
# [TRANSLATED]：definition_arcs
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
# [TRANSLATED]：labels（[TRANSLATED] labels.parquet / labels.jsonl → [TRANSLATED]）
# -----------------------
def clean_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        cols_best = ["ticker","fy","form","accno","concept","label_best","label_best_role","label_best_lang","label_doc","label_search_tokens"]
        cols_wide = ["ticker","fy","form","accno","concept"]
        return pd.DataFrame(columns=cols_best), pd.DataFrame(columns=cols_wide)
    best_df, wide_df = build_preferred_labels(df)
    return best_df, wide_df

# -----------------------
# [TRANSLATED]
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Clean calculation_edges / definition_arcs / labels with processed→clean mirroring")
    ap.add_argument("--root", type=str, default=None, help="[TRANSLATED]（[TRANSLATED] PROJECT_ROOT/data/processed）")
    ap.add_argument("--dry-run", action="store_true", help="[TRANSLATED]")
    args = ap.parse_args()

    in_root  = Path(args.root).expanduser().resolve() if args.root else DEFAULT_INPUT_DIR.resolve()
    out_root = DEFAULT_OUTPUT_DIR.resolve()

    if not in_root.exists():
        print(f"[INFO] [TRANSLATED]：{in_root}")
        return

    # [TRANSLATED]
    files_calc = list(in_root.rglob("calculation_edges.jsonl")) + list(in_root.rglob("calculation_edges.parquet"))
    files_def  = list(in_root.rglob("definition_arcs.jsonl"))  + list(in_root.rglob("definition_arcs.parquet"))
    files_lab  = list(in_root.rglob("labels.jsonl"))           + list(in_root.rglob("labels.parquet"))

    total = len(files_calc) + len(files_def) + len(files_lab)
    if total == 0:
        print(f"[INFO] [TRANSLATED] {in_root} [TRANSLATED] calculation_edges/definition_arcs/labels")
        return

    print(f"[INFO] [TRANSLATED]：{in_root}")
    print(f"[INFO] [TRANSLATED]：{out_root}（[TRANSLATED] processed [TRANSLATED]）")
    print(f"[INFO] [TRANSLATED]：calc={len(files_calc)}, def={len(files_def)}, labels={len(files_lab)}")

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
            # [TRANSLATED]
            ok = try_write_parquet(cleaned, out_dir / "calculation_edges.parquet")
            save_jsonl(cleaned, out_dir / "calculation_edges.jsonl")
            print(f"    -> [TRANSLATED]：{'calculation_edges.parquet, ' if ok else ''}calculation_edges.jsonl  rows={len(cleaned)}  dir={out_dir}")
        except Exception as e:
            print(f"    [WARN] [TRANSLATED]：{p}\n{e}")

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
            print(f"    -> [TRANSLATED]：{'definition_arcs.parquet, ' if ok else ''}definition_arcs.jsonl  rows={len(cleaned)}  dir={out_dir}")
        except Exception as e:
            print(f"    [WARN] [TRANSLATED]：{p}\n{e}")

    # 3) labels → [TRANSLATED]（[TRANSLATED]→[TRANSLATED]）
    for i, p in enumerate(files_lab, 1):
        try:
            print(f"[labs {i}/{len(files_lab)}] {p}")
            raw = read_table(p)
            best_df, wide_df = clean_labels(raw)
            out_dir = mirror_out_dir(p, in_root, out_root)
            if args.dry_run:
                print(f"    -> dry-run best_rows={len(best_df)} wide_rows={len(wide_df)} dir={out_dir}")
                continue

            # [TRANSLATED] best
            ok1 = try_write_parquet(best_df, out_dir / "labels_best.parquet")
            save_jsonl(best_df, out_dir / "labels_best.jsonl")

            # [TRANSLATED] wide（[TRANSLATED]：[TRANSLATED]，[TRANSLATED]）
            ok2 = try_write_parquet(wide_df, out_dir / "labels_wide.parquet")
            save_jsonl(wide_df, out_dir / "labels_wide.jsonl")

            # [TRANSLATED]：labels.* [TRANSLATED] best（[TRANSLATED] join [TRANSLATED]）
            ok3 = try_write_parquet(best_df, out_dir / "labels.parquet")
            save_jsonl(best_df, out_dir / "labels.jsonl")

            print(f"    -> [TRANSLATED]："
                f"{'labels_best.parquet, ' if ok1 else ''}labels_best.jsonl; "
                f"{'labels_wide.parquet, ' if ok2 else ''}labels_wide.jsonl; "
                f"{'labels.parquet, ' if ok3 else ''}labels.jsonl  dir={out_dir}")

        except Exception as e:
            print(f"    [WARN] [TRANSLATED]：{p}\n{e}")

    print("[DONE] [TRANSLATED]。")

if __name__ == "__main__":
    main()
