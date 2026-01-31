#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception as e:
                rows.append({"__parse_error__": str(e), "__line__": i})
    return pd.DataFrame(rows)


def _to_float_maybe(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        try:
            return float(x)
        except Exception:
            return None
    s = str(x).strip().replace(",", "")
    if s.endswith("%"):
        s = s[:-1]
        try:
            return float(s) / 100.0
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None


def upgrade_one(facts_path: Path) -> Optional[Path]:
    df = _read_jsonl(facts_path)
    if df.empty:
        return None

    # Ensure concept column
    if "concept" not in df.columns:
        df["concept"] = df.get("qname")

    # Ensure unit column (prefer normalized)
    if "unit" not in df.columns:
        df["unit"] = df.get("unit_normalized")
        if "unitRef" in df.columns:
            df["unit"] = df["unit"].where(df["unit"].notna(), df["unitRef"])  

    # Build value_num_clean with robust percent handling
    # Base numeric from value_num or parse value_display/value_raw
    base_num = df.get("value_num")
    if base_num is None:
        base_num = df.get("value")
    if base_num is None:
        base_num = df.get("value_display")
    df["__tmp_value_num"] = base_num.apply(_to_float_maybe) if base_num is not None else None

    # Percent detection
    qname_lower = df.get("qname", pd.Series([None]*len(df))).astype(str).str.lower()
    value_display = df.get("value_display", pd.Series([None]*len(df))).astype(str)
    unit_family = df.get("unit_family", pd.Series([None]*len(df)))
    is_percent = (
        unit_family.fillna("").astype(str).str.lower().eq("percent")
        | value_display.fillna("").str.contains("%", regex=False)
        | qname_lower.str.contains("percent|percentage", regex=True)
    )

    # If display has %, always parse from display to ratio
    from_display_ratio = value_display.fillna("")
    mask_display_pct = from_display_ratio.str.contains("%", regex=False)
    parsed_from_display = from_display_ratio.str.replace("%", "", regex=False).apply(_to_float_maybe)
    parsed_from_display = parsed_from_display.where(~mask_display_pct, parsed_from_display)
    if parsed_from_display is not None:
        # Convert percentage number to ratio (already handled in _to_float_maybe when % present)
        df["__pct_from_display"] = parsed_from_display
    else:
        df["__pct_from_display"] = None

    # Start with base numeric
    df["value_num_clean"] = df["__tmp_value_num"]

    # For percent rows without explicit '%', only divide when abs(value)>1
    need_div = (
        is_percent
        & (~mask_display_pct)
        & df["value_num_clean"].notna()
        & (df["value_num_clean"].abs() > 1.0)
    )
    df.loc[need_div, "value_num_clean"] = df.loc[need_div, "value_num_clean"] / 100.0

    # For rows with '%', prefer parsed_from_display ratio
    use_display = is_percent & mask_display_pct & df["__pct_from_display"].notna()
    df.loc[use_display, "value_num_clean"] = df.loc[use_display, "__pct_from_display"]

    # Cleanup temp columns
    for c in ["__tmp_value_num", "__pct_from_display"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # ---- [TRANSLATED] TextBlock：[TRANSLATED]；[TRANSLATED] ----
    def is_textblock_row(row: pd.Series) -> bool:
        qn = str(row.get("qname") or "")
        return "textblock" in qn.lower()

    def html_to_text_lines(html: str) -> list[str]:
        soup = BeautifulSoup(html, "lxml")
        lines: list[str] = []
        # [TRANSLATED]
        for tr in soup.find_all("tr"):
            t = " ".join(td.get_text(" ", strip=True) for td in tr.find_all(["td","th"]))
            t = re.sub(r"\s+", " ", t).strip()
            if t:
                lines.append(t)
        # [TRANSLATED]，[TRANSLATED]
        if not lines:
            txt = soup.get_text("\n", strip=True)
            for ln in txt.splitlines():
                ln = re.sub(r"\s+", " ", ln).strip()
                if ln:
                    lines.append(ln)
        return lines

    NUM_PAT = re.compile(r"([($]?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?%?)")

    extracted_rows: list[dict] = []
    drop_idx: set[int] = set()

    if "qname" in df.columns:
        for idx, row in df.iterrows():
            if not is_textblock_row(row):
                continue
            # [TRANSLATED] HTML [TRANSLATED]
            html = None
            for c in ("value_raw", "value_display"):
                v = row.get(c)
                if isinstance(v, str) and ("<" in v and ">" in v):
                    html = v
                    break
            if not html:
                # [TRANSLATED] HTML [TRANSLATED] textblock，[TRANSLATED]，[TRANSLATED]
                drop_idx.add(idx)
                continue

            lines = html_to_text_lines(html)
            found_any = False
            for ln in lines:
                for m in NUM_PAT.finditer(ln):
                    tok = m.group(1)
                    tok_clean = tok.strip()
                    # [TRANSLATED]/[TRANSLATED]/[TRANSLATED]/%
                    is_pct = tok_clean.endswith("%")
                    has_dollar = tok_clean.startswith("$") or " $" in ln
                    s = tok_clean.replace("$", "").replace(",", "")
                    neg = s.startswith("(") and s.endswith(")")
                    if neg:
                        s = s[1:-1]
                    s = s[:-1] if is_pct else s
                    try:
                        val = float(s)
                    except Exception:
                        continue
                    if is_pct:
                        val = val / 100.0

                    out = row.to_dict()
                    # [TRANSLATED]
                    out["extracted_from_textblock"] = True
                    out["parent_concept"] = out.get("qname")
                    out["concept"] = out.get("concept") or out.get("qname")
                    out["value_display"] = tok_clean
                    out["value_num_clean"] = val
                    out["value_num"] = val
                    # [TRANSLATED]（[TRANSLATED]）
                    if is_pct:
                        out["unit"] = "%"
                        out["unit_family"] = "percent"
                    elif has_dollar:
                        out["unit"] = "$"
                    # [TRANSLATED]
                    sig = str(out.get("dims_signature") or "")
                    if sig:
                        sig = sig + "|extracted=textblock"
                    else:
                        sig = "extracted=textblock"
                    out["dims_signature"] = sig
                    # [TRANSLATED]（[TRANSLATED]）
                    out["text_excerpt"] = (ln[:256] + "…") if len(ln) > 256 else ln

                    extracted_rows.append(out)
                    found_any = True

            # [TRANSLATED] TextBlock [TRANSLATED]
            drop_idx.add(idx)

    # [TRANSLATED] textblock [TRANSLATED]，[TRANSLATED]
    if drop_idx:
        df = df.drop(index=list(drop_idx))
    if extracted_rows:
        df = pd.concat([df, pd.DataFrame(extracted_rows)], ignore_index=True)

    # Write fact.jsonl next to facts.jsonl
    out_path = facts_path.with_name("fact.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            rec = r.to_dict()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return out_path


def find_facts_files(root: Path) -> list[Path]:
    return sorted(root.rglob("facts.jsonl"))


def main():
    ap = argparse.ArgumentParser(description="Upgrade processed facts.jsonl → fact.jsonl with improved fields")
    ap.add_argument("--root", default="data/processed", help="Root directory to scan")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    files = find_facts_files(root)
    print(f"[INFO] Found {len(files)} facts.jsonl files under {root}")
    ok = 0
    for i, p in enumerate(files, 1):
        try:
            out = upgrade_one(p)
            if out:
                ok += 1
                print(f"[{i}/{len(files)}] -> {out}")
        except Exception as e:
            print(f"[WARN] failed on {p}: {e}")
    print(f"[DONE] Upgraded {ok}/{len(files)} files.")


if __name__ == "__main__":
    main()


