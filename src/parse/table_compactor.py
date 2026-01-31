#555
##compact tables [TRANSLATED] HTML [TRANSLATED] <table> → [TRANSLATED] CSV/Markdown
from __future__ import annotations
import argparse, json, re
from io import StringIO
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None

from bs4 import XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
# -------- Regexes --------
UNIT_PATTERNS = [
    r"\b(in|amounts\s+in)\s+thousands\b",
    r"\b(in|amounts\s+in)\s+millions\b",
    r"\b(in|amounts\s+in)\s+billions\b",
    r"\bthousands\b",
    r"\bmillions\b",
    r"\bbillions\b",
]
UNIT_RE = re.compile("|".join(UNIT_PATTERNS), re.I)

CURRENCY_WORDS = r"USD|EUR|GBP|JPY|CNY|RMB|HKD|AUD|CAD|CHF|NTD|TWD|SGD"
CURR_RE = re.compile(rf"\b({CURRENCY_WORDS})\b", re.I)

SYMBOL_MAP = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "[TRANSLATED]": "CNY",
    "[TRANSLATED]": "CNY",
    "￥": "CNY",
}

PERIOD_RE = re.compile(r"(20\d{2}|19\d{2}|Q[1-4]|FY|FQ|Current|Prior|Previous)", re.I)
STRICT_NAME_RE = re.compile(r"^(?:[A-Z]{2,3})_([A-Z.\-]+)_(\d{4})_([0-9A-Z\-]+)_([0-9A-Za-z\-]+)\.html$", re.I)
ACCNO_RE       = re.compile(r"\d{10}-\d{2}-\d{6}|\d{18,}")
YEAR_RE        = re.compile(r"20\d{2}|19\d{2}")
FORM_RE        = re.compile(r"(10-K|10-Q|8-K|20-F|40-F|6-K)", re.I)
TICKER_RE      = re.compile(r"[A-Z]{1,6}")

def _clean_cell(x):
    if isinstance(x, str):
        x = x.strip()
        return (None if x == "" else x)
    return x

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--max_rows", type=int, default=120)
    ap.add_argument("--max_cols", type=int, default=6)
    ap.add_argument("--to_csv", action="store_true")
    ap.add_argument("--to_md", action="store_true")
    ap.add_argument("--both", action="store_true")
    return ap.parse_args()

def extract_tables_from_html(html: str) -> List[str]:
    if BeautifulSoup is None:
        parts = html.split("<table")
        tabs = []
        for i in range(1, len(parts)):
            frag = "<table" + parts[i]
            end = frag.lower().find("</table>")
            if end != -1:
                tabs.append(frag[:end+8])
        return tabs
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")
    return [str(t) for t in tables]

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)
    def _clean_cell(x):
        if isinstance(x, str):
            x = x.strip()
            return None if x == "" else x
        return x
    if hasattr(pd.DataFrame, "map"):   # pandas >= 2.2
        df = df.map(_clean_cell)
    else:
        df = df.applymap(_clean_cell)

    return df

def dataframe_from_html_table(html_table: str) -> Optional[pd.DataFrame]:
    try:
        dfs = pd.read_html(StringIO(html_table))  # [TRANSLATED] StringIO [TRANSLATED]
        if not dfs:
            return None
        return max(dfs, key=lambda d: (d.shape[0], d.shape[1]))
    except Exception:
        return None

def is_numeric_like(v: Any) -> bool:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str):
        s2 = v.replace(",", "").replace("(", "").replace(")", "").replace("$", "").strip()
        try:
            float(s2)
            return True
        except Exception:
            return False
    return False

def choose_account_col(df: pd.DataFrame) -> str:
    scores = {}
    for col in df.columns:
        nonnum_tokens = 0
        for v in df[col].head(50).tolist():
            if isinstance(v, str) and re.search(r"[A-Za-z]", v) and not is_numeric_like(v):
                nonnum_tokens += 1
        scores[col] = nonnum_tokens
    return max(scores, key=scores.get)

def parse_period_weight(header: str) -> float:
    h = header.upper()
    m_year = re.search(r"(20\d{2}|19\d{2})", h)
    year = int(m_year.group(0)) if m_year else None
    m_q = re.search(r"Q([1-4])", h)
    quarter = int(m_q.group(1)) if m_q else None
    if "CURRENT" in h and year is None:
        base = 3000
    elif "PRIOR" in h or "PREVIOUS" in h:
        base = 1000
    else:
        base = 0
    return (year or 0) * 10 + (quarter or 0) + base

def humanize_headers(cols: list, unit: str, currency: Optional[str]) -> list:
    suffix = []
    if currency:
        suffix.append(currency)
    if unit in {"1e3","1e6","1e9"}:
        suffix.append({"1e3":"thousands","1e6":"millions","1e9":"billions"}[unit])
    suffix_str = f" ({', '.join(suffix)})" if suffix else ""
    return [f"{str(c)}{suffix_str}" for c in cols]

def guess_unit_currency_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    unit = None
    currency = None
    if not text:
        return unit, currency
    # unit words
    m_u = UNIT_RE.search(text)
    if m_u:
        s = m_u.group(0).lower()
        if "million" in s:
            unit = "1e6"
        elif "billion" in s:
            unit = "1e9"
        elif "thousand" in s:
            unit = "1e3"
    # currency words
    m_c = CURR_RE.search(text)
    if m_c:
        currency = m_c.group(0).upper()
    else:
        # currency symbols
        for sym, cur in SYMBOL_MAP.items():
            if sym in text:
                currency = cur
                break
    return unit, currency

def guess_unit_currency(html: str, df: Optional[pd.DataFrame]) -> Tuple[str, Optional[str]]:
    # 1) scan html head part and whole doc for hints
    unit, currency = guess_unit_currency_from_text(html[:6000])
    if not unit or not currency:
        unit2, currency2 = guess_unit_currency_from_text(html)
        unit = unit or unit2
        currency = currency or currency2
    # 2) if still missing, look at header row & first few rows of df
    if df is not None:
        header_text = " ".join(map(str, df.columns.tolist()))
        u3, c3 = guess_unit_currency_from_text(header_text)
        unit = unit or u3
        currency = currency or c3
        # scan first few rows for currency symbols
        if not currency:
            sample = " ".join([
                " ".join(map(lambda x: str(x), df.iloc[i].tolist()))
                for i in range(min(5, len(df)))
            ])
            _, c4 = guess_unit_currency_from_text(sample)
            currency = currency or c4
    # defaults
    unit = unit or "1"
    return unit, currency

def pick_and_rename(df: pd.DataFrame, max_cols: int, unit: str, currency: Optional[str]) -> Tuple[pd.DataFrame, str, list]:
    account_col = choose_account_col(df)
    # select candidate period-like columns
    candidates = []
    for col in df.columns:
        if col == account_col:
            continue
        header = str(col)
        numeric_ratio = sum(is_numeric_like(x) for x in df[col].tolist()[:50]) / max(1, min(50, len(df[col])))
        if numeric_ratio > 0.5 or PERIOD_RE.search(header):
            candidates.append(col)
    if not candidates:
        candidates = df.columns.tolist()[-2:]
    # keep latest two by header recency
    period_cols = sorted(candidates, key=lambda c: parse_period_weight(str(c)), reverse=True)[:2]
    selected_cols = [account_col] + period_cols[: (max_cols - 1)]
    out = df[selected_cols].copy()
    # rename period headers to include unit/currency
    pretty_periods = humanize_headers(period_cols, unit, currency)
    new_cols = [str(account_col)] + pretty_periods
    out.columns = new_cols
    return out, str(account_col), list(map(str, period_cols))

def loose_parse_from_name_and_parents(path: Path) -> Tuple[str, str, str, str]:
    fname = path.name
    m = STRICT_NAME_RE.match(fname)
    if m:
        ticker, year, form, accno = m.groups()
        return ticker, year, form.upper(), accno

    text = str(path)
    accno  = (ACCNO_RE.search(text) or ACCNO_RE.search(fname))
    year   = (YEAR_RE.search(text) or YEAR_RE.search(fname))
    form   = (FORM_RE.search(text) or FORM_RE.search(fname))
    ticker = None
    for p in [path.parent, *path.parents]:
        m_t = TICKER_RE.match(p.name)
        if m_t:
            ticker = m_t.group(0)
            break
    if not ticker:
        m_t = TICKER_RE.search(fname)
        ticker = m_t.group(0) if m_t else "UNKNOWN"

    accno  = accno.group(0) if accno else fname.replace(".html","")
    year   = year.group(0) if year else "UNKNOWN"
    form   = form.group(0).upper() if form else "UNKNOWN"
    return ticker, year, form, accno

def to_markdown(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)

def main():
    args = parse_args()
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not (args.to_csv or args.to_md or args.both):
        args.both = True

    html_files = list(raw_root.rglob("*.html"))
    print(f"[info] Found {len(html_files)} HTML files under {raw_root}")

    total_files = 0
    total_tables = 0
    total_parsed = 0
    total_failed = 0

    for html_path in html_files:
        total_files += 1
        ticker, year, form, accno = loose_parse_from_name_and_parents(html_path)

        out_dir = out_root / ticker / year / f"{form}_{accno}"
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_html_out_dir = out_dir / "_raw_tables"
        raw_html_out_dir.mkdir(parents=True, exist_ok=True)

        html = html_path.read_text(encoding="utf-8", errors="ignore")
        tables = extract_tables_from_html(html)

        seen = len(tables)
        total_tables += seen
        parsed = 0
        failed = 0

        for i, t in enumerate(tables):
            df = dataframe_from_html_table(t)
            if df is None:
                failed += 1
                total_failed += 1
                (raw_html_out_dir / f"table_{i}.html").write_text(t, encoding="utf-8")
                continue

            df = clean_dataframe(df)
            unit, currency = guess_unit_currency(html, df)
            df_sel, account_col, period_cols_raw = pick_and_rename(df, args.max_cols, unit, currency)
            df_sel = df_sel.head(args.max_rows)

            base = out_dir / f"table_{i}"
            meta = {
                "type": "table_compact",
                "ticker": ticker,
                "year": year,
                "form": form,
                "accno": accno,
                "source_path": str(html_path),
                "section": None,
                "page_no": None,
                "unit_multiplier": unit,
                "currency": currency,
                "account_col": account_col,
                "period_cols_raw": period_cols_raw,
                "rows": int(df_sel.shape[0]),
                "cols": [str(c) for c in df_sel.columns],
            }
            (base.parent / f"{base.name}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            if args.to_csv or args.both:
                df_sel.to_csv(f"{base}.csv", index=False)

            if args.to_md or args.both:
                md = to_markdown(df_sel)
                footer = f"**Unit:** {unit}  **Currency:** {currency or 'NA'}  **Source:** {html_path.name}"
                (base.parent / f"{base.name}.md").write_text(md + "\n" + footer + "\n", encoding="utf-8")

            parsed += 1
            total_parsed += 1

        print(f"[info] {html_path.name}: tables={seen}, parsed={parsed}, failed={failed} -> {out_dir}")

    print(f"[SUMMARY] files={total_files}, tables_seen={total_tables}, tables_parsed={total_parsed}, tables_failed={total_failed}, out_root={out_root}")

if __name__ == "__main__":
    main()



'''
python src/parse/table_compactor.py `
  --raw_root data/raw_reports/standard `
  --out_root data/compact_tables `
  --both
'''
