# Write an upgraded text_clean.py that supports input/output directories and config-based defaults
from pathlib import Path
code = r"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
import sys, re, json, html, argparse, os
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import pandas as pd

try:
    import yaml
except Exception:
    yaml = None  # optional

# -----------------------------
# Config & Defaults
# -----------------------------

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
DEFAULT_INPUT_DIR = Path("data/processed")
DEFAULT_OUTPUT_DIR = Path("data/clean")

SCALE_MAP = {
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "million": 1e6, "mn": 1e6, "m": 1e6,
    "thousand": 1e3, "k": 1e3,
}

CURRENCY_TOKENS = {
    "$": "USD", "usd": "USD", "us$": "USD",
    "eur": "EUR", "€": "EUR",
    "cny": "CNY", "rmb": "CNY", "¥": "CNY",
    "jpy": "JPY", "¥¥": "JPY",
    "gbp": "GBP", "£": "GBP",
}

RE_NUMBER = re.compile(r"""
    (?P<prefix>\$|€|£|¥)?
    (?P<num>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)
    \s*
    (?P<unit>billion|million|thousand|bn|mn|m|k)?
""", re.IGNORECASE | re.VERBOSE)

RE_PERCENT = re.compile(r"(?P<pct>\d+(?:\.\d+)?)\s*%")
RE_CURRENCY_WORD = re.compile(r"\b(USD|EUR|CNY|RMB|JPY|GBP)\b", re.IGNORECASE)
RE_HEADER_NOISE = re.compile(r"""
    ^\s*(table\s+of\s+contents|contents|index|page\s+\d+|exhibit\s+\d+|signature[s]?|
    united\s+states|securities\s+and\s+exchange\s+commission)\s*$
""", re.IGNORECASE | re.VERBOSE)
RE_SHORT_PUNC = re.compile(r"^[\W_]+$")  # only punctuation/whitespace

# -----------------------------
# Utilities
# -----------------------------

def read_config(cfg_path: Path) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if cfg_path.exists() and yaml is not None:
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            # try to hydrate paths if present
            paths = raw.get("paths") or {}
            if "processed_dir" in paths:
                cfg["input_dir"] = Path(paths["processed_dir"])
            if "clean_dir" in paths:
                cfg["output_dir"] = Path(paths["clean_dir"])
        except Exception:
            pass
    return cfg

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def unescape_and_normalize(text: str) -> str:
    if text is None:
        return ""
    s = html.unescape(text)
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    return s.strip()

def is_noise_line(s: str, min_chars: int) -> bool:
    if not s:
        return True
    if len(s) < min_chars:
        if RE_NUMBER.search(s) or RE_PERCENT.search(s):
            pass
        else:
            return True
    if RE_HEADER_NOISE.search(s):
        return True
    if RE_SHORT_PUNC.match(s):
        return True
    return False

def split_sentences(s: str) -> List[str]:
    s = s.replace("。", ".").replace("；", ";").replace("！", "!").replace("？", "?")
    s = re.sub(r"([\.!?;：:])(\s|$)", r"\1\n", s)
    parts = [p.strip() for p in s.split("\n") if p.strip()]
    return parts

def parse_percent_tokens(s: str) -> Tuple[List[str], List[Optional[float]]]:
    raw, vals = [], []
    for m in RE_PERCENT.finditer(s):
        raw.append(m.group(0))
        try:
            vals.append(float(m.group("pct")) / 100.0)
        except Exception:
            vals.append(None)
    return raw, vals

def parse_currency_tokens(s: str) -> List[str]:
    tokens = set()
    for sym, code in CURRENCY_TOKENS.items():
        if sym.lower() in s.lower():
            tokens.add(code)
    for m in RE_CURRENCY_WORD.finditer(s):
        tokens.add(m.group(0).upper().replace("RMB", "CNY"))
    return sorted(tokens)

def _to_float(num_str: str) -> Optional[float]:
    try:
        return float(num_str.replace(",", ""))
    except Exception:
        return None

def parse_number_tokens(s: str) -> Tuple[List[str], List[Optional[float]]]:
    raw, vals = [], []
    for m in RE_NUMBER.finditer(s):
        token = m.group(0).strip()
        num = _to_float(m.group("num"))
        unit = m.group("unit").lower() if m.group("unit") else None
        scale = SCALE_MAP.get(unit, 1.0)
        if num is not None:
            num *= scale
        raw.append(token)
        vals.append(num)
    return raw, vals

def process_jsonl_line(line: Dict[str, Any], *, min_chars: int) -> List[Dict[str, Any]]:
    idx = line.get("idx")
    tag = line.get("tag")
    text = line.get("text") or ""
    text_norm = unescape_and_normalize(text)

    if is_noise_line(text_norm, min_chars=min_chars):
        return []

    out_rows: List[Dict[str, Any]] = []
    for sent in split_sentences(text_norm):
        if is_noise_line(sent, min_chars=min_chars):
            continue
        pct_raw, pct_vals = parse_percent_tokens(sent)
        num_raw, num_vals = parse_number_tokens(sent)
        currs = parse_currency_tokens(sent)

        out_rows.append({
            "idx_source": idx,
            "tag": tag,
            "text_raw": text,
            "text": sent,
            "length": len(sent),
            "numbers_raw": num_raw,
            "numbers": num_vals,
            "percents_raw": pct_raw,
            "percents": pct_vals,
            "currencies": currs,
            "has_numbers": bool(num_raw),
            "has_percents": bool(pct_raw),
        })
    return out_rows

def clean_one_file(input_path: Path, out_jsonl: Path, out_parquet: Optional[Path], *, min_chars: int = 30) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    total = 0
    with open(input_path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rows.extend(process_jsonl_line(obj, min_chars=min_chars))

    ensure_dir(out_jsonl)
    with open(out_jsonl, "w", encoding="utf-8") as f_out:
        for r in rows:
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

    df = pd.DataFrame(rows)
    parquet_written = None
    csv_fallback = None
    if out_parquet is not None:
        try:
            ensure_dir(out_parquet)
            df.to_parquet(out_parquet, index=False)
            parquet_written = str(out_parquet)
        except Exception:
            csv_fallback = str(out_parquet.with_suffix(".csv"))
            df.to_csv(csv_fallback, index=False)

    return {
        "input_file": str(input_path),
        "sentences": len(rows),
        "out_jsonl": str(out_jsonl),
        "out_parquet": parquet_written,
        "out_csv_fallback": csv_fallback,
    }

def main():
    cfg = read_config(DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Clean text.jsonl into sentence-level corpus with numeric/percent extraction.")
    in_grp = parser.add_mutually_exclusive_group(required=False)
    in_grp.add_argument("--input_file", help="Path to a single text.jsonl")
    in_grp.add_argument("--input_dir", help="Directory to scan for text.jsonl files recursively")
    parser.add_argument("--output_jsonl", help="Output JSONL file (only when --input_file is used)")
    parser.add_argument("--output_parquet", help="Output Parquet file (only when --input_file is used; falls back to CSV)")
    parser.add_argument("--output_dir", help="Base dir for outputs when --input_dir is used; mirror input structure under this dir")
    parser.add_argument("--pattern", default="text.jsonl", help="Filename pattern to search under input_dir (default: text.jsonl)")
    parser.add_argument("--min_chars", type=int, default=30, help="Minimum chars to keep unless contains numbers/percents.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # resolve defaults
    input_dir = Path(args.input_dir) if args.input_dir else cfg.get("input_dir", DEFAULT_INPUT_DIR)
    output_dir = Path(args.output_dir) if args.output_dir else cfg.get("output_dir", DEFAULT_OUTPUT_DIR)

    summary: Dict[str, Any] = {}
    results: List[Dict[str, Any]] = []

    if args.input_file:  # single file mode
        in_file = Path(args.input_file)
        out_jsonl = Path(args.output_jsonl) if args.output_jsonl else (in_file.with_name("text_corpus.jsonl"))
        out_parquet = Path(args.output_parquet) if args.output_parquet else (in_file.with_name("text_corpus.parquet"))
        res = clean_one_file(in_file, out_jsonl, out_parquet, min_chars=args.min_chars)
        results.append(res)
        summary = {"mode": "single_file"}

    else:  # directory mode
        # find all files named `pattern` under input_dir
        pattern = args.pattern
        files = list(input_dir.rglob(pattern))
        if args.verbose:
            print(f"[INFO] Found {len(files)} files under {input_dir} matching {pattern}")
        for f in files:
            # build mirrored relative path
            rel = f.relative_to(input_dir)  # e.g., AAPL/2023/10-K.../text.jsonl
            out_base_dir = output_dir / rel.parent  # mirror parent dir
            out_jsonl = out_base_dir / "text_corpus.jsonl"
            out_parquet = out_base_dir / "text_corpus.parquet"
            res = clean_one_file(f, out_jsonl, out_parquet, min_chars=args.min_chars)
            results.append(res)
        summary = {"mode": "directory", "input_dir": str(input_dir), "output_dir": str(output_dir), "files": len(results)}

    # Aggregate report
    agg = {
        "summary": summary,
        "results": results,
    }
    print(json.dumps(agg, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

p = Path("/mnt/data/text_clean_v2.py")
p.write_text(code, encoding="utf-8")
print(f"Wrote {p}")
