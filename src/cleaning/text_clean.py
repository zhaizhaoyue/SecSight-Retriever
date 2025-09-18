from pathlib import Path
import sys, re, json, html, argparse, os, time, logging
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import yaml

# =============================
# Logging
# =============================
def setup_logger(level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(lvl)

log = logging.getLogger(__name__)

# =============================
# Constants / Defaults
# =============================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_INPUT_DIR   = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR  = PROJECT_ROOT / "data" / "clean"

DEFAULT_MAX_SENT_LEN  = 2000
DEFAULT_HARD_WRAP_LEN = 2800
DEFAULT_HEARTBEAT_EVERY = 1000
# ---- 文本检索模式：放松去噪，避免过度丢失上下文，关闭额外的 RAG 轻滤
TEXT_RETRIEVAL_ONLY = True

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

# =============================
# Regexes
# =============================
RE_CIK = re.compile(r"\b0\d{9}\b")
RE_DATE_ISO = re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b")
MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
RE_DATE_LONG = re.compile(rf"\b(?:{MONTHS})\s+\d{{1,2}},\s*(?:19|20)\d{{2}}\b", re.IGNORECASE)
RE_COMMISSION_FILE = re.compile(r"\bCommission\s+File\s+Number\b", re.I)
RE_XBRL_QNAME = re.compile(r"\b[a-z][a-z0-9\-]*:[A-Za-z0-9\.]+Member\b", re.IGNORECASE)

# 更温和的软切分：去掉 : ; — - 的通用切分，避免把行项目切碎
RE_SOFT_SPLIT = re.compile(
    r"(?:(?<=\.)\s+)|(?<=\?)\s+|(?<=!)\s+"
    r"|(?=\bus-gaap:)|(?=" + RE_DATE_ISO.pattern + r")",
    re.IGNORECASE
)

RE_NUMBER = re.compile(r"""
    (?P<prefix>[$€£¥])?
    (?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)
    \s*
    (?P<unit>billion|million|thousand|bn|mn|m|k)?
""", re.IGNORECASE | re.VERBOSE)

RE_PERCENT = re.compile(r"(?P<pct>\d+(?:\.\d+)?)\s*%")
RE_CURRENCY_WORD = re.compile(r"\b(USD|EUR|CNY|RMB|JPY|GBP)\b", re.IGNORECASE)
RE_ISO_DUR = re.compile(r"\bP(?:(?P<years>\d+)Y)?(?:(?P<months>\d+)M)?(?:(?P<days>\d+)D)?\b", re.IGNORECASE)
RE_MEMBER_NEARBY = re.compile(r"\b[A-Za-z0-9:\.\-]*Member\b")
RE_MONTH_CONTEXT = re.compile(r"\b(month|months|preceding\s+\d+\s+months?)\b", re.IGNORECASE)
RE_LEGAL_RULE = re.compile(r"\b(Rule|Item)\s+\d+[a-z0-9\-]*\b", re.IGNORECASE)
RE_LEGAL_SECTION = re.compile(r"\bSection\s+\d+(?:\([a-z]\))?\b", re.IGNORECASE)
RE_LEGAL_PAR = re.compile(r"§\s*\d+(?:\.\d+)*", re.IGNORECASE)
RE_12B2 = re.compile(r"\b12b-2\b", re.IGNORECASE)
RE_HYPHEN_ID = re.compile(r"\b\d{3}-\d{5}\b")

RE_HEADER_NOISE = re.compile(r"""
    ^\s*(table\s+of\s+contents|contents|index|page\s+\d+|exhibit\s+\d+|signature[s]?|
    united\s+states|securities\s+and\s+exchange\s+commission)\s*$
""", re.IGNORECASE | re.VERBOSE)
RE_SHORT_PUNC = re.compile(r"^[\W_]+$")

# ===== 新增：轻量 RAG 过滤规则 =====
RE_BOILERPLATE = re.compile(
    r'^\s*(FORM\s+10-[KQ]|TABLE\s+OF\s+CONTENTS|INDEX|SIGNATURES)\s*$',
    re.IGNORECASE
)
RE_SHORT_NUM = re.compile(r'^[0-9\-\.,]{1,6}$')  # 纯数字/短标点数字
RE_WASH_DC = re.compile(r'^\s*Washington,\s*D\.C\.(?:\s*\d{5}(?:-\d{4})?)?\s*$', re.IGNORECASE)

# =============================
# Utilities
# =============================
def _abs_from_project(p: Optional[str]) -> Optional[Path]:
    if not p:
        return None
    q = Path(p)
    return q if q.is_absolute() else (PROJECT_ROOT / q)

def read_config(cfg_path: Path) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            paths = raw.get("paths") or {}
            in_dir  = _abs_from_project(paths.get("processed_dir"))
            out_dir = _abs_from_project(paths.get("clean_dir"))
            if in_dir:  cfg["input_dir"]  = in_dir
            if out_dir: cfg["output_dir"] = out_dir
            tc = raw.get("text_clean") or {}
            if "pattern" in tc:            cfg["pattern"] = tc["pattern"]
            if "min_chars" in tc:          cfg["min_chars"] = int(tc["min_chars"])
            if "max_sentence_len" in tc:   cfg["max_sentence_len"] = int(tc["max_sentence_len"])
            if "hard_wrap_len" in tc:      cfg["hard_wrap_len"] = int(tc["hard_wrap_len"])
        except Exception as e:
            log.warning(f"Failed to read config {cfg_path}: {e}")
    return cfg

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def to_ascii(s: str) -> str:
    repl = {"—": "-", "–": "-", "•": "*", "§": "§", "☒": "[X]", "☐": "[ ]", "\xa0": " "}
    out = [repl.get(ch, ch) for ch in s]
    s2 = "".join(out)
    s2 = re.sub(r"[ \t]+", " ", s2)
    s2 = re.sub(r"\s*\n\s*", "\n", s2)
    return s2.strip()

def unescape_and_normalize(text: str) -> str:
    if text is None:
        return ""
    s = html.unescape(text)
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    return s.strip()

# —— 保留更多可能是标题/小节名的文本
def is_probable_heading(s: str) -> bool:
    st = s.strip()
    return (
        bool(re.match(r"^note\s+\d+\b", st, re.I)) or
        (st.isupper() and 3 <= len(st) <= 70) or
        bool(re.match(r"^(item\s+\d+[a-z]?)\b", st, re.I)) or
        bool(re.match(r"^[A-Z][A-Za-z0-9&,\-\s]{0,70}$", st))
    )

def is_noise_line(s: str, min_chars: int) -> bool:
    # 文本检索模式：尽量别丢文本，只过滤空串或纯标点
    if TEXT_RETRIEVAL_ONLY:
        if not s:
            return True
        if RE_SHORT_PUNC.match(s):
            return True
        return False
    if not s:
        return True
    if is_probable_heading(s):
        return False
    if len(s) < min_chars:
        if (RE_NUMBER.search(s) or RE_PERCENT.search(s) or RE_ISO_DUR.search(s) or ("☒" in s) or ("☐" in s)):
            pass
        else:
            if RE_DATE_ISO.search(s) or RE_DATE_LONG.search(s) or RE_LEGAL_RULE.search(s) or RE_COMMISSION_FILE.search(s):
                return False
            return True
    if RE_HEADER_NOISE.search(s):
        return True
    if RE_SHORT_PUNC.match(s):
        return True
    return False

def split_sentences(s: str) -> List[str]:
    s = s.replace("。", ".").replace("；", ";").replace("！", "!").replace("？", "?")
    s = re.sub(r"([\.!?])(\s|$)", r"\1\n", s)
    parts = [p.strip() for p in s.split("\n") if p.strip()]
    return parts

def chunk_long_text(s: str, max_len: int = DEFAULT_MAX_SENT_LEN, hard_wrap: int = DEFAULT_HARD_WRAP_LEN) -> List[str]:
    if len(s) <= max_len:
        return [s]
    parts: List[str] = []
    cur = s
    while len(cur) > max_len:
        cut = None
        for m in RE_SOFT_SPLIT.finditer(cur):
            if m.start() <= max_len:
                cut = m.start()
            else:
                break
        if cut is None or cut < max_len * 0.6:
            cut = min(len(cur), hard_wrap)
        parts.append(cur[:cut].strip())
        cur = cur[cut:].lstrip()
    if cur:
        parts.append(cur)
    return [p for p in parts if p]

def _spans_union(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    spans.sort()
    merged = [list(spans[0])]
    for a, b in spans[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [tuple(x) for x in merged]

def _inside_any(i: int, spans: List[Tuple[int, int]]) -> bool:
    for a, b in spans:
        if a <= i < b:
            return True
    return False

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
        if sym in {"$", "€", "£", "¥", "¥¥"} and sym in s:
            tokens.add(code)
    for m in RE_CURRENCY_WORD.finditer(s):
        tokens.add(m.group(0).upper().replace("RMB", "CNY"))
    return sorted(tokens)

def parse_iso_durations(s: str) -> Tuple[List[str], List[Optional[int]]]:
    raws, totals = [], []
    for m in RE_ISO_DUR.finditer(s):
        raws.append(m.group(0))
        y = int(m.group("years") or 0)
        mo = int(m.group("months") or 0)
        d = int(m.group("days") or 0)
        totals.append(y*365 + mo*30 + d)
    return raws, totals

def _keep_for_rag(text: str) -> bool:
    """RAG 轻量过滤：丢短数字 & 常见封面/目录行 & 简短 Washington, D.C."""
    t = (text or "").strip()
    if not t:
        return False
    if RE_SHORT_NUM.fullmatch(t):
        return False
    if RE_BOILERPLATE.fullmatch(t):
        return False
    if RE_WASH_DC.fullmatch(t) and len(t) <= 30:
        return False
    return True

# —— 改良：数字不过度跳过，输出兼容原字段 + 详细字段
def parse_number_tokens(s: str) -> Tuple[List[str], List[Optional[float]], List[Dict[str, Any]]]:
    legal_spans: List[Tuple[int, int]] = []
    for rx in (RE_LEGAL_RULE, RE_LEGAL_SECTION, RE_LEGAL_PAR, RE_12B2):
        for m in rx.finditer(s):
            legal_spans.append(m.span())
    legal_spans = _spans_union(legal_spans)

    raw_tokens, values, details = [], [], []
    for m in RE_NUMBER.finditer(s):
        start, end = m.span()
        if _inside_any(start, legal_spans):
            continue

        token = m.group(0).strip()
        num_str = m.group("num")
        unit = m.group("unit").lower() if m.group("unit") else None
        prefix = (m.group("prefix") or "").strip()

        window = s[max(0, start-30):min(len(s), end+30)]
        near_xbrl = bool(RE_MEMBER_NEARBY.search(window) or RE_XBRL_QNAME.search(window))
        near_date = bool(RE_DATE_ISO.search(window) or RE_DATE_LONG.search(window))

        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", token):
            continue

        try:
            val = float(num_str.replace(",", ""))
        except Exception:
            val = None

        currency = None
        if prefix in {"$", "€", "£", "¥", "¥¥"}:
            currency = CURRENCY_TOKENS[prefix]

        suppress_scale = False
        if unit in {"m"} and RE_MONTH_CONTEXT.search(window):
            suppress_scale = True

        scale_name = None
        if unit and not suppress_scale:
            scale = SCALE_MAP.get(unit, 1.0)
            scale_name = unit.lower()
        else:
            scale = 1.0

        if val is not None:
            val_scaled = val * scale
        else:
            val_scaled = None

        raw_tokens.append(token)
        values.append(val_scaled)
        details.append({
            "raw": token,
            "value": val_scaled,
            "base_value": val,
            "unit_token": unit,
            "scale_used": scale_name,
            "currency": currency,
            "near_date": near_date,
            "near_xbrl_qname": near_xbrl,
            "window": window
        })

    return raw_tokens, values, details

# =============================
# Core pipeline
# =============================
def process_jsonl_line(
    line: Dict[str, Any],
    *,
    min_chars: int,
    max_sent_len: int = DEFAULT_MAX_SENT_LEN,
    hard_wrap_len: int = DEFAULT_HARD_WRAP_LEN,
    heading_state: Dict[str, Any] = None,
    file_meta: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    idx = line.get("idx") or line.get("idx_source")
    tag = line.get("tag")
    base_text = (line.get("text_raw") or line.get("text") or "")
    text_norm = unescape_and_normalize(base_text)

    if heading_state is None:
        heading_state = {"current_heading": None}

    out_rows: List[Dict[str, Any]] = []

    # 标题更新
    if is_probable_heading(text_norm):
        heading_state["current_heading"] = text_norm

    # 文本检索模式：尽量保留；否则遵循原有噪声判断
    if (not TEXT_RETRIEVAL_ONLY) and is_noise_line(text_norm, min_chars=min_chars) and not is_probable_heading(text_norm):
         return []

    for sent in split_sentences(text_norm):
        if (not TEXT_RETRIEVAL_ONLY) and is_noise_line(sent, min_chars=min_chars) and not is_probable_heading(sent):
            continue

        for seg in chunk_long_text(sent, max_len=max_sent_len, hard_wrap=hard_wrap_len):
            if (not TEXT_RETRIEVAL_ONLY) and is_noise_line(seg, min_chars=min_chars) and not is_probable_heading(seg):
                continue

            # 文本检索模式：不做额外轻量过滤，保留更多句子；非文本模式才启用
            if (not TEXT_RETRIEVAL_ONLY) and (not _keep_for_rag(seg)):
                continue

            pct_raw, pct_vals = parse_percent_tokens(seg)
            num_raw, num_vals, num_details = parse_number_tokens(seg)
            currs = parse_currency_tokens(seg)
            dur_raw, dur_days = parse_iso_durations(seg)

            has_checked = "☒" in seg
            has_unchecked = "☐" in seg

            row = {
                "idx_source": idx,
                "tag": tag,
                "text_raw": base_text,
                "text": seg,
                "text_ascii": to_ascii(seg),
                "length": len(seg),

                "numbers_raw": num_raw,
                "numbers": num_vals,
                "numbers_detail": num_details,
                "percents_raw": pct_raw,
                "percents": pct_vals,
                "currencies": currs,

                "durations_raw": dur_raw,
                "durations_days": dur_days,

                "has_numbers": bool(num_raw),
                "has_percents": bool(pct_raw),
                "has_checked": has_checked,
                "has_unchecked": has_unchecked,

                "heading": heading_state.get("current_heading"),
            }

            # —— 透传/兜底关键元数据：优先用输入行，其次用 file_meta
            passthrough_keys = ("ticker", "year", "form", "accno", "fy", "fq", "doc_date", "source_path")

            # 1) 行内优先
            for k in passthrough_keys:
                if k in line and line.get(k) is not None:
                    row[k] = line.get(k)

            # 2) file_meta 只补空缺
            if file_meta:
                for k in passthrough_keys:
                    if row.get(k) is None and file_meta.get(k) is not None:
                        row[k] = file_meta.get(k)

            out_rows.append(row)


    return out_rows

# —— 从路径里尽力解析元数据：.../{ticker}/{year}/{form_accno}/text.jsonl
FORM_RE = re.compile(r"^(10-[KQ])_(.+)$", re.I)
def infer_meta_from_path(input_path: Path, base_dir: Optional[Path]) -> Dict[str, Any]:
    try:
        p = input_path if input_path.is_dir() else input_path.parent
        if base_dir and base_dir in p.parents:
            rel = p.relative_to(base_dir)
        else:
            rel = p
        parts = list(rel.parts)
        ticker = parts[-3] if len(parts) >= 3 else None
        year   = parts[-2] if len(parts) >= 2 else None
        form_accno = parts[-1] if len(parts) >= 1 else None
        form, accno = None, None
        if form_accno:
            m = FORM_RE.match(form_accno)
            if m:
                form = m.group(1).upper()
                accno = m.group(2)
        return {
            "ticker": ticker,
            "year": year,
            "form": form,
            "accno": accno, 
            "source_path": str(input_path)
        }
    except Exception:
        return {"source_path": str(input_path)}

# =============================
# I/O
# =============================
def clean_one_file(
    input_path: Path,
    out_jsonl: Path,
    out_parquet: Optional[Path],
    *,
    min_chars: int = 30,
    max_sent_len: int = DEFAULT_MAX_SENT_LEN,
    hard_wrap_len: int = DEFAULT_HARD_WRAP_LEN,
    heartbeat_every: int = DEFAULT_HEARTBEAT_EVERY,
    base_input_dir: Optional[Path] = None
) -> Dict[str, Any]:
    t0 = time.time()
    ensure_dir(out_jsonl)
    out_rows_for_parquet: List[Dict[str, Any]] = []

    heading_state = {"current_heading": None}
    file_meta = infer_meta_from_path(input_path, base_input_dir)

    total_lines = 0
    total_sentences = 0

    log.info(f"[start] {input_path}")
    with open(input_path, "r", encoding="utf-8") as f_in, open(out_jsonl, "w", encoding="utf-8") as f_out:
        seen_texts = set()   # —— 单文件内按文本去重
        for line in f_in:
            total_lines += 1
            if heartbeat_every and (total_lines % heartbeat_every == 0):
                log.info(f"  processed {total_lines} input lines…")
            try:
                obj = json.loads(line)
            except Exception:
                continue

            produced = process_jsonl_line(
                obj,
                min_chars=min_chars,
                max_sent_len=max_sent_len,
                hard_wrap_len=hard_wrap_len,
                heading_state=heading_state,
                file_meta=file_meta
            )
            total_sentences += len(produced)

            # 流式写 JSONL（去重后写）
            for r in produced:
                t = (r.get("text") or "").strip()
                if not t:
                    continue
                if t in seen_texts:
                    continue
                seen_texts.add(t)
                f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

            # 若希望 Parquet 也与 JSONL 一样去重，可以改成只 append 写出的 r
            out_rows_for_parquet.extend(produced)

    parquet_written = None
    csv_fallback = None
    if out_parquet is not None:
        try:
            ensure_dir(out_parquet)
            df = pd.DataFrame(out_rows_for_parquet)
            df.to_parquet(out_parquet, index=False)
            parquet_written = str(out_parquet)
            log.info(f"  wrote parquet: {out_parquet} (rows={len(df)})")
        except Exception as e:
            csv_fallback = str(out_parquet.with_suffix(".csv"))
            pd.DataFrame(out_rows_for_parquet).to_csv(csv_fallback, index=False)
            log.warning(f"  parquet failed ({e}); wrote csv fallback: {csv_fallback}")

    elapsed = time.time() - t0
    log.info(f"[done] {input_path.name} | lines={total_lines} -> sentences={total_sentences} | {elapsed:.2f}s")

    return {
        "input_file": str(input_path),
        "lines_read": total_lines,
        "sentences": total_sentences,
        "out_jsonl": str(out_jsonl),
        "out_parquet": parquet_written,
        "out_csv_fallback": csv_fallback,
        "elapsed_sec": round(elapsed, 3),
    }

# =============================
# CLI
# =============================
def main():
    parser = argparse.ArgumentParser(description="Clean text.jsonl into sentence-level corpus with numeric/percent/duration extraction.")
    in_grp = parser.add_mutually_exclusive_group(required=False)
    in_grp.add_argument("--input_file", help="Path to a single text.jsonl")
    in_grp.add_argument("--input_dir", help="Directory to scan recursively for files")

    parser.add_argument("--output_jsonl", help="Output JSONL (only when --input_file is used)")
    parser.add_argument("--output_parquet", help="Output Parquet (only when --input_file is used; falls back to CSV)")
    parser.add_argument("--output_dir", help="Base output dir when --input_dir is used")
    parser.add_argument("--pattern", default=None, help="Filename pattern (default from config or 'text.jsonl')")
    parser.add_argument("--min_chars", type=int, default=None, help="Minimum chars unless numbers/percents/durations/checkboxes present")
    parser.add_argument("--max_sentence_len", type=int, default=None, help="Max chars per sentence/chunk")
    parser.add_argument("--hard_wrap_len", type=int, default=None, help="Emergency wrap length")
    parser.add_argument("--heartbeat", type=int, default=DEFAULT_HEARTBEAT_EVERY, help="Log every N input lines")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")

    args = parser.parse_args()
    setup_logger(args.log_level)

    cfg = read_config(DEFAULT_CONFIG_PATH)
    log.info(f"Using config: {DEFAULT_CONFIG_PATH} (exists={DEFAULT_CONFIG_PATH.exists()})")

    input_dir  = Path(args.input_dir) if args.input_dir else cfg.get("input_dir", DEFAULT_INPUT_DIR)
    output_dir = Path(args.output_dir) if args.output_dir else cfg.get("output_dir", DEFAULT_OUTPUT_DIR)
    pattern    = args.pattern or cfg.get("pattern", "text.jsonl")
    min_chars  = args.min_chars if args.min_chars is not None else cfg.get("min_chars", 30)
    max_sent_len = args.max_sentence_len if args.max_sentence_len is not None else cfg.get("max_sentence_len", DEFAULT_MAX_SENT_LEN)
    hard_wrap_len = args.hard_wrap_len if args.hard_wrap_len is not None else cfg.get("hard_wrap_len", DEFAULT_HARD_WRAP_LEN)

    log.info(f"input_dir={input_dir}")
    log.info(f"output_dir={output_dir}")
    log.info(f"pattern='{pattern}' | min_chars={min_chars} | max_sentence_len={max_sent_len} | hard_wrap_len={hard_wrap_len} | heartbeat={args.heartbeat}")

    results: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}

    if args.input_file:
        in_file = Path(args.input_file)
        out_jsonl = Path(args.output_jsonl) if args.output_jsonl else in_file.with_name("text_corpus.jsonl")
        out_parquet = Path(args.output_parquet) if args.output_parquet else in_file.with_name("text_corpus.parquet")
        res = clean_one_file(
            in_file, out_jsonl, out_parquet,
            min_chars=min_chars,
            max_sent_len=max_sent_len,
            hard_wrap_len=hard_wrap_len,
            heartbeat_every=args.heartbeat,
            base_input_dir=None
        )
        results.append(res)
        summary = {"mode": "single_file"}
    else:
        files = list(input_dir.rglob(pattern))
        log.info(f"Found {len(files)} files under {input_dir} matching '{pattern}'")
        t_all = time.time()
        for i, f in enumerate(files, 1):
            rel = f.relative_to(input_dir)
            out_base_dir = output_dir / rel.parent
            out_jsonl = out_base_dir / "text_corpus.jsonl"
            out_parquet = out_base_dir / "text_corpus.parquet"
            log.info(f"[{i}/{len(files)}] -> {rel}")
            res = clean_one_file(
                f, out_jsonl, out_parquet,
                min_chars=min_chars,
                max_sent_len=max_sent_len,
                hard_wrap_len=hard_wrap_len,
                heartbeat_every=args.heartbeat,
                base_input_dir=input_dir
            )
            results.append(res)
        log.info(f"All files finished in {time.time()-t_all:.2f}s")
        summary = {"mode": "directory", "input_dir": str(input_dir), "output_dir": str(output_dir), "files": len(results)}

    agg = {"summary": summary, "results": results}
    print(json.dumps(agg, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
