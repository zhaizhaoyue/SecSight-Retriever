from pathlib import Path
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
import json, re
import math
import pandas as pd
import numpy as np

STD = Path("data/raw_reports/standard")

# -------------------------------
# 文本解析（保持不变）
# -------------------------------
def parse_text_from_html(path: Path):
    raw = path.read_text(encoding="utf-8", errors="ignore")
    text_head = raw.lstrip()[:200].lower()

    # XML/XBRL 用 XML 解析器；否则用 lxml 的 HTML 解析器
    if text_head.startswith("<?xml") or "<xbrl" in text_head or "<xbrli:" in text_head:
        soup = BeautifulSoup(raw, "xml")
    else:
        # 屏蔽“把 XML 当 HTML 解析”的提示
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(raw, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    blocks = soup.find_all(["h1","h2","h3","h4","p","li","div","section"])
    items = []
    for i, b in enumerate(blocks, 1):
        txt = " ".join(b.get_text(" ", strip=True).split())
        if txt:
            items.append({"idx": i, "tag": b.name, "text": txt})
    return items

# -------------------------------
# 方案 A：增强的表格解析与清洗
# - 逐个 <table> 解析
# - 打分筛主财务表
# - 单位识别（in millions/thousands/billions）
# - 数字标准化（负号/千分位/括号负）
# - 年份列识别并长表化 (tidy)
# - 保留 caption / 附近标题上下文
# -------------------------------

NEARBY_WINDOW = 1200
TITLE_RE = re.compile(
    r"(consolidated\s+statements?\s+of\s+(operations|income|comprehensive\s+income)|"
    r"balance\s+sheets?|"
    r"cash\s+flows?)",
    re.I
)

def _read_html_raw(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def _extract_caption_and_nearby(raw_html: str, table_tag) -> tuple[str, str]:
    # caption
    cap_tag = table_tag.find("caption")
    cap = cap_tag.get_text(" ", strip=True) if cap_tag else ""

    # 附近上下文（左右窗口）
    # 用字符串定位 table 近似位置（可能不完美，但足够用于关键词/单位检测）
    snippet = str(table_tag)[:100]
    pos = raw_html.find(snippet)
    ctx = raw_html[max(0, pos-NEARBY_WINDOW):pos+NEARBY_WINDOW].lower() if pos != -1 else ""
    return cap, ctx

def _detect_scale(text: str) -> int:
    t = text.lower()
    if "in billions" in t or "(billions)" in t:
        return 1_000_000_000
    if "in millions" in t or "(millions)" in t:
        return 1_000_000
    if "in thousands" in t or "(thousands)" in t:
        return 1_000
    return 1

def _parse_number(x):
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    # 常见空值/破折号
    if s.lower() in {"na", "n/a", "nil", "none"}:
        return None
    if s in {"—","–","-","—"}:
        return None
    # 去掉千分位、统一负号
    s = s.replace(",", "")
    s = re.sub(r"[\u2212–—−]", "-", s)  # Unicode minus/dash → '-'
    # 括号负数
    if re.fullmatch(r"\(.*\)", s):
        s = "-" + s[1:-1]
    # 去掉百分号（如需保留百分号，改为返回额外字段）
    s = s.rstrip("%")
    try:
        return float(s)
    except:
        return None

def _normalize_header(df: pd.DataFrame) -> pd.DataFrame:
    # pandas.read_html 可能生成 MultiIndex 列
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join([str(x) for x in tup if str(x).strip() != "" and str(x) != "nan"]).strip()
            for tup in df.columns
        ]
    # 去除纯空/重复列名
    df.columns = [str(c).strip() or f"col_{i}" for i, c in enumerate(df.columns)]
    # 清理完全空的行
    df = df.dropna(how="all")
    return df

def _score_table(caption: str, nearby_ctx: str, df: pd.DataFrame) -> int:
    score = 0
    if TITLE_RE.search(caption) or TITLE_RE.search(nearby_ctx):
        score += 2
    # 年份列
    if any(re.search(r"(19|20)\d{2}", str(c)) for c in df.columns):
        score += 1
    # 常见财务关键字
    header_str = " ".join(map(str, df.columns)).lower()
    if any(k in header_str for k in ["net income", "total assets", "cash and cash equivalents",
                                     "operating activities", "liabilities"]):
        score += 1
    return score

def _longify(df: pd.DataFrame, scale: int, meta: dict) -> pd.DataFrame:
    if df.shape[1] < 2 or df.shape[0] < 2:
        return pd.DataFrame()

    # 假定第一列为科目（line item）
    item_col = df.columns[0]

    # 识别年份列（标题含 20xx/19xx）
    year_cols = []
    for c in df.columns[1:]:
        m = re.search(r"(19|20)\d{2}", str(c))
        if m:
            year_cols.append((c, m.group(0)))

    if not year_cols:
        # 有些表用 "Year Ended December 31, 2024" 这样的列名
        for c in df.columns[1:]:
            m = re.search(r"(year\s+ended|three|twelve|months).*?(19|20)\d{2}", str(c), re.I)
            if m:
                y = re.findall(r"(19|20)\d{2}", str(c))
                if y:
                    year_cols.append((c, y[-1]))

    if not year_cols:
        return pd.DataFrame()

    # melt 为长表
    melted = df.melt(id_vars=[item_col], value_vars=[c for c,_ in year_cols],
                     var_name="period_col", value_name="raw_value")
    y_map = {c:y for c,y in year_cols}
    melted["period_year"] = melted["period_col"].map(y_map)
    melted["value"] = melted["raw_value"].map(_parse_number).map(lambda v: v*scale if v is not None else None)

    tidy = melted.drop(columns=["period_col","raw_value"]).rename(columns={item_col: "line_item"})
    # 基本清理：去掉全空/标题行
    tidy["line_item"] = tidy["line_item"].astype(str).str.strip()
    tidy = tidy[tidy["line_item"].str.len() > 0]
    # 附加元信息
    for k,v in meta.items():
        tidy[k] = v
    # 丢掉完全空值
    tidy = tidy.dropna(subset=["value"], how="all")
    return tidy

def parse_tables_from_html_advanced(path: Path, report_id: str):
    """
    返回两个 DataFrame:
      - df_raw  : 原表（横表），合并所有通过筛选的表，追加 __table_id__/caption/score
      - df_tidy : 清洗后的长表 line_item, period_year, value, caption, report_id, __table_id__
    """
    raw_html = _read_html_raw(path)
    soup = BeautifulSoup(raw_html, "lxml")
    table_tags = soup.find_all("table")

    raw_parts = []
    tidy_parts = []

    for i, tbl in enumerate(table_tags, 1):
        # 单表解析为 DataFrame
        try:
            dfs = pd.read_html(str(tbl))
        except Exception:
            continue
        if not dfs:
            continue
        df = dfs[0]
        if df.shape[0] < 3 or df.shape[1] < 2:
            continue

        df = _normalize_header(df)
        caption, ctx = _extract_caption_and_nearby(raw_html, tbl)
        scale = _detect_scale(caption + " " + ctx)
        score = _score_table(caption, ctx, df)

        # 只保留“疑似三大报表”的表（你也可以把 score==0 的先收集到 raw，但不入 tidy）
        if score == 0:
            continue

        df_raw = df.copy()
        df_raw["__table_id__"] = i
        df_raw["caption"] = caption
        df_raw["score"] = score
        raw_parts.append(df_raw)

        meta = {"__table_id__": i, "report_id": report_id, "caption": caption, "score": score, "scale": scale}
        df_tidy = _longify(df, scale, meta)
        if not df_tidy.empty:
            tidy_parts.append(df_tidy)

    df_raw_all  = pd.concat(raw_parts,  ignore_index=True) if raw_parts  else pd.DataFrame()
    df_tidy_all = pd.concat(tidy_parts, ignore_index=True) if tidy_parts else pd.DataFrame()
    return df_raw_all, df_tidy_all

# -------------------------------
# 兼容：保留原始的 parse_tables_from_html（供旧逻辑/回退用）
# -------------------------------
def parse_tables_from_html(path: Path):
    try:
        tables = pd.read_html(path, flavor="lxml")
    except Exception:
        tables = []
    dfs = []
    for i, t in enumerate(tables, 1):
        t.columns = [str(c) for c in t.columns]
        t["__table_id__"] = i
        dfs.append(t)
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        df_all = df_all.map(lambda v: "" if (pd.isna(v) or (isinstance(v, float) and math.isnan(v))) else str(v))
    else:
        df_all = pd.DataFrame()
    return df_all

# -------------------------------
# 单文件解析：增加 tidy 输出
# -------------------------------
def parse_one(std_file: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 解析文本
    if std_file.suffix.lower() in {".htm",".html"}:
        text_items = parse_text_from_html(std_file)
    else:
        raw = std_file.read_text(encoding="utf-8", errors="ignore")
        lines = [l.strip() for l in raw.splitlines()]
        text_items = [{"idx": i+1, "tag": "p", "text": l} for i, l in enumerate(lines) if l]

    (out_dir / "text.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in text_items),
        encoding="utf-8"
    )

    # --- 表格（增强版 + 兼容原版） ---
    df_raw = pd.DataFrame()
    df_tidy = pd.DataFrame()
    if std_file.suffix.lower() in {".htm",".html"}:
        # report_id = 目录名（如 10-K_0000xxxx），便于追踪
        report_id = out_dir.name
        try:
            df_raw, df_tidy = parse_tables_from_html_advanced(std_file, report_id=report_id)
        except Exception:
            # 兜底：使用原先的“简单合并版”，保证至少有历史兼容输出
            df_raw = parse_tables_from_html(std_file)
            df_tidy = pd.DataFrame()

    # 写出：原始横表（兼容历史）
    if not df_raw.empty:
        df_raw.to_parquet(out_dir / "tables.parquet", index=False)
        df_raw.head(2000).to_csv(out_dir / "tables_sample.csv", index=False, encoding="utf-8-sig")

    # 写出：长表（新增）
    if not df_tidy.empty:
        df_tidy.to_parquet(out_dir / "tables_tidy.parquet", index=False)
        df_tidy.head(2000).to_csv(out_dir / "tables_tidy_sample.csv", index=False, encoding="utf-8-sig")

# -------------------------------
# 批量
# -------------------------------
def batch_parse():
    for f in STD.glob("*.*"):
        # US_{ticker}_{year}_{form}_{accno}.{ext}
        m = re.match(r"US_(.+?)_(\d{4})_(10-[KQ])_(.+)\.(.+)$", f.name, re.I)
        if not m:
            continue
        ticker, year, form, accno, ext = m.groups()
        out_dir = Path(f"data/processed/{ticker}/{year}/{form}_{accno}")
        print(f"[parse] {f.name} -> {out_dir}")
        parse_one(f, out_dir)

if __name__ == "__main__":
    batch_parse()
    print("✅ 全部解析完成：文本(text.jsonl)、表格(tables.parquet/tables_sample.csv)、长表(tables_tidy.parquet/tables_tidy_sample.csv)")
