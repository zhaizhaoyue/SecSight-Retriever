from pathlib import Path
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
import json, re
import math
import pandas as pd
import numpy as np

STD = Path("data/raw_reports/standard")

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

def parse_one(std_file: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
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

    df = parse_tables_from_html(std_file) if std_file.suffix.lower() in {".htm",".html"} else pd.DataFrame()
    if not df.empty:
        df.to_parquet(out_dir / "tables.parquet", index=False)
        df.head(2000).to_csv(out_dir / "tables_sample.csv", index=False, encoding="utf-8-sig")

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
    print("✅ 全部解析完成：文本(text.jsonl)、表格(tables.parquet/tables_sample.csv)")
