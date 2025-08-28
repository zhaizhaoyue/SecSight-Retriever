from pathlib import Path
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
import json
import re
import sys

STD = Path("data/raw_reports/standard")

# ======== 配置 ========
# 识别“事实串”的高密阈值（满足其一就归入 facts_like.jsonl）
FACT_TOKEN_PATTERNS = [r"\bus-gaap:", r"\bxbrli:", r"\biso4217:"]
FACT_TOKEN_MIN_COUNT = 3  # 同一文本中命中以上 token 的总数 >= 3

# HTML 中保留的标签；div/section 只有在不含下述细粒度标签时才保留，避免父子重复
FINE_TAGS = {"p", "li", "h1", "h2", "h3", "h4"}
BLOCK_TAGS = ["h1", "h2", "h3", "h4", "p", "li", "div", "section"]

# 常见噪声（目录、签名页、页眉页脚等）
RE_NOISE = re.compile(
    r"^\s*(table\s+of\s+contents|contents|index|page\s+\d+|exhibit\s+\d+|signature[s]?|"
    r"united\s+states|securities\s+and\s+exchange\s+commission)\s*$",
    re.IGNORECASE
)
RE_ONLY_PUNC = re.compile(r"^[\W_]+$")  # 纯标点/非字母数字
RE_MULTI_SPACE = re.compile(r"[ \t]+")

# 标准文件名：US_{ticker}_{year}_{form}_{accno}.{ext}
NAME_RE = re.compile(r"^US_(.+?)_(\d{4})_(10-[KQ])_(.+)\.(.+)$", re.I)
ACCNO_RE = re.compile(r"(\d{10}-\d{2}-\d{6})")
FNAME_RE = re.compile(
    r"""
    ^
    (?:US_)?                                   # optional leading 'US_'
    (?P<ticker>[A-Z]{1,10})_                   # ticker
    (?P<year>\d{4})_                           # year in filename
    (?P<form>10\-K|10\-Q|20\-F|40\-F|8\-K)     # form
    _(?P<accno>\d{10}\-\d{2}\-\d{6})           # accession
    (?:_(?P<ticker2>[a-z0-9\-]+))?             # sometimes ticker repeats lower-case
    _(?P<docdate>\d{8})                        # e.g. 20230930
    (?:[._]\w+)?                               # ← 允许 ".xxx" 或 "_xxx"（修复点）
    \.(?P<ext>xml|html)$                       # 简化：最终扩展名
    """,
    re.X | re.I,
)

def normalize_accno(s: str) -> str:
    m = ACCNO_RE.search(s)
    return m.group(1) if m else s


# ======== 小工具 ========
def normalize_spaces(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = RE_MULTI_SPACE.sub(" ", s)
    return s.strip()

def is_noise_line(s: str) -> bool:
    if not s:
        return True
    if RE_NOISE.search(s):
        return True
    if RE_ONLY_PUNC.match(s):
        return True
    return False

def count_fact_tokens(text: str) -> int:
    cnt = 0
    for pat in FACT_TOKEN_PATTERNS:
        cnt += len(re.findall(pat, text))
    return cnt

def parse_meta_from_filename(p: Path) -> dict:
    m = NAME_RE.match(p.name)
    if not m:
        return {}
    ticker, year, form, accno, ext = m.groups()
    return {
        "ticker": ticker,
        "year": year,
        "form": form.upper(),
        "accno": accno,
        "ext": ext.lower(),
        "source_path": str(p)
    }


# ======== HTML → 文本抽取（避免父子重复 + 去噪 + XBRL 高密分流） ========
def parse_text_from_html(path: Path):
    """
    返回 (text_items, facts_like_items)
    text_items: 纯自然语言/标题等适合 RAG 的文本
    facts_like_items: 含大量 XBRL token 的“事实样”行，供单独存放
    """
    raw = path.read_text(encoding="utf-8", errors="ignore")
    text_head = raw.lstrip()[:200].lower()

    # XML/XBRL 明显特征：直接用 xml 解析器（但这里我们仍按 HTML 抽纯文本，不抽标签）
    if text_head.startswith("<?xml") or "<xbrl" in text_head or "<xbrli:" in text_head:
        soup = BeautifulSoup(raw, "xml")
    else:
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(raw, "lxml")

    # 清除不需要的节点
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # 先收集候选 block，避免父子重复：
    blocks = []
    for b in soup.find_all(BLOCK_TAGS):
        # 如果是 div/section 且内含细粒度段落标签，则跳过（避免把容器与子段落重复收集）
        if b.name in {"div", "section"} and b.find(list(FINE_TAGS)):
            continue
        blocks.append(b)

    text_items = []
    facts_like_items = []
    seen = set()  # 去重（同一文件内完全相同的文本跳过）

    for i, b in enumerate(blocks, 1):
        txt = b.get_text(" ", strip=True)
        txt = normalize_spaces(txt)
        if not txt:
            continue
        if is_noise_line(txt):
            continue
        if txt in seen:
            continue
        seen.add(txt)

        # 高密 XBRL token 判定：进 facts_like 分流
        fact_hits = count_fact_tokens(txt)
        rec = {"idx": i, "tag": b.name, "text": txt}
        if fact_hits >= FACT_TOKEN_MIN_COUNT:
            rec["fact_token_hits"] = fact_hits
            facts_like_items.append(rec)
        else:
            text_items.append(rec)

    return text_items, facts_like_items


# ======== 单文件处理 ========
def parse_one(std_file: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = parse_meta_from_filename(std_file)
    if not meta:
        print(f"[skip-name] 非标准命名，跳过: {std_file.name}")
        return

    suffix = std_file.suffix.lower()

    # 1) 严格分流：XML/XBRL 完全不进文本通道
    if suffix in {".xml", ".xbrl"}:
        print(f"[skip-xml] {std_file.name}")
        return

    # 2) HTML：走 HTML 抽取 + XBRL 高密分流
    if suffix in {".htm", ".html"}:
        try:
            text_items, facts_like_items = parse_text_from_html(std_file)
        except Exception as e:
            print(f"[error] 解析 HTML 失败: {std_file.name} -> {e}")
            return

        # 写 text.jsonl（仅当有数据）
        if text_items:
            out_text = out_dir / "text.jsonl"
            with out_text.open("w", encoding="utf-8") as f:
                for x in text_items:
                    x.update(meta)
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
            print(f"[ok] text  -> {out_text} ({len(text_items)} lines)")
        else:
            print(f"[warn] 无可用文本: {std_file.name}")

        # # 写 facts_like.jsonl（含高密 XBRL token 的行）
        # if facts_like_items:
        #     out_facts = out_dir / "facts_like.jsonl"
        #     with out_facts.open("w", encoding="utf-8") as f:
        #         for x in facts_like_items:
        #             x.update(meta)
        #             f.write(json.dumps(x, ensure_ascii=False) + "\n")
        #     print(f"[ok] facts -> {out_facts} ({len(facts_like_items)} lines)")

        return

    # 3) 其他（如 .txt）：逐行文本（谨慎保留；如不需要可改为直接跳过）
    try:
        raw = std_file.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[error] 读取失败: {std_file.name} -> {e}")
        return

    lines = [normalize_spaces(l) for l in raw.splitlines()]
    items = []
    seen = set()
    j = 0
    for l in lines:
        if not l:
            continue
        if is_noise_line(l):
            continue
        if l in seen:
            continue
        seen.add(l)
        j += 1
        items.append({"idx": j, "tag": "p", "text": l})

    if items:
        out_text = out_dir / "text.jsonl"
        with out_text.open("w", encoding="utf-8") as f:
            for x in items:
                x.update(meta)
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"[ok] text  -> {out_text} ({len(items)} lines)")
    else:
        print(f"[warn] {std_file.name} 无可用文本")


# ======== 批量 ========
def batch_parse():
    SKIP_EXTS = {".xml", ".xsd", ".xbrl"}     # 完全跳过这些（Arelle 另行处理）
    TEXT_OK_EXTS = {".htm", ".html", ".txt"}  # 仅这些进入文本清洗

    if not STD.exists():
        print(f"[error] 未找到目录: {STD.resolve()}")
        return

    files = sorted(STD.glob("*.*"))  # 不递归，保持旧逻辑；加个排序稳定输出
    print(f"[info] 标准文件数: {len(files)} in {STD}")

    seen = set()  # 去重：同一个 (ticker, year, form, accno) 只处理一次

    for f in files:
        m = NAME_RE.match(f.name)
        if not m:
            # 非标准命名跳过，避免写到意外目录
            continue

        ticker, year, form, accno_raw, ext = m.groups()
        form = form.upper()
        ext_dot = "." + ext.lower()

        # 1) 跳过 XML/XSD/XBRL（这些走 Arelle，不建 processed 目录）
        if ext_dot in SKIP_EXTS:
            print(f"[skip-xml] {f.name}")
            continue

        # 2) 仅处理允许的文本扩展名
        if ext_dot not in TEXT_OK_EXTS:
            print(f"[skip-ext] {f.name}")
            continue

        # 3) 规范化 accession，去掉文件名里附带的 _aapl-20230930_cal 等后缀
        accno = normalize_accno(accno_raw)

        # 4) 去重：同一个 filing 只处理一次
        key = (ticker, year, form, accno)
        if key in seen:
            print(f"[skip-dup] {f.name} (same filing: {key})")
            continue
        seen.add(key)

        out_dir = Path(f"data/processed/{ticker}/{year}/{form}_{accno}")
        print(f"[parse] {f.name} -> {out_dir}")
        try:
            parse_one(f, out_dir)
        except Exception as e:
            # 单文件异常不影响后续
            print(f"[error] 处理失败: {f.name} -> {e}")

if __name__ == "__main__":
    batch_parse()
    print("✅ 完成：输出 RAG 文本 (text.jsonl)，并将高密 XBRL 行分流至 facts_like.jsonl（若有）")
