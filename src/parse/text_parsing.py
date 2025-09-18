#222222
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
import json
import re
import sys
import uuid
from datetime import datetime, timezone

# -----------------------------
# 常量 & 配置
# -----------------------------
STD = Path("data/raw_reports/standard")

SCHEMA_VERSION = "0.3.0"
WRITE_FACTS_LIKE = False        # 是否输出 facts_like.jsonl
TEXT_RETRIEVAL_ONLY = True
CHUNK_SIZE_TOK = 900           # 文本切块大小（token 近似）
CHUNK_OVERLAP_TOK = 120         # 块间重叠
MAX_BLOCK_CHARS = 10_000_000        # 单 block 过长时先做硬切，防止超长段落
LANG_DEFAULT = "en"
PAGE_ID_RE   = re.compile(r'\b(?:p(?:age)?[-_]?)(\d{1,4})\b', re.I)   # p12/page-12/page12
PAGE_TEXT_RE = re.compile(r'^\s*page\s+(\d{1,4})\s*$', re.I)  
# 尝试使用 tiktoken（若不可用，回退到空白分词计数）
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(s: str) -> int:
        return len(_enc.encode(s))
except Exception:
    _enc = None
    def count_tokens(s: str) -> int:
        return max(1, len(s.split()))

# XBRL token 高密判定
FACT_TOKEN_PATTERNS = [r"\bus-gaap:", r"\bxbrli:", r"\biso4217:"]
FACT_TOKEN_MIN_COUNT = 3

# HTML 中保留的标签；div/section 只有在不含下述细粒度标签时才保留，避免父子重复
FINE_TAGS = {"p", "li", "h1", "h2", "h3", "h4"}
BLOCK_TAGS = ["h1","h2","h3","h4","p","li","div","section","td","th"]


# 噪声过滤
RE_NOISE = re.compile(
    r"^\s*(table\s+of\s+contents|contents|index|page\s+\d+|exhibit\s+\d+|signature[s]?|"
    r"united\s+states|securities\s+and\s+exchange\s+commission)\s*$",
    re.IGNORECASE
)
RE_ONLY_PUNC = re.compile(r"^[\W_]+$")
RE_MULTI_SPACE = re.compile(r"[ \t]+")

# 文件名解析
NAME_RE = re.compile(r"^US_(.+?)_(\d{4})_(10-[KQ])_(.+)\.(.+)$", re.I)
ACCNO_RE = re.compile(r"(\d{10}-\d{2}-\d{6})")
FNAME_RE = re.compile(
    r"""
    ^
    (?:US_)?                                   # optional 'US_'
    (?P<ticker>[A-Z]{1,10})_
    (?P<year>\d{4})_
    (?P<form>10\-K|10\-Q|20\-F|40\-F|8\-K)_
    (?P<accno>\d{10}\-\d{2}\-\d{6})
    (?:_(?P<ticker2>[a-z0-9\-]+))?             # e.g., aapl-20230930
    _(?P<docdate>\d{8})
    (?:[._]\w+)?                               # allow ".xxx" or "_xxx"
    \.(?P<ext>xml|html|htm|txt)$
    """,
    re.X | re.I,
)

# Part/Item 解析（简化模式）
RE_PART = re.compile(r"^\s*part\s+([ivx]+)\s*[\.\-:]?\s*(.*)$", re.I)                    # Part I/II/III/IV
RE_ITEM = re.compile(r"^\s*item\s+(\d+(\.\d+)*)\s*[\.\-:]?\s*(.*)$", re.I)               # Item 1, Item 1A, 7, 7A, 8...
RE_WHITES = re.compile(r"\s+")


# -----------------------------
# 小工具
# -----------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def normalize_accno(s: str) -> str:
    m = ACCNO_RE.search(s)
    return m.group(1) if m else s

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

def is_heading_candidate(tag, txt: str) -> bool:
    # 1) 明确的 heading 标签
    if tag.name in {"h1", "h2", "h3", "h4"}:
        return True
    # 2) 含粗体/strong 的单句块，且长度适中
    if tag.name in {"div", "p"}:
        if tag.find(["b", "strong"]) and 5 <= len(txt) <= 180:
            return True
    # 3) 文本本身像 "Item 7A ..." / "Part II ..."
    if RE_ITEM.match(txt) or RE_PART.match(txt):
        return True
    return False

def css_path_for_tag(tag) -> str:
    parts = []
    cur = tag
    while cur is not None and getattr(cur, "name", None):
        parent = cur.parent
        if parent and getattr(parent, "find_all", None):
            sibs = [s for s in parent.find_all(cur.name, recursive=False)]
            idx = sibs.index(cur) + 1 if sibs else 1
        else:
            idx = 1
        # 只有在有合法标签名时才拼接
        name = cur.name if isinstance(cur.name, str) else "node"
        parts.append(f"{name}:nth-of-type({idx})")
        cur = parent
    return " > ".join(reversed(parts))

def xpath_for_tag(tag) -> str:
    parts = []
    cur = tag
    while cur is not None and getattr(cur, "name", None):
        # 跳过 BeautifulSoup 的根节点 [document]
        if cur.name == "[document]":
            cur = cur.parent
            continue

        parent = cur.parent
        if parent and getattr(parent, "find_all", None):
            sibs = [s for s in parent.find_all(cur.name, recursive=False)]
            idx = sibs.index(cur) + 1 if sibs else 1
        else:
            idx = 1

        parts.append(f"/{cur.name}[{idx}]")
        cur = parent

    path = "".join(reversed(parts)) or "/"
    # 确保 /html[1] 开头
    if not path.startswith("/html["):
        path = "/html[1]" + path
    return path

def nearest_anchor(tag) -> str | None:
    cur = tag
    while cur and getattr(cur, "name", None):
        for attr in ("id", "name"):
            v = cur.get(attr)
            if v and isinstance(v, str) and v.strip():
                return v.strip()
        cur = cur.parent
    return None

def detect_page_no(tag) -> int | None:
    # 向上找容器属性
    cur = tag
    while cur and getattr(cur, "name", None):
        for attr in ("id", "name", "data-page", "aria-label"):
            v = cur.get(attr)
            if v:
                m = PAGE_ID_RE.search(str(v))
                if m:
                    return int(m.group(1))
        cls = cur.get("class") or []
        if not isinstance(cls, (list, tuple)):
            cls = [cls]
        for c in cls:
            if not c:
                continue
            m = PAGE_ID_RE.search(str(c))
            if m:
                return int(m.group(1))
        cur = cur.parent

    # 向前找“Page N”文本
    prev = tag; steps = 0
    while prev is not None and steps < 8:
        prev = prev.previous_sibling; steps += 1
        if hasattr(prev, "get_text"):
            t = normalize_spaces(prev.get_text(" ", strip=True))
            m = PAGE_TEXT_RE.match(t)
            if m:
                return int(m.group(1))
    return None

def load_dei_from_facts(facts_file: Path) -> dict[str, str]:
    """从 facts.jsonl 里读 fy/fq/doc_date（取第一条或 meta 行）"""
    if not facts_file.exists():
        return {}
    with facts_file.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            out = {}
            if row.get("period_fy"):
                out["fy"] = row["period_fy"]
            if row.get("period_fq"):
                out["fq"] = row["period_fq"]
            if row.get("doc_date"):
                out["doc_date"] = row["doc_date"]
            return out
    return {}

def parse_meta_from_filename(p: Path) -> dict:
    m = FNAME_RE.match(p.name)
    if not m:
        # 回退到旧的 NAME_RE（不包含 docdate/txt）
        m2 = NAME_RE.match(p.name)
        if not m2:
            return {}
        ticker, year, form, accno, ext = m2.groups()
        docdate = None
    else:
        g = m.groupdict()
        ticker, year, form, accno, docdate, ext = (
            g["ticker"], g["year"], g["form"], g["accno"], g["docdate"], g["ext"]
        )

    return {
        "ticker": ticker.upper(),
        "year": int(year),
        "form": form.upper(),
        "accno": normalize_accno(accno),
        "doc_date": docdate,
        "ext": ext.lower(),
        "source_path": p.as_posix(),   # ✅ 统一为 POSIX 风格路径
    }


def chunk_by_tokens(text: str, max_tokens: int, overlap_tokens: int) -> list[tuple[int, int, str]]:
    """返回 [(start_tok, end_tok, chunk_text), ...]。不依赖外网库，tiktoken 可选。"""
    if _enc:
        tokens = _enc.encode(text)
        n = len(tokens)
        chunks = []
        i = 0
        while i < n:
            j = min(n, i + max_tokens)
            piece = _enc.decode(tokens[i:j])
            chunks.append((i, j, piece))
            if j == n:
                break
            i = max(0, j - overlap_tokens)
        return chunks
    else:
        # 退化为基于空白的 token
        words = text.split()
        n = len(words)
        chunks = []
        i = 0
        while i < n:
            j = min(n, i + max_tokens)
            piece = " ".join(words[i:j])
            chunks.append((i, j, piece))
            if j == n:
                break
            i = max(0, j - overlap_tokens)
        return chunks


# -----------------------------
# HTML → 文本抽取
# -----------------------------
def parse_text_from_html(path: Path):
    """
    返回 (text_items, facts_like_items)
    text_items: 适合 RAG 的文本块（不含高密 XBRL token）
    facts_like_items: 含大量 XBRL token 的“事实样”行，供单独存放
    """
    raw = path.read_text(encoding="utf-8", errors="ignore")
    text_head = raw.lstrip()[:200].lower()
    # 解析器选择
    if text_head.startswith("<?xml") or "<xbrl" in text_head or "<xbrli:" in text_head:
        soup = BeautifulSoup(raw, "xml")
    else:
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(raw, "lxml")

    # 移除脚本与样式
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # 收集 block，避免父子重复
    blocks = []
    for b in soup.find_all(BLOCK_TAGS):
        if b.name in {"div", "section"} and b.find(list(FINE_TAGS)):
            continue
        blocks.append(b)

    text_items = []
    facts_like_items = []
    seen_texts = set()

    # Part/Item 追踪（弱规则）
    cur_part = None      # "Part I", "Part II", ...
    cur_item = None      # "Item 7", "Item 8", ...
    cur_heading = None   # 具体标题文本
    cur_section_path = None

    def update_heading_context(s: str, tag_name: str):
        nonlocal cur_part, cur_item, cur_heading, cur_section_path
        # 仅对标题级标签尝试识别
        candidate = s.strip()
        got = False

        m_part = RE_PART.match(candidate)
        if m_part:
            roman = m_part.group(1).upper()
            rest = m_part.group(2).strip()
            cur_part = f"Part {roman}"
            cur_heading = candidate
            got = True

        m_item = RE_ITEM.match(candidate)
        if m_item:
            num = m_item.group(1)
            title = m_item.group(3).strip()
            heading = f"Item {num}"
            if title:
                heading = f"{heading}. {title}"
            cur_item = heading
            cur_heading = heading
            got = True

        # 若是 h1/h2/h3/h4，但没命中 Part/Item，也将其当作 heading
        if tag_name in {"h1", "h2", "h3", "h4"} and not got:
            cur_heading = candidate

        # 生成 section_path
        parts = []
        if cur_part:
            parts.append(cur_part)
        if cur_item:
            parts.append(cur_item)
        if cur_heading and (not parts or cur_heading not in parts[-1]):
            parts.append(cur_heading)
        cur_section_path = " / ".join(parts) if parts else None

    created_at = now_iso()

    for i, b in enumerate(blocks, 1):
        txt = b.get_text(" ", strip=True)
        txt = normalize_spaces(txt)
        if b.name in {"div", "span"} and b.parent and b.parent.name in {"td", "th"}:
            cell = b.parent
            txt_full = normalize_spaces(cell.get_text(" ", strip=True))
            if txt_full and len(txt_full) > len(txt):  # 只在更完整时替换
                txt = txt_full
                b = cell
        RE_FRAGMENT = re.compile(r"^\W*(of|and|or)\b.+\)$", re.I)
        if len(txt) < 25 and RE_FRAGMENT.match(txt):
            continue

        if not txt or ((not TEXT_RETRIEVAL_ONLY) and is_noise_line(txt)) or txt in seen_texts:
            continue
        seen_texts.add(txt)

        # 遇到标题，更新上下文    
        if is_heading_candidate(b, txt):
            update_heading_context(txt, b.name)

        if b.name in {"h1", "h2", "h3", "h4"}:
            update_heading_context(txt, b.name)

        # 记录 tag 定位
        css_path = css_path_for_tag(b)
        x_path   = xpath_for_tag(b)
        page_no  = detect_page_no(b)
        anchor   = nearest_anchor(b)
        fact_hits = count_fact_tokens(txt)
        # 去除极短无意义文本
        if len(txt) < 3:
            continue


        css_path = css_path_for_tag(b)

        rec_base = {
            "idx_source": i,
            "tag": b.name,
            "text_raw": txt,
            "css_path": css_path,
            "part": cur_part,
            "item": cur_item,
            "heading": cur_heading,
            "section_path": cur_section_path,
            "xpath": x_path,
            "page_no": page_no,
            "page_anchor": anchor,
        }

        if (TEXT_RETRIEVAL_ONLY) or (fact_hits < FACT_TOKEN_MIN_COUNT):
            # 控制超长 block
            text_for_chunk = txt if len(txt) <= MAX_BLOCK_CHARS else txt[:MAX_BLOCK_CHARS]
            toks = count_tokens(text_for_chunk)

            # 切块
            if toks > CHUNK_SIZE_TOK:
                chunks = chunk_by_tokens(text_for_chunk, CHUNK_SIZE_TOK, CHUNK_OVERLAP_TOK)
                for si, sj, piece in chunks:
                    rec = dict(rec_base)
                    rec["text"] = piece
                    rec["tokens"] = count_tokens(piece)
                    rec["chunk_tok_start"] = si
                    rec["chunk_tok_end"] = sj
                    text_items.append(rec)
            else:
                rec = dict(rec_base)
                rec["text"] = text_for_chunk
                rec["tokens"] = toks
                text_items.append(rec)
        else:
            # 非文本模式下才分流到 facts_like
            rec = dict(rec_base)
            rec["fact_token_hits"] = fact_hits
            facts_like_items.append(rec)

    return text_items, facts_like_items


# -----------------------------
# 单文件处理
# -----------------------------
def parse_one(std_file: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = parse_meta_from_filename(std_file)
    if not meta:
        print(f"[skip-name] 非标准命名，跳过: {std_file.name}")
        return

    suffix = std_file.suffix.lower()
    created_at = now_iso()

    # 1) 严格分流：XML/XBRL 完全不进文本通道
    if suffix in {".xml", ".xbrl", ".xsd"}:
        print(f"[skip-xml] {std_file.name}")
        return

    # 2) HTML/TXT：做抽取 + 分流
    if suffix in {".htm", ".html"}:
        try:
            text_items, facts_like_items = parse_text_from_html(std_file)
        except Exception as e:
            print(f"[error] 解析 HTML 失败: {std_file.name} -> {e}")
            return
    else:
        # 纯文本 .txt：简化处理
        try:
            raw = std_file.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[error] 读取失败: {std_file.name} -> {e}")
            return

        text_items, facts_like_items = [], []
        seen = set()
        j = 0
        for l in raw.splitlines():
            l = normalize_spaces(l)
            if not l or ((not TEXT_RETRIEVAL_ONLY) and is_noise_line(l)) or l in seen:
                continue
            seen.add(l)
            j += 1
            toks = count_tokens(l)
            # 仍然按 chunk 逻辑切
            if toks > CHUNK_SIZE_TOK:
                chunks = chunk_by_tokens(l, CHUNK_SIZE_TOK, CHUNK_OVERLAP_TOK)
                for si, sj, piece in chunks:
                    text_items.append({
                        "idx_source": j,
                        "tag": "p",
                        "text": piece,
                        "tokens": count_tokens(piece),
                        "chunk_tok_start": si,
                        "chunk_tok_end": sj,
                        "css_path": None,
                        "part": None, "item": None, "heading": None, "section_path": None,
                    })
            else:
                text_items.append({
                    "idx_source": j, "tag": "p", "text": l, "tokens": toks,
                    "css_path": None, "part": None, "item": None, "heading": None, "section_path": None,
                })

    # 写出 text.jsonl
    if text_items:
        out_text = out_dir / "text.jsonl"
        with out_text.open("w", encoding="utf-8") as f:
            for x in text_items:
                # 统一 schema 包装
                rec = {
                    "schema_version": SCHEMA_VERSION,
                    "source_path": meta["source_path"],
                    "ticker": meta["ticker"],
                    "form": meta["form"],
                    "year": meta["year"],
                    "accno": meta["accno"],
                    "doc_date": meta.get("doc_date"),              # 我们提供 css_path 作为定位
                    "css_path": x.get("css_path"),
                    "language": LANG_DEFAULT,
                    "tokens": x.get("tokens"),
                    "created_at": created_at,
                    "id": str(uuid.uuid4()),
                    "section": x.get("section_path"),
                    "heading": x.get("heading"),
                    "statement_hint": None,
                    "text": x.get("text"),
                    "tag": x.get("tag"),
                    "idx_source": x.get("idx_source"),
                    "chunk_tok_start": x.get("chunk_tok_start"),
                    "chunk_tok_end": x.get("chunk_tok_end"),
                    "part": x.get("part"),
                    "item": x.get("item"),
                    "page_no": x.get("page_no"),
                    "page_anchor": x.get("page_anchor"),
                    "xpath": x.get("xpath"),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[ok] text  -> {out_text} ({len(text_items)} lines)")
    else:
        print(f"[warn] 无可用文本: {std_file.name}")

    # 写出 facts_like.jsonl
    if WRITE_FACTS_LIKE and facts_like_items:
        out_facts_like = out_dir / "facts_like.jsonl"
        with out_facts_like.open("w", encoding="utf-8") as f:
            for x in facts_like_items:
                rec = {
                    "schema_version": SCHEMA_VERSION,
                    "source_path": meta["source_path"],
                    "ticker": meta["ticker"],
                    "form": meta["form"],
                    "year": meta["year"],
                    "accno": meta["accno"],
                    "doc_date": meta.get("doc_date"),
                    "language": LANG_DEFAULT,
                    "created_at": created_at,
                    "id": str(uuid.uuid4()),
                    "tag": x.get("tag"),
                    "idx_source": x.get("idx_source"),
                    "text": x.get("text_raw"),
                    "fact_token_hits": x.get("fact_token_hits"),
                    "css_path": x.get("css_path"),
                    "part": x.get("part"),
                    "item": x.get("item"),
                    "heading": x.get("heading"),
                    "section": x.get("section_path"),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[ok] facts_like -> {out_facts_like} ({len(facts_like_items)} lines)")


# -----------------------------
# 批量
# -----------------------------
def batch_parse():
    SKIP_EXTS = {".xml", ".xsd", ".xbrl"}
    TEXT_OK_EXTS = {".htm", ".html", ".txt"}

    if not STD.exists():
        print(f"[error] 未找到目录: {STD.resolve()}")
        return

    files = sorted(STD.glob("*.*"))
    print(f"[info] 标准文件数: {len(files)} in {STD}")

    seen = set()

    for f in files:
        m = FNAME_RE.match(f.name) or NAME_RE.match(f.name)
        if not m:
            # 非标准命名跳过，避免写到意外目录
            print(f"[skip-name] {f.name}")
            continue

        # 用新的 meta 解析函数做一次规范化
        meta = parse_meta_from_filename(f)
        form = meta["form"]
        accno = meta["accno"]
        ext_dot = "." + meta["ext"]

        # 跳过 XML 等
        if ext_dot in SKIP_EXTS:
            print(f"[skip-xml] {f.name}")
            continue

        # 仅处理允许的文本扩展名
        if ext_dot not in TEXT_OK_EXTS:
            print(f"[skip-ext] {f.name}")
            continue

        key = (meta["ticker"], meta["year"], form, accno)
        if key in seen:
            print(f"[skip-dup] {f.name} (same filing: {key})")
            continue
        seen.add(key)

        out_dir = Path(f"data/processed/{meta['ticker']}/{meta['year']}/{form}_{accno}")
        print(f"[parse] {f.name} -> {out_dir}")
        try:
            parse_one(f, out_dir)
        except Exception as e:
            print(f"[error] 处理失败: {f.name} -> {e}")

if __name__ == "__main__":
    batch_parse()
    print("✅ 完成：输出 RAG 文本 (text.jsonl)，可选高密 XBRL 行 (facts_like.jsonl)")
