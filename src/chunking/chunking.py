#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chunk everything except labels.jsonl:
- Mirror tree from <input_root> (default: data/silver) to <output_root> (default: data/chunked)
- text_corpus.jsonl / text.jsonl     -> token sliding-window chunks  -> *text_chunks.jsonl
- fact.jsonl                          -> group-by-N record chunks    -> *fact_chunks.jsonl
- calculation_edges.jsonl             -> group-by-N record chunks    -> *calc_chunks.jsonl
- definition_arcs.jsonl               -> group-by-N record chunks    -> *def_chunks.jsonl
- labels_best.jsonl / labels_wide.jsonl -> group-by-N record chunks  -> *labels_*_chunks.jsonl
- labels.jsonl                        -> copy as-is (no chunking)
- Unknown *.jsonl (not labels.jsonl)  -> default group-by-N record chunks -> *generic_chunks.jsonl
"""

from __future__ import annotations
import argparse
import json
import shutil
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

# ----------------------------- Utilities -----------------------------
def discover_project_root(start: Optional[Path] = None) -> Path:
    p = (start or Path(__file__).resolve()).parent
    for parent in [p, *p.parents]:
        if (parent / ".git").exists():
            return parent
    return p.parent.parent  # fallback: typical src/*/chunking.py

def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n

# ----------------------------- Chunkers -----------------------------
def _as_tokens(text: str) -> List[str]:
    # 简单健壮：按空白切分近似 token
    return text.split()

def chunk_text(text: str, max_tokens: int, overlap: int) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    返回:
      - chunks: List[str]
      - spans : List[(start, end)]  以 token 为单位的半开区间 [start, end)
    """
    toks = _as_tokens(text)
    if not toks:
        return [], []
    step = max(1, max_tokens - overlap)
    chunks: List[str] = []
    spans: List[Tuple[int, int]] = []
    i = 0
    while i < len(toks):
        start = i
        end = min(i + max_tokens, len(toks))
        window = toks[start:end]
        if not window:
            break
        chunks.append(" ".join(window))
        spans.append((start, end))
        i += step
    return chunks, spans

TEXT_FIELDS = ("text", "content", "text_clean", "text_raw", "paragraph")

def process_text_jsonl(
    in_file: Path,
    out_file: Path,
    *,
    relpath_under_input: Optional[str],
    max_tokens: int,
    overlap: int,
    min_row_tokens: int = 6,  # 仍然保留：短行不单独成块，会与邻近行合并
    default_schema_version: str = "0.3.0",
    default_language: str = "en",
) -> int:
    """
    文本按“整文件级”滑窗切块：
      - 逐行读取，短行(< min_row_tokens)合并到缓冲，不单独出块；
      - 缓冲 token 数 >= max_tokens 则吐出一个 chunk；
      - 相邻 chunk 保留 overlap token 的尾部作为上下文；
      - 不输出原始 'tokens' 字段；增强 tokens_in_chunk / row_tokens_total / chunk_span_tokens。
    """
    # 一些小工具
    def toks(s: str) -> List[str]:
        return (s or "").split()

    def tok_len(s: str) -> int:
        return len(toks(s))

    # 全局缓冲（跨行）
    buf_texts: List[str] = []          # 保存原始行文本（按行）
    buf_tok_counts: List[int] = []     # 每行 token 数（和 buf_texts 对齐）
    buf_total = 0                      # 缓冲累计 token 数
    total_seen_tokens = 0              # 整个文件累计 token（用于 span 辅助理解）

    # 记录一些元数据
    first_row_meta: Optional[Dict[str, Any]] = None
    accno_fallback: Optional[str] = None
    out_rows: List[Dict[str, Any]] = []

    # 收集所有可用的文本行（保持原有 TEXT_FIELDS 逻辑）
    lines: List[Tuple[str, Dict[str, Any]]] = []
    for row in read_jsonl(in_file):
        # 记住首行 meta 以便透传 schema/language 等
        if first_row_meta is None:
            first_row_meta = dict(row)
        if not accno_fallback and row.get("accno"):
            accno_fallback = row.get("accno")

        txt = None
        used_key = None
        for k in TEXT_FIELDS:
            v = row.get(k)
            if isinstance(v, str) and v.strip():
                txt = v
                used_key = k
                break
        if not txt:
            continue

        # 过滤过短的行：< min_row_tokens 的行不单独出块，先合并到缓冲中
        if tok_len(txt) < min_row_tokens:
            # 先入缓冲（等和邻行组成更大的上下文）
            buf_texts.append(txt)
            ct = tok_len(txt)
            buf_tok_counts.append(ct)
            buf_total += ct
            total_seen_tokens += ct
            continue

        # 对于长度合适的行，先把它推进缓冲，再根据预算决策是否吐块
        buf_texts.append(txt)
        ct = tok_len(txt)
        buf_tok_counts.append(ct)
        buf_total += ct
        total_seen_tokens += ct

        # 预算达标，尝试吐块（可一次吐多个，直到 buf_total < max_tokens）
        while buf_total >= max_tokens:
            # 1) 组装 chunk 文本（整个缓冲连接）
            chunk_text = "\n".join(buf_texts).strip()
            # 2) 截取头部的 max_tokens 作为这个 chunk 的正文（近似按空白 token）
            token_list = toks(chunk_text)
            head_tokens = token_list[:max_tokens]
            chunk_body = " ".join(head_tokens)
            # 3) 输出一个 chunk
            o = dict(first_row_meta or {})
            o.pop("tokens", None)
            o["schema_version"] = (o.get("schema_version") or default_schema_version)
            o["language"] = (o.get("language") or default_language)
            o["text"] = chunk_body

            # chunk_id 采用 accno 优先，否则用文件名
            base = accno_fallback or in_file.stem
            o["chunk_id"] = f"{base}::text::chunk-{len(out_rows)}"
            o["chunk_index"] = len(out_rows)  # 先填，稍后回填 chunk_count
            o["chunk_count"] = 0
            o["source_file"] = str(in_file)
            if relpath_under_input is not None:
                o["relpath"] = relpath_under_input
            o["text_field"] = "text"  # 聚合后是统一 text 输出

            o["tokens_in_chunk"] = len(head_tokens)
            # row_tokens_total 改为“全文件累计 token”（更符合检索统计语义）
            # 如果你更想用“原文本总 token”，可以先把所有行连接后再算一次。
            o["row_tokens_total"] = total_seen_tokens
            # 这里的 span 用“全文件 token 累计”的滑动窗口近似（可选）
            start_token = max(0, total_seen_tokens - buf_total)
            end_token = start_token + len(head_tokens)
            o["chunk_span_tokens"] = [start_token, end_token]

            out_rows.append(o)

            # 4) 构建 overlap：保留尾部 overlap token，其余从左侧移除
            keep = min(overlap, max_tokens)  # overlap 不应超过 max_tokens
            tail_tokens = token_list[max_tokens - keep:max_tokens]  # 要保留的尾部 token
            # 用行级近似把缓冲左侧弹出，直到只剩下 overlap 的 token 数
            # 先把缓冲完全展开为 token，再回填为一行（简单近似）
            # 为保持实现最小化，这里直接重建缓冲为一个“单行”的 overlap 字符串
            buf_texts = [" ".join(tail_tokens)]
            buf_tok_counts = [len(tail_tokens)]
            buf_total = len(tail_tokens)

    # 文件结束后，如果缓冲里还有内容，也要吐出 1 个尾块
    if buf_total > 0 and buf_texts:
        token_list = " ".join(buf_texts).split()
        chunk_body = " ".join(token_list[:max_tokens])
        o = dict(first_row_meta or {})
        o.pop("tokens", None)
        o["schema_version"] = (o.get("schema_version") or default_schema_version)
        o["language"] = (o.get("language") or default_language)
        o["text"] = chunk_body
        base = accno_fallback or in_file.stem
        o["chunk_id"] = f"{base}::text::chunk-{len(out_rows)}"
        o["chunk_index"] = len(out_rows)
        o["chunk_count"] = 0
        o["source_file"] = str(in_file)
        if relpath_under_input is not None:
            o["relpath"] = relpath_under_input
        o["text_field"] = "text"
        o["tokens_in_chunk"] = min(len(token_list), max_tokens)
        o["row_tokens_total"] = total_seen_tokens
        start_token = max(0, total_seen_tokens - buf_total)
        end_token = start_token + o["tokens_in_chunk"]
        o["chunk_span_tokens"] = [start_token, end_token]
        out_rows.append(o)

    # 回填 chunk_count
    total_chunks = len(out_rows)
    for i, r in enumerate(out_rows):
        r["chunk_index"] = i
        r["chunk_count"] = total_chunks

    return write_jsonl(out_file, out_rows)



def process_group_jsonl(
    in_file: Path,
    out_file: Path,
    *,
    group_size: int,
    relpath_under_input: Optional[str] = None,
    add_summary_text: bool = False,
    token_budget: Optional[int] = None,  # 新增：近似 token 上限（例如 350）
) -> int:
    """
    分两步：
      1) 先按 linkrole 分桶（同一报表片段放一起，语义更集中）
      2) 在每个桶中按 group_size 切块；若指定 token_budget，就依据估算的 token 数提前换组
    """
    rows = list(read_jsonl(in_file))
    if not rows:
        return write_jsonl(out_file, [])

    # -------- helpers --------
    def est_edge_tokens(r: Dict[str, Any]) -> int:
        # 先看 fact 的可读文本
        if r.get("rag_text"):
            s = str(r.get("rag_text"))
            return max(12, (len(s) // 4) + 1)  # ~4 chars ≈ 1 token
        # 退化：概念+值+期间
        if r.get("concept") or r.get("qname"):
            s = "|".join([
                str(r.get("concept") or r.get("qname") or ""),
                str(r.get("value_display") or r.get("value_raw") or ""),
                str(r.get("period_label") or ""),
            ])
            return max(12, (len(s) // 4) + 1)

        # 原有 cal/def 逻辑
        parts = []
        parts.append(r.get("parent_concept") or "")
        parts.append(r.get("child_concept") or "")
        parts.append(r.get("from_concept") or "")
        parts.append(r.get("to_concept") or "")
        parts.append(r.get("arcrole") or "")
        parts.append(r.get("preferred_label") or "")
        parts.append(r.get("linkrole") or "")
        s = "|".join(parts)
        for ns in ("us-gaap:", "dei:", "srt:", "aapl:", "ifrs-full:", "xbrli:"):
            s = s.replace(ns, "")
        return max(8, (len(s) // 4) + 1)


    def pack_with_budget(bucket: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        if not token_budget:
            # 仅按 group_size
            return [bucket[i:i+group_size] for i in range(0, len(bucket), group_size)]
        groups: List[List[Dict[str, Any]]] = []
        cur: List[Dict[str, Any]] = []
        cur_tok = 0
        for r in bucket:
            e = est_edge_tokens(r)
            # 如果当前空且单条已超预算，也要容忍（避免死循环）
            if cur and (cur_tok + e) > token_budget:
                groups.append(cur)
                cur, cur_tok = [], 0
            cur.append(r)
            cur_tok += e
            if len(cur) >= group_size:
                groups.append(cur)
                cur, cur_tok = [], 0
        if cur:
            groups.append(cur)
        return groups

    # -------- 1) linkrole 分桶 --------
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        lr = r.get("linkrole") or "__NA__"
        buckets.setdefault(lr, []).append(r)

# -------- 2) 每桶切块 + 汇总输出 --------
    out_rows: List[Dict[str, Any]] = []

    # 先枚举所有组，确定 chunk_count
    all_groups: List[Tuple[str, List[Dict[str, Any]]]] = []
    for lr, bucket in buckets.items():
        for grp in pack_with_budget(bucket):
            all_groups.append((lr, grp))
    total = len(all_groups)

    # ✅ 逐组写出（注意：以下全部代码都在这个 for 里！）
    for gidx, (lr, g) in enumerate(all_groups):
        edge_count = len(g)
        tickers     = sorted({r.get("ticker") for r in g if r.get("ticker")})
        forms       = sorted({r.get("form") for r in g if r.get("form")})
        years       = sorted({r.get("year") for r in g if r.get("year") is not None})
        accnos      = sorted({r.get("accno") for r in g if r.get("accno")})
        file_types  = sorted({r.get("file_type") for r in g if r.get("file_type")})

        # 兼容 cal/def 的端点聚合
        def _endpoints(r):
            if "parent_concept" in r or "child_concept" in r:   # calculation_edges
                return r.get("parent_concept"), r.get("child_concept")
            else:                                               # definition_arcs
                return r.get("from_concept"), r.get("to_concept")

        parents   = sorted({p for p, _ in (_endpoints(r) for r in g) if p})
        children  = sorted({c for _, c in (_endpoints(r) for r in g) if c})
        arcroles  = sorted({r.get("arcrole") for r in g if r.get("arcrole")})
        linkroles = sorted({r.get("linkrole") for r in g if r.get("linkrole")})

        # chunk_id 前缀
        acc = next((r.get("accno") for r in g if r.get("accno")), None)
        rp  = (relpath_under_input or str(in_file.name)).replace("\\", "/").replace("/", "|")
        base = acc or rp

        # kind（cal/def/fact/generic）
        if len(file_types) == 1:
            kind = file_types[0] or "generic"
        else:
            stem = in_file.stem.lower()
            if "calculation" in stem:
                kind = "cal"
            elif "definition" in stem or stem.startswith("def"):
                kind = "def"
            elif "fact" == stem:
                kind = "fact"
            else:
                kind = "generic"

        # === 修正块：仅对 fact 的 edges 做最小兜底 ===
        if kind == "fact":
            balance_like_re = re.compile(
                r"(?:CashAndCashEquivalents(?:AtCarryingValue)?"
                r"|MarketableSecurities(?:Current|Noncurrent)?"
                r"|Assets|Liabilities|StockholdersEquity|ShareholdersEquity"
                r"|RetainedEarnings|AccumulatedOtherComprehensiveIncome"
                r"|ShortTerm|LongTerm|Debt|Inventory|Receivables"
                r"|PropertyPlantAndEquipment|AccumulatedDepreciation"
                r"|OtherAccruedLiabilities(?:Current|Noncurrent)?"
                r"|AccruedIncomeTaxes(?:Current|Noncurrent)?)",
                flags=re.IGNORECASE
            )
            known_balance = {
                "us-gaap:CashAndCashEquivalentsAtCarryingValue",
                "us-gaap:MarketableSecuritiesCurrent",
                "us-gaap:MarketableSecuritiesNoncurrent",
            }

            for r in g:
                # 1) period 以 context.period 为准（若存在）
                per = ((r.get("context") or {}).get("period") or {})
                for src, dst in (("start_date", "period_start"),
                                 ("end_date",   "period_end"),
                                 ("instant",    "instant")):
                    v = per.get(src)
                    if v is not None:
                        r[dst] = v

                concept = (r.get("concept") or r.get("qname") or "").strip()
                concept_l = concept.lower()

                # 2) TextBlock：去掉度量/小数占位；补个占位 value_display
                if concept_l.endswith("textblock"):
                    for k in ("unit", "unit_normalized", "unit_family", "decimals"):
                        if k in r:
                            r[k] = None
                    if not r.get("value_display"):
                        r["value_display"] = "[HTML TextBlock ~0 chars]"

                # 3) 资产负债表优先级（instant + balance-like/已知科目）
                has_instant = bool(r.get("instant"))
                if has_instant and (
                    balance_like_re.search(concept) or concept in known_balance
                ):
                    r["statement_hint"] = "balance"

                # 4) EPS/股本等典型利润表条目（可选，小修）
                if re.search(
                    r"(EarningsPerShare|DilutedShares|BasicShares|AntidilutiveSecurities|WeightedAverageNumber)",
                    concept, flags=re.IGNORECASE
                ):
                    r["statement_hint"] = "income"

                # 5) 空维度签名统一为 None
                if r.get("dims_signature") == "":
                    r["dims_signature"] = None

        # linkrole 短标签
        lr_tag = (lr or "__NA__").split("/")[-1][:48]
        chunk_id = f"{base}::{kind}::{lr_tag}::group::{gidx}"

        # 清理冗余字段
        for r in g:
            if r.get("preferred_label") is None:
                r.pop("preferred_label", None)

        # 简要摘要（前 10 条）
        lines = []
        for r in g:
            a, b = _endpoints(r)
            if not (a and b):
                continue
            mid = r.get("arcrole") or r.get("weight")
            mid = (mid.split("/")[-1] if isinstance(mid, str) else str(mid)) if mid is not None else ""
            lines.append(f"{a} --{mid}--> {b}")
            if len(lines) >= 10:
                break
        summary_text = "\n".join(lines)

        
        # ✅ 对 fact 的兜底：若还是空，则用 rag_text/概念行生成摘要
        if (not summary_text) and kind == "fact":
            fact_lines = []
            for r in g:
                rt = r.get("rag_text")
                if isinstance(rt, str) and rt.strip():
                    fact_lines.append(rt.strip())
                else:
                    c  = r.get("concept") or r.get("qname") or "(no concept)"
                    vd = r.get("value_display")
                    pl = r.get("period_label")
                    # 只拼有用的
                    parts = [str(c)]
                    if vd: parts.append(str(vd))
                    if pl: parts.append(f"({pl})")
                    fact_lines.append(": ".join(parts))
                if len(fact_lines) >= 40:  # 避免过长
                    break
            summary_text = "\n".join(fact_lines)

        out = {
            "parent_concepts": parents,
            "arcroles":        arcroles,
            "child_concepts":  children,
            "chunk_id":        chunk_id,
            "chunk_index":     gidx,
            "chunk_count":     total,
            "source_file":     str(in_file),
            "edge_count":      edge_count,
            "file_type":       (file_types[0] if len(file_types) == 1 else file_types or kind),
            "tickers":         tickers,
            "forms":           forms,
            "years":           years,
            "accnos":          accnos,
            "linkroles":       [lr],
            "edges":           g,
            "summary_text":    summary_text,
        }
        if relpath_under_input is not None:
            out["relpath"] = relpath_under_input

        # ✅ 永远 append（不受任何 if 的影响）
        out_rows.append(out)

    # 循环结束再统一写出
    return write_jsonl(out_file, out_rows)




# ----------------------------- Orchestrator -----------------------------
FILE_POLICIES = {
    "text_corpus.jsonl": ("text", "text_chunks.jsonl"),
    "text.jsonl": ("text", "text_chunks.jsonl"),
    "fact.jsonl": ("group", "fact_chunks.jsonl"),
    "calculation_edges.jsonl": ("group", "calc_chunks.jsonl"),
    "definition_arcs.jsonl": ("group", "def_chunks.jsonl"),
    "labels_best.jsonl": ("group", "labels_best_chunks.jsonl"),
    "labels_wide.jsonl": ("group", "labels_wide_chunks.jsonl"),
    "labels.jsonl": ("copy", "labels.jsonl"),
}

def run(
    input_root: Path,
    output_root: Path,
    *,
    max_tokens: int = 300,
    overlap: int = 60,
    group_size: int = 50,
    copy_labels: bool = True,
    chunk_unknown_jsonl: bool = True,
) -> Dict[str, int]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    stats = {
        "text_files": 0,
        "group_files": 0,
        "labels_copied": 0,
        "unknown_chunked": 0,
        "skipped": 0,
        "chunks_written": 0,
    }

    for f in input_root.rglob("*.jsonl"):
        if not f.is_file():
            continue
        rel_dir = f.relative_to(input_root).parent
        relpath = str(f.relative_to(input_root)).replace("\\", "/")
        out_dir = (output_root / rel_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        name = f.name
        policy, out_name = FILE_POLICIES.get(name, (None, None))

        # labels.jsonl → copy
        if policy == "copy":
            if copy_labels:
                shutil.copy2(f, out_dir / out_name)
                stats["labels_copied"] += 1
            else:
                stats["skipped"] += 1
            continue

        # 文本滑窗
        if policy == "text":
            n = process_text_jsonl(
                f, out_dir / out_name,
                relpath_under_input=relpath,
                max_tokens=max_tokens, overlap=overlap,
            )
            stats["text_files"] += 1
            stats["chunks_written"] += n
            continue

        # 结构分组
        if policy == "group":
            n = process_group_jsonl(
                f, out_dir / out_name, group_size=group_size,
                relpath_under_input=relpath,
                add_summary_text=True,        # ✅ 打开摘要，便于向量化
                token_budget=350,             # ✅ 防止单块过大（可改 300–500）
            )
            stats["group_files"] += 1
            stats["chunks_written"] += n
            continue

        # 未知的 *.jsonl：按 group 分块（可关闭）
        if policy is None and name != "labels.jsonl":
            if chunk_unknown_jsonl:
                n = process_group_jsonl(
                    f, out_dir / "generic_chunks.jsonl", group_size=group_size,
                    relpath_under_input=relpath,
                    add_summary_text=True,     # 同上
                    token_budget=350,          # 同上
                )
                stats["unknown_chunked"] += 1
                stats["chunks_written"] += n
            else:
                stats["skipped"] += 1
            continue

        stats["skipped"] += 1

    return stats

# ----------------------------- CLI -----------------------------
def parse_args() -> argparse.Namespace:
    prj = discover_project_root()
    ap = argparse.ArgumentParser(description="Chunk all JSONL except labels.jsonl; mirror silver -> chunked.")
    ap.add_argument("--input-root", type=Path, default=prj / "data" / "silver",
                    help="Input root (default: <PROJECT_ROOT>/data/silver)")
    ap.add_argument("--output-root", type=Path, default=prj / "data" / "chunked",
                    help="Output root (default: <PROJECT_ROOT>/data/chunked)")
    ap.add_argument("--max-tokens", type=int, default=300, help="Max tokens per text chunk")
    ap.add_argument("--overlap", type=int, default=48, help="Token overlap between text chunks")
    ap.add_argument("--group-size", type=int, default=20, help="Records per group chunk")
    ap.add_argument("--no-copy-labels", action="store_true", help="Do not copy labels.jsonl")
    ap.add_argument("--no-unknown", action="store_true", help="Skip unknown *.jsonl instead of chunking generically")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    stats = run(
        input_root=args.input_root,
        output_root=args.output_root,
        max_tokens=args.max_tokens,
        overlap=args.overlap,
        group_size=args.group_size,
        copy_labels=not args.no_copy_labels,
        chunk_unknown_jsonl=not args.no_unknown,
    )
    print(json.dumps({"summary": stats}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
