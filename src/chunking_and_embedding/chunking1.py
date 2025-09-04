#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse, json, re, sys
from typing import List, Dict, Any

# -------- 切块参数（只切 text） --------
MAX_TOKENS_PER_CHUNK = 900     # ≈ 3.6k 字符
MAX_CHARS_PER_CHUNK  = 3600
SENT_SPLIT_RE = re.compile(r'(?<=[\.\?\!。！？])\s+')
TABLE_LEADIN_RE = re.compile(
    r"(the following table|the following|were as follows|are as follows|presented?\s+(?:below|as follows)|\(in (?:millions|thousands)\):)$",
    re.IGNORECASE
)
TABLE_MARK_RE = re.compile(r'^\s*\[(?:TABLE|Table)\b.*\]\s*$')

def approx_tokens(s: str) -> int:
    return max(1, len(s)//4)

def is_table_marker(line: str) -> bool:
    s = line.strip()
    return bool(TABLE_MARK_RE.match(s) or TABLE_LEADIN_RE.search(s))

def title_from_meta(heading: str | None, m: Dict[str, Any]) -> str:
    h = (heading or "").strip() or "Untitled"
    ticker = (m.get("ticker") or "").strip()
    form   = (m.get("form") or "").strip()
    fy     = m.get("fy") or m.get("year")
    fy_str = f"FY{fy}" if fy else ""
    suffix = " ".join(x for x in [ticker, form, fy_str] if x)
    return f"{h} [{suffix}]".strip() if suffix else h

def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ls = line.strip()
            if ls:
                rows.append(json.loads(ls))
    return rows

def group_by_heading(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    cur_h, cur_paras, cur_meta = None, [], {}
    def commit():
        nonlocal cur_h, cur_paras, cur_meta
        if cur_h is not None and cur_paras:
            sections.append({"heading": cur_h, "paras": cur_paras[:], "meta_base": cur_meta.copy()})
        cur_h, cur_paras, cur_meta = None, [], {}
    for r in rows:
        h = r.get("heading")
        t = (r.get("text") or "").strip()
        if cur_h is None or h != cur_h:
            if cur_h is not None: commit()
            cur_h = h
            cur_meta = {k: r.get(k) for k in [
                "ticker","form","accno","fy","fq","year","doc_date","source_path",
                "created_at","page_no","page_anchor","language"
            ]}
        if t:
            cur_paras.append(t)
    commit()
    return sections

def emit_chunks_from_paragraphs(paras: List[str], push, max_tok=MAX_TOKENS_PER_CHUNK, max_chars=MAX_CHARS_PER_CHUNK):
    buf: List[str] = []; btok = bchr = 0
    def flush():
        nonlocal buf, btok, bchr
        if buf:
            push("\n\n".join(buf))
            buf.clear(); btok = 0; bchr = 0
    for p in paras:
        if is_table_marker(p):
            flush(); push(p.strip()); continue
        ptok, pchr = approx_tokens(p), len(p)
        if buf and (btok + ptok > max_tok or bchr + pchr > max_chars):
            flush()
        if ptok <= max_tok and pchr <= max_chars:
            if (btok + ptok <= max_tok) and (bchr + pchr <= max_chars):
                buf.append(p); btok += ptok; bchr += pchr
            else:
                flush(); buf.append(p); btok = ptok; bchr = pchr
        else:
            sents = [s.strip() for s in SENT_SPLIT_RE.split(p) if s.strip()]
            cur: List[str] = []; ctok = cchr = 0
            for s in sents:
                stok, schr = approx_tokens(s), len(s)
                if cur and (ctok + stok > max_tok or cchr + schr > max_chars):
                    flush(); push(" ".join(cur)); cur=[]; ctok=cchr=0
                if stok > max_tok or schr > max_chars:
                    flush(); push(s)
                else:
                    cur.append(s); ctok += stok; cchr += schr
            if cur:
                flush(); push(" ".join(cur))
    flush()

def chunk_one_file(in_path: Path, out_path: Path, max_tokens: int, max_chars: int, overwrite: bool=False) -> dict:
    if out_path.exists() and not overwrite:
        return {"in": str(in_path), "out": str(out_path), "status": "skip (exists)"}
    rows = load_rows(in_path)
    sections = group_by_heading(rows)
    chunks: List[Dict[str, Any]] = []
    for sec in sections:
        heading, paras, meta_b = sec["heading"], sec["paras"], sec["meta_base"]
        local_blocks: List[str] = []
        emit_chunks_from_paragraphs(paras, local_blocks.append, max_tokens, max_chars)
        for content in local_blocks:
            fy = meta_b.get("fy") or meta_b.get("year")
            fq = meta_b.get("fq") or "FY"
            accno = meta_b.get("accno") or "UNKNOWN"
            chunk_id = f"{accno}::text::chunk-{len(chunks)}"
            meta = {
                "ticker":     meta_b.get("ticker"),
                "form":       meta_b.get("form"),
                "accno":      accno,
                "fy":         fy,
                "fq":         fq,
                "doc_date":   meta_b.get("doc_date"),
                "created_at": meta_b.get("created_at"),
                "source_path":meta_b.get("source_path"),
                "heading":    heading,
                "chunk_id":   chunk_id,
            }
            chunks.append({
                "id": chunk_id,
                "title": title_from_meta(heading, {**meta_b, "fy": fy}),
                "content": content,
                "meta": meta
            })
    total = len(chunks)
    for i, ch in enumerate(chunks):
        ch["meta"]["chunk_index"] = i
        ch["meta"]["chunk_count"] = total
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    return {"in": str(in_path), "out": str(out_path), "status": f"ok ({total} chunks)"}

def compute_out_path(in_file: Path, in_root: Path, out_root: Path) -> Path:
    """
    把 data/silver/**/text_corpus.jsonl -> data/chunked/**/text_chunks.jsonl
    其他文件名如果被处理，则按 mirror/<file_stem>/text_chunks.jsonl 兜底。
    """
    rel = in_file.relative_to(in_root)
    if in_file.name == "text_corpus.jsonl":
        return out_root / rel.parent / "text_chunks.jsonl"
    # 兜底（基本不会触发，但保证健壮性）
    return out_root / rel.parent / in_file.stem / "text_chunks.jsonl"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", default="data/silver", help="输入根目录（默认 data/silver）")
    ap.add_argument("--out-root", default="data/chunked", help="输出根目录（默认 data/chunked）")
    ap.add_argument("--workers", type=int, default=6, help="并发线程数")
    ap.add_argument("--max_tokens", type=int, default=MAX_TOKENS_PER_CHUNK)
    ap.add_argument("--max_chars",  type=int, default=MAX_CHARS_PER_CHUNK)
    ap.add_argument("--overwrite", action="store_true", help="已存在则覆盖")
    args = ap.parse_args()

    in_root  = Path(args.in_root).resolve()
    out_root = Path(args.out_root).resolve()

    # 只匹配 text_corpus.jsonl
    files = sorted(in_root.rglob("text_corpus.jsonl"))

    if not files:
        print(f"[warn] No files matched: {in_root} / pattern=text_corpus.jsonl")
        sys.exit(0)

    print(f"[info] Found {len(files)} file(s). workers={args.workers}")
    jobs = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in files:
            outp = compute_out_path(f, in_root, out_root)
            jobs.append(ex.submit(
                chunk_one_file, f, outp, args.max_tokens, args.max_chars, args.overwrite
            ))
        for fut in as_completed(jobs):
            res = fut.result()
            print(f"[{res['status']}] {res['in']} -> {res['out']}")

if __name__ == "__main__":
    main()



'''
python src/chunking_and_embedding/chunking1.py `
  --in-root data\silver `
  --out-root data\chunked `
  --workers 8 `
  --max_tokens 500 `
  --max_chars 3600 `
  --overwrite
  '''