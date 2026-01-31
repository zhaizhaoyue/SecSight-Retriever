#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse, json, re, sys
from typing import List, Dict, Any

# -------- Chunking parameters (text-only) --------
MAX_TOKENS_PER_CHUNK = 900     # About 3.6k characters
MAX_CHARS_PER_CHUNK  = 3600
MIN_TOKENS_PER_CHUNK = 120
OVERLAP_TOKENS = 80
SENT_SPLIT_RE = re.compile(
    r'(?<=[\.\?\!。！？])\s+'          # Sentence terminators
    r'|(?<=;)\s+'                     # Semicolons
    r'|(?<=:)\s+'                     # Colons
    r'|(?<=[—–])\s+'                  # Long or short dashes
)
BULLET_LINE_RE = re.compile(
    r'^\s*(?:[•\-–—]|'
    r'\(\s*(?:[ivxlcdm]+|\d+|[a-zA-Z])\s*\)|'  # (i)/(1)/(a)
    r'(?:\d+\.|[a-zA-Z]\.))\s+'
)

TABLE_LEADIN_RE = re.compile(
    r"(the following table|the following|were as follows|are as follows|presented?\s+(?:below|as follows)|\(in (?:millions|thousands)\):)\s*(?:\[\w+\])?\s*$",
    re.IGNORECASE
)

TABLE_MARK_RE = re.compile(r'^\s*\[(?:TABLE|Table)\b.*\]\s*$')

def approx_tokens(s: str) -> int:
    return max(1, len(s)//4)

def _tail_tokens(txt: str, n: int) -> str:
    toks = txt.split()
    if len(toks) <= n:
        return txt
    return " ".join(toks[-n:])

def _split_units(p: str) -> List[str]:
    """Split by newline first; treat bullet/enum lines as standalone units, otherwise apply SENT_SPLIT_RE."""
    units: List[str] = []
    for line in (p or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if BULLET_LINE_RE.match(line):
            units.append(line)
        else:
            parts = [s.strip() for s in SENT_SPLIT_RE.split(line) if s.strip()]
            units.extend(parts if parts else [line])
    return units

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
        if cur_paras:
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

def emit_chunks_from_paragraphs(
    paras: List[str],
    include_heading: str | None,
    push,
    max_tok=MAX_TOKENS_PER_CHUNK,
    max_chars=MAX_CHARS_PER_CHUNK,
    min_tok=MIN_TOKENS_PER_CHUNK,
    overlap_tok=OVERLAP_TOKENS
):
    """
    Improvements:
    - Merge table lead-ins with adjacent chunks
    - Finer unit splitting (supports bullet lists)
    - Support chunk overlaps via overlap_tok
    - Try to keep each chunk >= min_tok
    - Inject heading at the beginning of each chunk to help retrieval
    """
    buf: List[str] = []       # Buffer of units for the current chunk
    btok = bchr = 0
    last_overlap = ""         # Tail overlap from the previous chunk
    pending_table_leadin = "" # Deferred table lead-in text

    heading_prefix = (include_heading or "").strip()
    heading_prefix = (heading_prefix + "\n\n") if heading_prefix else ""

    def _cur_buf_text() -> str:
        parts = []
        if last_overlap:
            parts.append(last_overlap)
        if pending_table_leadin:
            parts.append(pending_table_leadin)
        parts.extend(buf)
        return "\n\n".join(parts).strip()

    def _flush(force=False):
        nonlocal buf, btok, bchr, last_overlap, pending_table_leadin
        if not buf and not pending_table_leadin:
            return
        text = _cur_buf_text()
        if not text:
            # Clear placeholders
            buf.clear(); btok = bchr = 0; pending_table_leadin = ""
            return
        # Try to satisfy the minimum token count unless forcing a flush
        if not force and approx_tokens(text) < min_tok:
            return
        # Emit chunk with heading_prefix prepended to content
        push(heading_prefix + text)
        # Compute overlap
        last_overlap = _tail_tokens(text, overlap_tok)
        # Reset buffers
        buf.clear(); btok = bchr = 0; pending_table_leadin = ""

    for p in paras:
        # Merge table markers/lead-ins into current or next chunk
        if is_table_marker(p):
            # Consume pending lead-ins until the next flush
            pending_table_leadin = (pending_table_leadin + "\n\n" + p.strip()).strip() if pending_table_leadin else p.strip()
            # Do not flush immediately here
            continue

        # Iterate units
        units = _split_units(p)
        for u in units:
            utok, uchr = approx_tokens(u), len(u)
            # Flush if adding the unit would exceed limits
            if buf and (btok + utok > max_tok or bchr + uchr > max_chars):
                _flush(force=True)
            # For oversized units: emit standalone chunk, splitting further when possible
            if utok > max_tok or uchr > max_chars:
                # Flush buffered content first
                _flush(force=True)
                # Re-split the oversized unit by sentences to avoid ragged cuts
                subparts = [s.strip() for s in SENT_SPLIT_RE.split(u) if s.strip()] or [u]
                cur: List[str] = []; ctok = cchr = 0
                for sp in subparts:
                    stok, schr = approx_tokens(sp), len(sp)
                    if cur and (ctok + stok > max_tok or cchr + schr > max_chars):
                        buf = cur[:] ; btok = ctok ; bchr = cchr
                        _flush(force=True)
                        cur.clear(); ctok = cchr = 0
                    cur.append(sp); ctok += stok; cchr += schr
                if cur:
                    buf = cur[:] ; btok = ctok ; bchr = cchr
                    _flush(force=True)
            else:
                # Add to buffer normally
                buf.append(u); btok += utok; bchr += uchr
                # Flush proactively when chunk is already large enough
                if btok >= max_tok * 0.9 or bchr >= max_chars * 0.9:
                    _flush(force=True)

    # Final flush: relax the minimum token guard
    if buf or pending_table_leadin:
        _flush(force=True)

def chunk_one_file(in_path: Path, out_path: Path, max_tokens: int, max_chars: int, overwrite: bool=False) -> dict:
    if out_path.exists() and not overwrite:
        return {"in": str(in_path), "out": str(out_path), "status": "skip (exists)"}
    rows = load_rows(in_path)
    sections = group_by_heading(rows)
    chunks: List[Dict[str, Any]] = []
    OTHER_YEARS_RE = re.compile(
    r"\b20(1\d|2[0-5])\b\s+(?:"
    r"(?:annual\s+report(?:\s+on\s+form)?\s+)?10(?:-|[\u2013\u2014])k"
    r"|form\s+10(?:-|[\u2013\u2014])k"
    r")",
    re.I
    )
    for sec in sections:
        heading, paras, meta_b = sec["heading"], sec["paras"], sec["meta_base"]
        local_blocks: List[str] = []
        emit_chunks_from_paragraphs(
            paras,
            include_heading=heading,   # inject heading into chunk content
            push=local_blocks.append,
            max_tok=max_tokens,
            max_chars=max_chars
        )

        def _section_tag_from_text(t: str) -> str | None:
            tl = (t or "").lower()
            if "item 1a" in tl and "risk" in tl:
                return "risk_factors"
            if "forward-looking" in tl:
                return "fwd_statements"
            if "management’s discussion" in tl or "md&a" in tl or "management's discussion" in tl:
                return "md&a"
            return None


        def _mentions_other_years(text: str) -> bool:
            return bool(OTHER_YEARS_RE.search(text))

        for content in local_blocks:
            fy = meta_b.get("fy") or meta_b.get("year")
            fq = meta_b.get("fq") or "FY"
            accno = meta_b.get("accno") or "UNKNOWN"
            chunk_id = f"{accno}::text::chunk-{len(chunks)}"

            # Fall back to content-based detection if heading is inconclusive
            sec_tag = _section_tag_from_text(heading) or _section_tag_from_text(content)

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
                "section_tag": sec_tag,
                "mentions_other_years": _mentions_other_years(content),
            }

            chunks.append({
                "id": chunk_id,
                "title": title_from_meta(heading, {**meta_b, "fy": fy}),
                "content": content,   # content already includes heading prefix
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
    Map data/silver/**/text_corpus.jsonl -> data/chunked/**/text_chunks.jsonl
    Fallback: mirror/<file_stem>/text_chunks.jsonl for other filenames.
    """
    rel = in_file.relative_to(in_root)
    if in_file.name == "text_corpus.jsonl":
        return out_root / rel.parent / "text_chunks.jsonl"
    # Safety fallback (rare, but keeps things robust)
    return out_root / rel.parent / in_file.stem / "text_chunks.jsonl"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", default="data/silver", help="Input root directory (default: data/silver)")
    ap.add_argument("--out-root", default="data/chunked", help="Output root directory (default: data/chunked)")
    ap.add_argument("--workers", type=int, default=6, help="Number of worker threads")
    ap.add_argument("--max_tokens", type=int, default=MAX_TOKENS_PER_CHUNK)
    ap.add_argument("--max_chars",  type=int, default=MAX_CHARS_PER_CHUNK)
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    in_root  = Path(args.in_root).resolve()
    out_root = Path(args.out_root).resolve()

    # Only match text_corpus.jsonl
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
