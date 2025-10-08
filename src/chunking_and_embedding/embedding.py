#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embed all text_chunks.jsonl under data/chunked/
-> build one FAISS index in data/index/text_index.faiss (with explicit IDs)
-> save idmap + meta (+ build_info) for lookup (order-agnostic, no more misalignment)

Key changes:
- [NEW] Stable traversal plus conflict resolution by mtime/path keywords/text hash to avoid mismatched records per rid
- [NEW] Enforce full rid (accno::text::chunk-N) as the unique key and derive chunk_index from it
- [NEW] Persist text sha256 and source path into idmap.jsonl & meta.jsonl for downstream validation
- [NEW] Emit build_info.json recording inputs and model for auditing
"""

from __future__ import annotations
import argparse, json, sys, hashlib, os, time
from pathlib import Path
from typing import List, Dict, Iterable, Optional, Tuple, TYPE_CHECKING

import numpy as np
import re

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# ========= Regex helpers =========
# [NEW] Enforce strict rid format (avoid bare chunk-XXX as primary key)
RID_RE   = re.compile(r'^\d{10}-\d{2}-\d{6}::text::chunk-\d+$')
CHUNK_RE = re.compile(r'::text::chunk-(\d+)$')

def parse_chunk_no(rid: str) -> Optional[int]:  # [NEW]
    m = CHUNK_RE.search(rid or "")
    return int(m.group(1)) if m else None

def sha256_text(s: str) -> str:  # [NEW]
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def file_mtime(p: Path) -> float:  # [NEW]
    try:
        return p.stat().st_mtime
    except Exception:
        return 0.0

def now_iso() -> str:  # [NEW]
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

# ========= JSONL reader =========
def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"[WARN] {path} line {i} invalid json: {e}", file=sys.stderr)

# ========= Text construction =========
def build_text(rec: Dict, use_title=True) -> str:
    title = (rec.get("title") or "").strip()
    content = (rec.get("content") or "").strip()
    if use_title and title:
        return f"{title}\n\n{content}"
    return content

def make_snippet(text: str, max_chars: int = 420) -> str:
    s = " ".join((text or "").split())
    return s[:max_chars] + ("..." if len(s) > max_chars else "")

# ========= Embedding =========
def encode_texts(model: "SentenceTransformer", texts: List[str], batch_size=64) -> np.ndarray:
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,   # cosine via IP
        show_progress_bar=True,
    )
    return np.asarray(vecs, dtype="float32")

def stable_id(s: str) -> int:
    """Deterministic 64-bit non-negative integer from string id."""
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest()
    return int(h, 16) & ((1 << 63) - 1)

# ========= Conflict priority via path keywords =========
# [NEW] Resolve conflicts via path keywords and mtime to pick the authoritative record
def path_priority_score(path: str, prefer_keywords: Tuple[str, ...]) -> int:
    p = path.lower()
    score = 0
    for kw in prefer_keywords:
        if kw and kw.lower() in p:
            score += 1
    return score

# ========= Main workflow =========
def build_index(
    input_root: Path | str = Path("data/chunked"),
    output_dir: Path | str = Path("data/index"),
    *,
    model_name: str = "BAAI/bge-base-en-v1.5",
    batch_size: int = 64,
    use_title: bool = False,
    prefer_keywords: Tuple[str, ...] = ("latest", "final", "clean"),
    strict_rid: bool = False,
) -> dict:
    """Build an embedding index from chunked text files."""
    from sentence_transformers import SentenceTransformer
    import faiss

    input_root = Path(input_root)
    output_dir = Path(output_dir)
    prefer_keywords = tuple(kw for kw in prefer_keywords if kw)

    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()

    files = sorted(
        input_root.rglob("text_chunks.jsonl"),
        key=lambda x: (x.parent.as_posix(), x.name),
    )
    print(f"[INFO] found {len(files)} files (sorted)")

    by_id: Dict[str, Dict] = {}

    for f in files:
        mtime = file_mtime(f)
        for rec in iter_jsonl(f):
            rid = rec.get("id") or (rec.get("meta", {}) or {}).get("chunk_id")
            if not rid:
                continue
            if strict_rid and not RID_RE.match(str(rid)):
                print(f"[WARN] skip non-full rid: {rid} (file={f})", file=sys.stderr)
                continue

            text_content = build_text(rec, use_title=use_title)
            if not text_content:
                continue

            h = sha256_text(text_content)
            entry = {
                "rid": rid,
                "title": rec.get("title"),
                "heading": rec.get("heading"),
                "snippet": make_snippet(text_content, 420),
                "meta": rec.get("meta"),
                "text": text_content,
                "text_hash": h,
                "src_path": str(f),
                "src_mtime": mtime,
                "path_prio": path_priority_score(str(f), prefer_keywords),
            }

            prev = by_id.get(rid)
            if prev is None:
                by_id[rid] = entry
            else:
                if prev["text_hash"] == h:
                    if not prev.get("heading") and entry.get("heading"):
                        prev["heading"] = entry["heading"]
                    if not prev.get("title") and entry.get("title"):
                        prev["title"] = entry["title"]
                    if entry["path_prio"] > prev["path_prio"]:
                        prev["src_path"] = entry["src_path"]
                        prev["src_mtime"] = entry["src_mtime"]
                        prev["path_prio"] = entry["path_prio"]
                else:
                    choose = prev
                    if entry["path_prio"] > prev["path_prio"]:
                        choose = entry
                    elif entry["path_prio"] == prev["path_prio"] and entry["src_mtime"] >= prev["src_mtime"]:
                        choose = entry
                    if choose is entry:
                        by_id[rid] = entry
                        print(
                            f"[WARN] rid conflict (diff content): kept NEW {entry['src_path']} ; drop OLD {prev['src_path']} | id={rid}",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"[WARN] rid conflict (diff content): kept OLD {prev['src_path']} ; drop NEW {entry['src_path']} | id={rid}",
                            file=sys.stderr,
                        )

    all_ids: List[str] = []
    all_id64: List[int] = []
    all_texts: List[str] = []
    all_metas: List[Dict] = []

    for rid, item in by_id.items():
        chunk_index = parse_chunk_no(rid)
        all_ids.append(rid)
        all_id64.append(stable_id(rid))
        all_texts.append(item["text"])
        meta_out = {
            "id": rid,
            "title": item.get("title"),
            "heading": item.get("heading"),
            "snippet": item.get("snippet"),
            "meta": item.get("meta"),
            "chunk_index": chunk_index,
            "text_hash": item.get("text_hash"),
            "src_path": item.get("src_path"),
            "src_mtime": item.get("src_mtime"),
        }
        all_metas.append(meta_out)

    print(f"[INFO] total unique records: {len(all_ids)}")

    vecs = encode_texts(model, all_texts, batch_size=batch_size)
    assert vecs.shape[0] == len(all_ids)

    base = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap2(base)
    id_array = np.asarray(all_id64, dtype="int64")
    index.add_with_ids(vecs, id_array)
    print(f"[INFO] index built: {index.ntotal} vectors")

    output_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(output_dir / "text_index.faiss"))

    with (output_dir / "idmap.jsonl").open("w", encoding="utf-8") as f:
        for rid in all_ids:
            item = by_id[rid]
            f.write(
                json.dumps(
                    {
                        "id": rid,
                        "id64": stable_id(rid),
                        "src_path": item["src_path"],
                        "text_hash": item["text_hash"],
                    },
                    ensure_ascii=False,
                ) + "\n"
            )

    with (output_dir / "meta.jsonl").open("w", encoding="utf-8") as f:
        for rec in all_metas:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    build_info = {
        "built_at_utc": now_iso(),
        "model": model_name,
        "input_root": str(input_root),
        "files_count": len(files),
        "prefer_keywords": prefer_keywords,
        "records": len(all_ids),
    }
    with (output_dir / "build_info.json").open("w", encoding="utf-8") as bf:
        json.dump(build_info, bf, ensure_ascii=False, indent=2)

    print(f"[DONE] saved index + idmap + meta (+ build_info) under {output_dir}")
    return {
        "records": len(all_ids),
        "files": len(files),
        "output_dir": str(output_dir),
    }


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=Path("data/chunked"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/index"))
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--use-title", action="store_true", help="prepend title to content")
    ap.add_argument(
        "--prefer-keywords",
        default="latest,final,clean",
        help="comma separated path keywords for conflict resolving",
    )
    ap.add_argument("--strict-rid", action="store_true", help="require full rid format accno::text::chunk-N")
    args = ap.parse_args(list(argv) if argv is not None else None)

    prefer_keywords = tuple(s.strip() for s in args.prefer_keywords.split(",") if s.strip())

    build_index(
        input_root=args.input_root,
        output_dir=args.output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        use_title=args.use_title,
        prefer_keywords=prefer_keywords,
        strict_rid=args.strict_rid,
    )



if __name__ == "__main__":
    main()
