#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embed all text_chunks.jsonl under data/chunked/
→ build one FAISS index in data/index/text_index.faiss
→ save ids + meta for lookup
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"[WARN] {path} line {i} invalid json: {e}", file=sys.stderr)


def build_text(rec: Dict, use_title=True) -> str:
    title = (rec.get("title") or "").strip()
    content = (rec.get("content") or "").strip()
    if use_title and title:
        return f"{title}\n\n{content}"
    return content


def encode_texts(model: SentenceTransformer, texts: List[str], batch_size=64) -> np.ndarray:
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(vecs, dtype="float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=Path("data/chunked"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/index"))
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    # load encoder
    model = SentenceTransformer(args.model)
    dim = model.get_sentence_embedding_dimension()

    all_ids, all_texts, all_metas = [], [], []

    files = list(args.input_root.rglob("text_chunks.jsonl"))
    print(f"[INFO] found {len(files)} files")

    for f in files:
        for rec in iter_jsonl(f):
            rid = rec.get("id") or rec.get("meta", {}).get("chunk_id")
            if not rid:
                continue
            text = build_text(rec)
            if not text:
                continue
            all_ids.append(rid)
            all_texts.append(text)
            all_metas.append({
                "id": rid,
                "title": rec.get("title"),
                "meta": rec.get("meta"),
            })

    print(f"[INFO] total records: {len(all_ids)}")

    # encode all
    vecs = encode_texts(model, all_texts, batch_size=args.batch_size)

    # build FAISS index
    index = faiss.IndexFlatIP(dim)   # inner product, since we normalized
    index.add(vecs)
    print(f"[INFO] index built: {index.ntotal} vectors")

    # ensure output dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # save index
    faiss.write_index(index, str(args.output_dir / "text_index.faiss"))

    # save ids
    with (args.output_dir / "ids.txt").open("w", encoding="utf-8") as f:
        for rid in all_ids:
            f.write(rid + "\n")

    # save meta
    with (args.output_dir / "meta.jsonl").open("w", encoding="utf-8") as f:
        for rec in all_metas:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[DONE] saved index + ids + meta under {args.output_dir}")


if __name__ == "__main__":
    main()
