#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import json
import argparse
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def iter_jsonl_files(root: Path):
    """递归遍历 root 下所有 .jsonl 文件"""
    for p in root.rglob("*.jsonl"):
        yield p

def stream_chunks(path: Path):
    """逐行读取 JSONL，要求必须有 text 字段"""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "text" in obj and obj["text"].strip():
                yield obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk_dir", type=str, required=True, help="chunked 文件夹路径")
    ap.add_argument("--out", type=str, required=True, help="FAISS 索引输出目录")
    ap.add_argument("--model", type=str, default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    chunk_dir = Path(args.chunk_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载 embedding 模型
    enc = SentenceTransformer(args.model, device=args.device)
    dim = enc.get_sentence_embedding_dimension()

    # 建 HNSW 索引
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efSearch = 128

    metas: List[Dict[str, Any]] = []
    buf_text, buf_meta = [], []
    total = 0

    for file in iter_jsonl_files(chunk_dir):
        print(f"[scan] {file}")
        for obj in stream_chunks(file):
            text = obj["text"]
            meta = {k: v for k, v in obj.items() if k != "text"}
            buf_text.append(text)
            buf_meta.append(meta)

            if len(buf_text) >= args.batch:
                vecs = enc.encode(buf_text, batch_size=args.batch,
                                  normalize_embeddings=True,
                                  show_progress_bar=False)
                vecs = vecs.astype("float32")
                index.add(vecs)
                metas.extend(buf_meta)
                total += len(buf_text)
                print(f"[add] total={total}")
                buf_text, buf_meta = [], []

    # flush 最后一个批次
    if buf_text:
        vecs = enc.encode(buf_text, batch_size=args.batch,
                          normalize_embeddings=True,
                          show_progress_bar=False)
        vecs = vecs.astype("float32")
        index.add(vecs)
        metas.extend(buf_meta)
        total += len(buf_text)

    # 保存索引和元数据
    faiss.write_index(index, str(out_dir / "hnsw.index"))
    with (out_dir / "meta.jsonl").open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"[OK] indexed {total} chunks | dim={dim} | saved to {out_dir}")

if __name__ == "__main__":
    main()


'''
python scripts/build_faiss_from_dir.py `
  --chunk_dir data/chunked `
  --out data/index/faiss_bge_base_en `
  --model BAAI/bge-base-en-v1.5 `
  --device cuda `
  --batch 128

  '''