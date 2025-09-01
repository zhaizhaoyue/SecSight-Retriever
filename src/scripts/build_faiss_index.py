#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import json
import argparse
from typing import Dict, Any, List, Iterable
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ---------- helpers ----------
def should_include(path: Path, names: List[str]) -> bool:
    """只接受指定文件名（大小写不敏感）"""
    n = path.name.lower()
    return any(n == x.lower() for x in names)

def iter_jsonl_files(root: Path, include_names: List[str]) -> Iterable[Path]:
    """递归遍历 root 下目标 jsonl 文件"""
    for p in root.rglob("*.jsonl"):
        if should_include(p, include_names):
            yield p

def stream_chunks(path: Path):
    """逐行读取 JSONL，只要有非空 text 字段"""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            t = (obj.get("text") or "").strip()
            if t:
                yield obj

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk_dir", type=str, required=True, help="chunked 文件夹路径")
    ap.add_argument("--out", type=str, required=True, help="FAISS 索引输出目录")
    ap.add_argument("--model", type=str, default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch", type=int, default=128)
    # 允许自定义要包含的文件名，默认只处理 text 和 labels_best
    ap.add_argument(
        "--include",
        type=str,
        default="text_chunks.jsonl,labels_best_chunks.jsonl",
        help="以逗号分隔的文件名白名单（默认：text_chunks.jsonl,labels_best_chunks.jsonl）",
    )
    args = ap.parse_args()

    include_names = [s.strip() for s in args.include.split(",") if s.strip()]
    chunk_dir = Path(args.chunk_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 加载 embedding 模型
    enc = SentenceTransformer(args.model, device=args.device)
    dim = enc.get_sentence_embedding_dimension()

    # 2) 建 HNSW 索引
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efSearch = 128

    metas: List[Dict[str, Any]] = []
    buf_text, buf_meta = [], []
    total = 0

    for file in iter_jsonl_files(chunk_dir, include_names):
        print(f"[scan] {file}")
        for obj in stream_chunks(file):
            text = obj["text"]
            meta = {k: v for k, v in obj.items() if k != "text"}
            meta["text"] = text[:1000]           # 存前1000字符，够 BM25 & 重排预览
            meta["text_preview"] = text[:300]    # 可选：更短摘要供 UI
            meta["page"] = obj.get("page")
            buf_text.append(text)
            buf_meta.append(meta)

            if len(buf_text) >= args.batch:
                vecs = enc.encode(
                    buf_text,
                    batch_size=args.batch,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).astype("float32")
                index.add(vecs)
                metas.extend(buf_meta)
                total += len(buf_text)
                print(f"[add] total={total}")
                buf_text, buf_meta = [], []

    # flush
    if buf_text:
        vecs = enc.encode(
            buf_text,
            batch_size=args.batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        index.add(vecs)
        metas.extend(buf_meta)
        total += len(buf_text)

    # 3) 保存索引与元数据
    faiss.write_index(index, str(out_dir / "hnsw.index"))
    with (out_dir / "meta.jsonl").open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"[OK] indexed {total} chunks | dim={dim} | saved to {out_dir}")
    print(f"[INFO] included files: {include_names}")

if __name__ == "__main__":
    main()





'''
python src/scripts/build_faiss_index.py `
  --chunk_dir data/chunked `
  --out data/index/faiss_bge_base_en `
  --model BAAI/bge-base-en-v1.5 `
  --device cuda `
  --batch 128
'''