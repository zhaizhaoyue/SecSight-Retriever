#防止chunk index错位的工具脚本
# src/rag/retriever/utils_faiss.py
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, Tuple, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNK_RE = re.compile(r'::text::chunk-(\d+)$')

def load_index_and_maps(index_path: Path, idmap_path: Path, meta_path: Path):
    index = faiss.read_index(str(index_path))

    id64_to_id: Dict[int, str] = {}
    with idmap_path.open(encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            id64_to_id[int(j["id64"])] = j["id"]

    id_to_meta: Dict[str, dict] = {}
    with meta_path.open(encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            id_to_meta[j["id"]] = j

    return index, id64_to_id, id_to_meta

def embed_query(model: SentenceTransformer, q: str) -> np.ndarray:
    v = model.encode([q], normalize_embeddings=True)
    return np.asarray(v, dtype="float32")

def chunk_no_from_id(rid: str) -> str:
    m = CHUNK_RE.search(rid or "")
    return m.group(1) if m else "NA"

def make_row(rank, score_str, meta, rid):
    chunk_no = chunk_no_from_id(rid)
    title = meta.get("title", "N/A")
    heading = meta.get("heading") or ""
    snippet = meta.get("snippet") or ""
    out = []
    out.append("=" * 80)
    out.append(f"[{rank:02}] {score_str} | {title} | chunk={chunk_no} | id={rid}")
    if heading:
        out.append(f"     heading: {heading}")
    if snippet:
        out.append(f"     content: {snippet}")
    return "\n".join(out)

def show_results_dense(D: np.ndarray, I: np.ndarray, id64_to_id: Dict[int, str], id_to_meta: Dict[str, dict], topk: int = 8):
    # D/I: shape = (1, k)
    lines: List[str] = []
    for r, (score, id64) in enumerate(zip(D[0], I[0]), 1):
        rid = id64_to_id.get(int(id64))
        if rid is None:
            # 极端情况：索引命中无映射，跳过
            continue
        meta = id_to_meta.get(rid, {})
        score_str = f"dense={float(score):.6f}"  # 这里是纯 dense；hybrid 时可换 fused
        lines.append(make_row(r, score_str, meta, rid))
    print("\n".join(lines) if lines else "[WARN] no results")
