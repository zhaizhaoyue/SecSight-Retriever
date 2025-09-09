#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embed all text_chunks.jsonl under data/chunked/
→ build one FAISS index in data/index/text_index.faiss (with explicit IDs)
→ save idmap + meta (+ build_info) for lookup (order-agnostic, no more misalignment)

主要改动：
- [NEW] 稳定遍历 + 去重冲突决策（按 mtime/路径关键词 + 文本哈希），杜绝同 rid 不同内容的“串档”
- [NEW] 严格使用完整 rid（accno::text::chunk-N）作为唯一键；统一从 rid 解析 chunk_index
- [NEW] 将 text 的 sha256 和来源路径写入 idmap.jsonl & meta.jsonl，检索端可校验一致性
- [NEW] 输出 build_info.json 记录本次构建所用源与模型，便于后验审计
"""

from __future__ import annotations
import argparse, json, sys, hashlib, os, time
from pathlib import Path
from typing import List, Dict, Iterable, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re

# ========= 正则与小工具 =========
# [NEW] 严格 rid 格式（避免用短 chunk-XXX 作为主键）
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

# ========= 读 JSONL =========
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

# ========= 文本构建 =========
def build_text(rec: Dict, use_title=True) -> str:
    title = (rec.get("title") or "").strip()
    content = (rec.get("content") or "").strip()
    if use_title and title:
        return f"{title}\n\n{content}"
    return content

def make_snippet(text: str, max_chars: int = 420) -> str:
    s = " ".join((text or "").split())
    return s[:max_chars] + ("..." if len(s) > max_chars else "")

# ========= 向量编码 =========
def encode_texts(model: SentenceTransformer, texts: List[str], batch_size=64) -> np.ndarray:
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

# ========= 冲突优先级（路径关键词） =========
# [NEW] 依据路径关键词/mtime的冲突裁决：谁作为“权威版本”
def path_priority_score(path: str, prefer_keywords: Tuple[str, ...]) -> int:
    p = path.lower()
    score = 0
    for kw in prefer_keywords:
        if kw and kw.lower() in p:
            score += 1
    return score

# ========= 主流程 =========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=Path("data/chunked"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/index"))
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--use-title", action="store_true", help="prepend title to content")
    # [NEW] 冲突裁决策略参数：路径关键词优先（逗号分隔，越多越优先）
    ap.add_argument("--prefer-keywords", default="latest,final,clean", help="comma separated path keywords for conflict resolving")
    # [NEW] 遇到 rid 不是完整格式时的动作（warn 或 strict）
    ap.add_argument("--strict-rid", action="store_true", help="require full rid format accno::text::chunk-N")
    args = ap.parse_args()

    prefer_keywords = tuple([s.strip() for s in args.prefer_keywords.split(",") if s.strip()])

    # load encoder
    model = SentenceTransformer(args.model)
    dim = model.get_sentence_embedding_dimension()

    # stable traversal order
    # [CHANGED] 强制排序，避免文件系统无序导致的顺序漂移
    files = sorted(args.input_root.rglob("text_chunks.jsonl"), key=lambda x: (x.parent.as_posix(), x.name))
    print(f"[INFO] found {len(files)} files (sorted)")

    # [CHANGED] 不再用 set(seen) 简单去重；改为 rid->权威记录 的 dict
    by_id: Dict[str, Dict] = {}   # rid -> {text, meta, heading, title, snippet, src_path, src_mtime, text_hash}

    # 收集并做冲突决策
    for f in files:
        mtime = file_mtime(f)
        for rec in iter_jsonl(f):
            # [CHANGED] 严格取 rid：优先 rec["id"]; 次选 meta.chunk_id；并可校验格式
            rid = rec.get("id") or (rec.get("meta", {}) or {}).get("chunk_id")
            if not rid:
                continue
            if args.strict_rid and not RID_RE.match(str(rid)):
                print(f"[WARN] skip non-full rid: {rid} (file={f})", file=sys.stderr)
                continue

            text = build_text(rec, use_title=args.use_title)
            if not text:
                continue

            # [NEW] 计算文本哈希 & 记录来源
            h = sha256_text(text)
            entry = {
                "rid": rid,
                "title": rec.get("title"),
                "heading": rec.get("heading"),
                "snippet": make_snippet(text, 420),
                "meta": rec.get("meta"),
                "text": text,               # [NEW] 保存用于编码的一致文本
                "text_hash": h,             # [NEW]
                "src_path": str(f),         # [NEW]
                "src_mtime": mtime,         # [NEW]
                "path_prio": path_priority_score(str(f), prefer_keywords),  # [NEW]
            }

            prev = by_id.get(rid)
            if prev is None:
                by_id[rid] = entry
            else:
                # [NEW] 同 rid 冲突：哈希相同 → 合并非空字段；哈希不同 → 路径关键词优先，其次 mtime 更新的胜出
                if prev["text_hash"] == h:
                    # 合并非空 heading/title（保留较“优”者）
                    if not prev.get("heading") and entry.get("heading"):
                        prev["heading"] = entry["heading"]
                    if not prev.get("title") and entry.get("title"):
                        prev["title"] = entry["title"]
                    # 路径优先分高者覆盖 src_path 信息（可选）
                    if entry["path_prio"] > prev["path_prio"]:
                        prev["src_path"] = entry["src_path"]
                        prev["src_mtime"] = entry["src_mtime"]
                        prev["path_prio"] = entry["path_prio"]
                else:
                    # 文本不同：先看路径优先级，再看 mtime
                    choose = prev
                    if entry["path_prio"] > prev["path_prio"]:
                        choose = entry
                    elif entry["path_prio"] == prev["path_prio"] and entry["src_mtime"] >= prev["src_mtime"]:
                        choose = entry
                    if choose is entry:
                        by_id[rid] = entry
                        print(f"[WARN] rid conflict (diff content): kept NEW {entry['src_path']} ; drop OLD {prev['src_path']} | id={rid}",
                              file=sys.stderr)
                    else:
                        # 保留 prev，提示被丢弃的 entry
                        print(f"[WARN] rid conflict (diff content): kept OLD {prev['src_path']} ; drop NEW {entry['src_path']} | id={rid}",
                              file=sys.stderr)

    # 展平用于编码/落盘
    all_ids: List[str] = []
    all_id64: List[int] = []
    all_texts: List[str] = []
    all_metas: List[Dict] = []

    for rid, item in by_id.items():
        # [NEW] 统一从 rid 解析 chunk_index（不再信任外部提供的 chunk_index）
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
            "chunk_index": chunk_index,     # [NEW]
            "text_hash": item.get("text_hash"),  # [NEW]
            "src_path": item.get("src_path"),    # [NEW]
            "src_mtime": item.get("src_mtime"),  # [NEW]
        }
        all_metas.append(meta_out)

    print(f"[INFO] total unique records: {len(all_ids)}")

    # encode all
    vecs = encode_texts(model, all_texts, batch_size=args.batch_size)
    assert vecs.shape[0] == len(all_ids)

    # build FAISS index with explicit ids
    base = faiss.IndexFlatIP(dim)  # IP with normalized embeddings == cosine
    index = faiss.IndexIDMap2(base)
    id_array = np.asarray(all_id64, dtype="int64")
    index.add_with_ids(vecs, id_array)
    print(f"[INFO] index built: {index.ntotal} vectors")

    # ensure output dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # save index
    faiss.write_index(index, str(args.output_dir / "text_index.faiss"))

    # save id map (id <-> id64 + source + hash)  [CHANGED]
    with (args.output_dir / "idmap.jsonl").open("w", encoding="utf-8") as f:
        for rid in all_ids:
            item = by_id[rid]
            f.write(json.dumps({
                "id": rid,
                "id64": stable_id(rid),
                "src_path": item["src_path"],   # [NEW]
                "text_hash": item["text_hash"], # [NEW]
            }, ensure_ascii=False) + "\n")

    # save meta (order-agnostic; use id to lookup)  [CHANGED: 增加 chunk_index/src/hash 字段]
    with (args.output_dir / "meta.jsonl").open("w", encoding="utf-8") as f:
        for rec in all_metas:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # [NEW] 保存 build_info.json 记录来源、模型与时间
    build_info = {
        "built_at_utc": now_iso(),
        "model": args.model,
        "input_root": str(args.input_root),
        "files_count": len(files),
        "prefer_keywords": prefer_keywords,
        "records": len(all_ids),
    }
    with (args.output_dir / "build_info.json").open("w", encoding="utf-8") as bf:
        json.dump(build_info, bf, ensure_ascii=False, indent=2)

    print(f"[DONE] saved index + idmap + meta (+ build_info) under {args.output_dir}")


if __name__ == "__main__":
    main()
