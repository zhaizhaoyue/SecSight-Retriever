# src/rag/retriever_dense.py
'''


python -m src.rag.retriever.dense `
  --q "What did Microsoft highlight as key growth drivers in 2023?" `
  --ticker MSFT `
  --form 10-K `
  --year 2023 `
  --topk 8 `
  --normalize `
  --content-dir "data/chunked"

  '''
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def load_metas(meta_path: Path) -> List[Dict[str, Any]]:
    metas: List[Dict[str, Any]] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            # 拍平内层 meta 到顶层（你的结构是外层 id/title + 内层 meta{...}）
            if isinstance(raw.get("meta"), dict):
                flat = dict(raw)
                inner = flat.pop("meta") or {}
                for k, v in inner.items():
                    flat.setdefault(k, v)
            else:
                flat = raw
            flat.setdefault("chunk_id", flat.get("chunk_id") or flat.get("id"))
            flat.setdefault("file_type", flat.get("file_type") or "text")
            flat.setdefault("title", flat.get("title") or "")
            metas.append(flat)
    return metas

def to_score(metric_type: int, dist: float) -> float:
    if metric_type == faiss.METRIC_INNER_PRODUCT:
        return float(dist)
    return 1.0 / (1.0 + float(dist))

def _iter_jsonl_files(base: Path) -> List[Path]:
    if base.is_file():
        return [base]
    files: List[Path] = []
    for p in base.rglob("*.jsonl"):
        files.append(p)
    return files

def fetch_contents(content_path: Optional[Path], chunk_ids: Set[str]) -> Dict[str, str]:
    """
    轻量取正文：扫描单文件或目录下所有 *.jsonl，直到把需要的 chunk_id 都找齐或文件读完。
    行格式允许：
      {"chunk_id": "...", "content": "..."} 或 {"id": "...", "text": "..."} 等。
    """
    if not content_path:
        return {}
    found: Dict[str, str] = {}
    files = _iter_jsonl_files(content_path)
    targets = set([cid for cid in chunk_ids if cid])
    if not targets or not files:
        return {}

    for fp in files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    cid = obj.get("chunk_id") or obj.get("id") or obj.get("chunkId")
                    if cid in targets:
                        txt = obj.get("content") or obj.get("text") or obj.get("raw_text") or ""
                        if txt:
                            found[cid] = txt
                            targets.remove(cid)
                            if not targets:  # 全找齐就提前停止
                                return found
        except Exception:
            continue
    return found

def main():
    ap = argparse.ArgumentParser(description="Pure dense (vector-only) retriever with optional content lookup for snippets")
    ap.add_argument("--index-dir", default="data/index", help="folder with text_index.faiss + meta.jsonl")
    ap.add_argument("--faiss", default=None, help="override faiss path")
    ap.add_argument("--meta", default=None, help="override meta path")
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--normalize", action="store_true",
                    help="normalize query embeddings (use if index built with normalized vectors + inner-product)")
    ap.add_argument("--q", "--query", dest="query", required=True)
    ap.add_argument("--topk", type=int, default=8)
    # 过滤器（只用 meta 字段）
    ap.add_argument("--ticker")
    ap.add_argument("--year", type=int)
    ap.add_argument("--form")
    # 新增：正文来源（2选1，或都不传=不取正文）
    ap.add_argument("--content-path", default=None, help="path to a JSONL containing {chunk_id|id, content|text}")
    ap.add_argument("--content-dir", default=None, help="directory to scan recursively for *.jsonl with chunk contents")
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    faiss_path = Path(args.faiss) if args.faiss else (index_dir / "text_index.faiss")
    meta_path  = Path(args.meta)  if args.meta  else (index_dir / "meta.jsonl")

    if not faiss_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing files. faiss={faiss_path} meta={meta_path}")

    # 1) 加载 FAISS & meta
    index = faiss.read_index(str(faiss_path))
    metas = load_metas(meta_path)
    if index.ntotal != len(metas):
        raise ValueError(f"index.ntotal={index.ntotal} != meta lines={len(metas)} (one-to-one required)")
    metric_type = getattr(index, "metric_type", faiss.METRIC_INNER_PRODUCT)

    # --- 新增: 如果是 IDMap，建立 label -> meta 的映射 ---
    def try_get_labels_from_idmap(index):
        try:
            arr = faiss.vector_to_array(getattr(index, "id_map"))
            return arr.astype(np.int64)
        except Exception:
            return None

    labels = try_get_labels_from_idmap(index)
    label2meta = None
    if labels is not None:
        if len(labels) != len(metas):
            raise ValueError(f"idmap labels ({len(labels)}) != metas ({len(metas)})")
        label2meta = {int(lbl): metas[i] for i, lbl in enumerate(labels)}

    # 2) Encoder（仅用于 query）
    try:
        enc = SentenceTransformer(args.model, device=args.device)
    except Exception:
        enc = SentenceTransformer(args.model, device="cpu")

    # 3) 编码查询
    qvec = enc.encode([args.query], normalize_embeddings=args.normalize, show_progress_bar=False)
    qvec = qvec.astype("float32")

    # 4) 向量检索
    k_search = max(args.topk * 100, 2000)
    D, I = index.search(qvec, k_search)

    # 5) 过滤（基于 meta 字段）
    tick = args.ticker.upper() if args.ticker else None
    form = args.form.upper() if args.form else None
    year = args.year
    results: List[Dict[str, Any]] = []

    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue

        if label2meta is not None:
            # IDMap 模式: idx 是标签 ID
            m = label2meta.get(int(idx))
            if m is None:
                continue
        else:
            # 普通索引: idx 是行号
            if not (0 <= idx < len(metas)):
                continue  # 防御性检查
            m = metas[int(idx)]

        if tick and str(m.get("ticker","")).upper() != tick:
            continue
        if form and str(m.get("form","")).upper() != form:
            continue
        if year is not None:
            try:
                if int(m.get("fy")) != int(year):
                    continue
            except Exception:
                continue
        results.append({
            "score": to_score(metric_type, float(dist)),
            "meta": m,
            "faiss_id": int(idx),
        })
        if len(results) >= args.topk:
            break

    if not results:
        print("[INFO] No hits after filters. Try removing filters or --topk larger.")
        return

    # 6) （可选）按 chunk_id 批量取正文，做 snippet
    content_root: Optional[Path] = None
    if args.content_path:
        content_root = Path(args.content_path)
    elif args.content_dir:
        content_root = Path(args.content_dir)

    contents: Dict[str, str] = {}
    if content_root:
        ids_needed = { r["meta"].get("chunk_id") for r in results if r["meta"].get("chunk_id") }
        contents = fetch_contents(content_root, ids_needed)

    # 7) 打印
    print(f"Query: {args.query}")
    print("="*80)
    for i, r in enumerate(results, 1):
        m = r["meta"]
        cid = m.get("chunk_id")
        snippet = contents.get(cid) or m.get("title") or ""
        print(f"[{i:02d}] score={r['score']:.4f} | {m.get('ticker')} {m.get('fy')} {m.get('form')} "
              f"| chunk={m.get('chunk_index')} | id={cid}")
        print("     ", snippet[:240].replace("\n"," "))
        print("-"*80)

if __name__ == "__main__":
    main()


