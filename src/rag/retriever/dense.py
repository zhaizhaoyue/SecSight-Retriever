# src/rag/retriever_dense.py
'''
python -m src.rag.retriever.dense `
  --q "How did foreign exchange rates impact Tesla’s financial results in 2023??" `
  --ticker TSLA --form 10-K --year 2023 `
  --topk 8 `
  --content-dir data/chunked
'''
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .utils_faiss import (
    load_index_and_maps,   # 读取 faiss + idmap.jsonl + meta.jsonl
    embed_query,           # 归一化编码查询
    chunk_no_from_id,      # 从 rid 解析 chunk-N
)

# ====== 常量 & 小工具 ======
RID_RE = re.compile(r'^\d{10}-\d{2}-\d{6}::text::chunk-\d+$')

def _iter_jsonl_files(base: Path) -> List[Path]:
    if base.is_file():
        return [base]
    files: List[Path] = []
    for p in base.rglob("*.jsonl"):
        files.append(p)
    # 稳定顺序，避免文件系统顺序漂移
    return sorted(files, key=lambda x: (x.parent.as_posix(), x.name))

def _get(meta_rec: dict, key: str, default=None):
    """先看顶层，再看 meta 内层，适配两种写法。"""
    if key in meta_rec:
        return meta_rec.get(key, default)
    inner = meta_rec.get("meta") or {}
    return inner.get(key, default)

def fetch_contents_strict(content_path: Optional[Path], rids: Set[str]) -> Dict[str, str]:
    if not content_path or not rids:
        return {}
    wanted = set(rids)
    out: Dict[str, str] = {}
    for fp in _iter_jsonl_files(content_path):
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    key = obj.get("id") or obj.get("chunk_id") or obj.get("chunkId")
                    if not key or not RID_RE.match(key):
                        # 严格模式：跳过短键，避免跨 accno/FY 串读
                        continue
                    if key in wanted:
                        txt = obj.get("content") or obj.get("text") or obj.get("raw_text") or ""
                        if txt:
                            out[key] = txt
                            wanted.remove(key)
                            if not wanted:
                                return out
        except Exception:
            continue
    return out

# ====== 主程序 ======
def main():
    ap = argparse.ArgumentParser(description="Pure dense retriever (explicit id; no position misalignment)")
    ap.add_argument("--index-dir", default="data/index", help="dir with text_index.faiss + idmap.jsonl + meta.jsonl")
    ap.add_argument("--faiss", default=None, help="override faiss path")
    ap.add_argument("--meta",  default=None, help="override meta path")
    ap.add_argument("--idmap", default=None, help="override idmap path")
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--q", "--query", dest="query", required=True)
    ap.add_argument("--topk", type=int, default=8)

    # 过滤条件（基于 meta 字段）
    ap.add_argument("--ticker")
    ap.add_argument("--year", type=int)
    ap.add_argument("--form")

    # 严格校验（默认开启；可用 --no-strict-x 关闭）
    ap.add_argument("--strict-year",  dest="strict_year",  action="store_true",  default=True)
    ap.add_argument("--no-strict-year",  dest="strict_year",  action="store_false")
    ap.add_argument("--strict-ticker",dest="strict_ticker",action="store_true",  default=True)
    ap.add_argument("--no-strict-ticker",dest="strict_ticker",action="store_false")
    ap.add_argument("--strict-form",  dest="strict_form",  action="store_true",  default=True)
    ap.add_argument("--no-strict-form",  dest="strict_form",  action="store_false")

    # 正文来源（2选1，或都不传=不取正文）
    ap.add_argument("--content-path", default=None, help="JSONL with full id + {content|text}")
    ap.add_argument("--content-dir",  default=None, help="folder to scan recursively for *.jsonl")
    # 展示控制
    ap.add_argument("--show-heading", action="store_true", help="print heading when available")
    ap.add_argument("--show-content", action="store_true", help="print content snippet if found")
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    faiss_path = Path(args.faiss) if args.faiss else (index_dir / "text_index.faiss")
    meta_path  = Path(args.meta)  if args.meta  else (index_dir / "meta.jsonl")
    idmap_path = Path(args.idmap) if args.idmap else (index_dir / "idmap.jsonl")

    # 一次性检查三件套
    missing = [str(p) for p in (faiss_path, idmap_path, meta_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing)}")

    # 1) 加载 FAISS + id 映射 + meta 字典
    index, id64_to_id, id_to_meta = load_index_and_maps(faiss_path, idmap_path, meta_path)
    metric_type = getattr(index, "metric_type", faiss.METRIC_INNER_PRODUCT)

    # 2) Encoder（仅用于 query）
    try:
        enc = SentenceTransformer(args.model, device=args.device)
    except Exception:
        enc = SentenceTransformer(args.model, device="cpu")

    # 3) 编码查询并检索（I 返回的是 id64）
    qv = embed_query(enc, args.query)                      # (1, dim)
    k_search = max(args.topk * 100, 2000)                  # 留足余量，后面做过滤
    D, I = index.search(qv, k_search)

    # 4) 过滤（基于 meta.jsonl；全程以 rid 为主键）
    want_ticker = args.ticker.upper() if args.ticker else None
    want_form   = args.form.upper()   if args.form   else None
    want_year   = int(args.year)      if args.year   is not None else None

    results: List[Dict[str, Any]] = []
    for dist, id64 in zip(D[0], I[0]):
        if int(id64) == -1:
            continue
        rid = id64_to_id.get(int(id64))
        if not rid:
            continue
        meta = id_to_meta.get(rid, {})

        mticker = str(_get(meta, "ticker", "")).upper()
        mform   = str(_get(meta, "form", "")).upper()
        mfy     = _get(meta, "fy", None)

        if want_ticker and mticker != want_ticker:
            continue
        if want_form and mform != want_form:
            continue
        if want_year is not None:
            try:
                if int(mfy) != want_year:
                    continue
            except Exception:
                continue

        # 评分（IP 时 dist 本身就是相似度；如为 L2 则转为相似度）
        score = float(dist) if metric_type == faiss.METRIC_INNER_PRODUCT else 1.0 / (1.0 + float(dist))
        results.append({"rid": rid, "score": score})

        if len(results) >= args.topk:
            break

    if not results:
        print("[INFO] No hits after filters. Try removing filters or increase --topk.")
        return

    # 5) 取正文（可选），严格按完整 id
    content_root: Optional[Path] = None
    if args.content_path:
        content_root = Path(args.content_path)
    elif args.content_dir:
        content_root = Path(args.content_dir)

    contents: Dict[str, str] = {}
    if content_root:
        contents = fetch_contents_strict(content_root, {r["rid"] for r in results})

    # 6) 打印（用 rid 解析真实 chunk 号，永不再错位）
    print(f"Query: {args.query}")
    print("=" * 80)
    for i, r in enumerate(results, 1):
        rid  = r["rid"]
        meta = id_to_meta.get(rid, {})

        ticker = _get(meta, "ticker", "NA")
        fy     = _get(meta, "fy", "NA")
        form   = _get(meta, "form", "NA")
        chunk_no = chunk_no_from_id(rid)

        # —— 严格护栏（展示前再次校验）——
        err = []
        if args.strict_ticker and args.ticker and str(ticker).upper() != want_ticker:
            err.append("ticker")
        if args.strict_form and args.form and str(form).upper() != want_form:
            err.append("form")
        if args.strict_year and (args.year is not None):
            try:
                if int(fy) != want_year:
                    err.append("year")
            except Exception:
                err.append("year")
        if err:
            print(f"[WARN] skip at print (mismatch {','.join(err)}): id={rid}", file=sys.stderr)
            continue

        heading = meta.get("heading") or ""
        snippet = contents.get(rid) or meta.get("snippet") or meta.get("title") or ""
        snippet = " ".join((snippet or "").split())

        print(f"[{i:02d}] score={r['score']:.4f} | {ticker} {fy} {form} | chunk={chunk_no} | id={rid}")
        if args.show_heading and heading:
            print("     heading:", heading)
        if args.show_content:
            if rid in contents:
                print("     content:", snippet[:400])
            else:
                print("     content: [WARN] not found in content source; showing meta snippet/title:")
                print("              ", snippet[:400])
        else:
            # 默认至少给点上下文
            if heading:
                print("     heading:", heading)
            if snippet:
                print("     content:", snippet[:400])
        print("-" * 80)


if __name__ == "__main__":
    main()
