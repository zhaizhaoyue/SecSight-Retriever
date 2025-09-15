


'''

  
python -m src.rag.retriever.hybrid `
  --query "What risks related to climate change did ExxonMobil highlight in its 2023 10-K?" `
  --index-dir data/index `
  --content-dir data/chunked `
  --model BAAI/bge-base-en-v1.5 `
  --ticker XOM --form "10-K" --year 2023 `
  --topk 8 `
  --bm25-topk 200 --dense-topk 200 --ce-candidates 256 `
  --w-bm25 1.0 --w-dense 2.0 `
  --ce-weight 0.3 `
  --show-chars 1000




'''
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
from typing import Optional, List, Dict, Any, Set

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.rag.retriever.bm25_text import BM25TextRetriever, BM25TextConfig
from src.rag.retriever.dense import DenseRetriever

# ---------------- Cross-Encoder reranker ----------------
class CrossEncoderReranker:
    def __init__(self, model_name: str, device: Optional[str] = None, max_length: int = 512, batch_size: int = 16):
        # Robust device selection with fallback
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)

    def score(self, query: str, docs: List[str]) -> List[float]:
        scores: List[float] = []
        for i in range(0, len(docs), self.batch_size):
            pairs = [(query, d if d else " ") for d in docs[i:i+self.batch_size]]
            enc = self.tokenizer(
                pairs, padding=True, truncation=True,
                return_tensors="pt", max_length=self.max_length
            )
            with torch.no_grad():
                outputs = self.model(**{k: v.to(self.device) for k, v in enc.items()})
                logits = outputs.logits
                if logits.shape[1] == 1:  # regression head
                    s = logits.squeeze(-1).detach().cpu().tolist()
                else:                      # classification head (use prob of class 1)
                    s = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
                scores.extend(s)
        return scores

# ---------------- content loader (全文抓取) ----------------
def _iter_jsonl_files(base: Path) -> List[Path]:
    if base.is_file():
        return [base]
    return [p for p in base.rglob("*.jsonl")]

def fetch_contents(content_root: Optional[Path], chunk_ids: Set[str]) -> Dict[str, str]:
    """
    从 content_root 下的 jsonl 文件中按 id/chunk_id/chunkId 抓取全文 content/text/raw_text/page_text/body
    """
    if not content_root or not chunk_ids:
        return {}
    files = _iter_jsonl_files(content_root)
    if not files:
        return {}
    found: Dict[str, str] = {}
    targets = set([cid for cid in chunk_ids if cid])
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
                        txt = (
                            obj.get("content") or obj.get("text")
                            or obj.get("raw_text") or obj.get("page_text")
                            or obj.get("body") or ""
                        )
                        if txt:
                            found[cid] = txt
                            targets.remove(cid)
                            if not targets:
                                return found
        except Exception:
            continue
    return found

# ---------------- Hybrid (RRF + 全局 CE 重排) ----------------
class HybridRetrieverRRF:
    def __init__(self, bm25_cfg: BM25TextConfig, dense: DenseRetriever,
                     reranker: Optional[CrossEncoderReranker], k: float = 60.0, w_bm25: float = 2.0, w_dense: float = 2.0,
                 ce_weight: float = 0.5):
        self.bm25 = BM25TextRetriever(bm25_cfg)
        self.dense = dense
        self.reranker = reranker
        self.k = float(k)
        self.w_bm25 = float(w_bm25)
        self.w_dense = float(w_dense)
        self.ce_weight = float(ce_weight)

    def _rrf(self, bm25_results, dense_results, ce_candidates: int) -> List[Dict[str, Any]]:
        # 名次表（从 1 开始）
        bm25_rank = {r["id"]: i+1 for i, r in enumerate(bm25_results)}
        dense_rank = {r["id"]: i+1 for i, r in enumerate(dense_results)}

        # 合并候选（保留更长的 snippet，合并 meta）
        pool: Dict[str, Dict[str, Any]] = {}
        def _upsert(results):
            for r in results:
                rid = r["id"]
                rec = pool.setdefault(rid, {"id": rid, "snippet": "", "meta": {}})
                s = (r.get("snippet") or "").strip()
                if len(s) > len(rec["snippet"]):
                    rec["snippet"] = s
                if isinstance(r.get("meta"), dict):
                    # 右侧优先，避免丢失已有字段
                    rec["meta"] = {**rec["meta"], **r["meta"]}

        _upsert(bm25_results)
        _upsert(dense_results)

        fused: List[Dict[str, Any]] = []
        for rid, rec in pool.items():
            rb = bm25_rank.get(rid)
            rd = dense_rank.get(rid)
            rrf = 0.0
            if rb is not None:
                rrf += self.w_bm25 / (self.k + rb)
            if rd is not None:
                rrf += self.w_dense / (self.k + rd)
            fused.append({
                "id": rid,
                "rrf_score": float(rrf),
                "rank_bm25": rb,
                "rank_dense": rd,
                "snippet": rec["snippet"],
                "meta": rec["meta"],
            })

        fused.sort(key=lambda x: x["rrf_score"], reverse=True)
        # 给 CE 留更大的候选池
        return fused[:max(1, int(ce_candidates))]

    def _apply_filters(self, results: List[Dict[str, Any]], *, ticker: Optional[str], form: Optional[str], year: Optional[int]) -> List[Dict[str, Any]]:
        if not results:
            return results
        if not (ticker or form or year):
            return results
        out = []
        for r in results:
            m = r.get("meta", {}) or {}
            if ticker and m.get("ticker") != ticker:
                continue
            if form and m.get("form") != form:
                continue
            if year is not None:
                fy = m.get("fy") or m.get("year")
                try:
                    if int(fy) != int(year):
                        continue
                except Exception:
                    continue
            out.append(r)
        return out

    def search(self, query: str, topk: int = 8, *,
            content_path: Optional[str] = None, content_dir: Optional[str] = None,
            bm25_topk: int = 200, dense_topk: int = 200, ce_candidates: int = 256,
            strict_filters: bool = True,
            **filters) -> List[Dict[str, Any]]:
        # 默认启用过滤
        pass_filters = filters if strict_filters else {}
        bm25_results = self.bm25.search(query, topk=bm25_topk, **pass_filters) or []
        dense_results = self.dense.search(query, topk=dense_topk, content_path=content_path,
                                        content_dir=content_dir, **pass_filters) or []
        # 检索后再做一次后置过滤，防止底层检索器不支持或元数据缺失
        if strict_filters:
            bm25_results = self._apply_filters(bm25_results, ticker=filters.get("ticker"), form=filters.get("form"), year=filters.get("year"))
            dense_results = self._apply_filters(dense_results, ticker=filters.get("ticker"), form=filters.get("form"), year=filters.get("year"))

        if not bm25_results and not dense_results:
            return []

        fused = self._rrf(bm25_results, dense_results, ce_candidates=ce_candidates)

        root: Optional[Path] = None
        if content_path:
            root = Path(content_path)
        elif content_dir:
            root = Path(content_dir)

        contents: Dict[str, str] = {}
        if root is not None:
            ids_needed = {rec["id"] for rec in fused}
            contents = fetch_contents(root, ids_needed)

        ce_docs = [contents.get(rec["id"]) or rec["snippet"] or " " for rec in fused]
        if (self.reranker is None) or (self.ce_weight <= 1e-6) or (ce_candidates <= 0):
            ce_scores = [0.0] * len(fused)
        else:
            ce_scores = self.reranker.score(query, ce_docs)
            # 归一化到 [0,1]，稳住与 RRF 的量纲
            lo, hi = min(ce_scores), max(ce_scores)
            if hi > lo:
                ce_scores = [(s - lo) / (hi - lo) for s in ce_scores]


        # 加权融合：将 RRF 和 CE 结果加权合并
        out: List[Dict[str, Any]] = []
        for rec, ce, doc in zip(fused, ce_scores, ce_docs):
            final_score = rec["rrf_score"] * (1 - self.ce_weight) + ce * self.ce_weight
            out.append({
                "id": rec["id"],
                "rrf_score": rec["rrf_score"],
                "ce_score": float(ce),
                "rerank_score": float(ce),   # 兼容旧字段
                "rank_bm25": rec["rank_bm25"],
                "rank_dense": rec["rank_dense"],
                "snippet": rec["snippet"],
                "content": doc,
                "meta": rec["meta"],
                "final_score": final_score,
            })

        out.sort(key=lambda x: x["final_score"], reverse=True)
        return out[:topk]

# ---------------- CLI ----------------
def _cli():
    ap = argparse.ArgumentParser(description="Hybrid retriever (RRF fusion + CE rerank) with content display.")
    ap.add_argument("--query", required=True)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--k", type=float, default=60.0, help="RRF parameter (default=60)")
    ap.add_argument("--content-path", default=None, help="file or folder of jsonl contents")
    ap.add_argument("--content-dir", default=None, help="folder of jsonl contents")
    ap.add_argument("--json-out", action="store_true", help="print JSON array instead of pretty text")
    ap.add_argument("--show-chars", type=int, default=800, help="preview length of content in pretty print")
    ap.add_argument("--w-bm25", type=float, default=1.0)
    ap.add_argument("--w-dense", type=float, default=2.0)
    ap.add_argument("--ce-weight", type=float, default=0.5, help="weight for CE rerank (default=0.5)")
    ap.add_argument("--bm25-topk", type=int, default=200, help="BM25候选条数（用于RRF融合前）")
    ap.add_argument("--dense-topk", type=int, default=200, help="Dense候选条数（用于RRF融合前）")
    ap.add_argument("--ce-candidates", type=int, default=256, help="送入CE的候选池大小")


    # filters
    ap.add_argument("--ticker"); ap.add_argument("--form"); ap.add_argument("--year", type=int)
    args = ap.parse_args()

    # init sub-retrievers
    bm25_cfg = BM25TextConfig(index_dir=args.index_dir)
    dense = DenseRetriever(index_dir=args.index_dir, model=args.model, device=args.device)

    reranker = None
    if (args.ce_weight is None) or (args.ce_weight > 1e-6 and args.ce_candidates > 0):
        reranker = CrossEncoderReranker(model_name=args.rerank_model, device=args.device)


    hybrid = HybridRetrieverRRF(bm25_cfg, dense, reranker, k=args.k, w_bm25=args.w_bm25, w_dense=args.w_dense, ce_weight=args.ce_weight)

    records = hybrid.search(
        query=args.query, topk=args.topk,
        content_path=args.content_path, content_dir=args.content_dir,
        bm25_topk=args.bm25_topk, dense_topk=args.dense_topk, ce_candidates=args.ce_candidates,
        ticker=args.ticker, form=args.form, year=args.year
    )


    if not records:
        print("[INFO] No hits after fusion.")
        return

    if args.json_out:
        print(json.dumps(records, ensure_ascii=False))
        return

    # Pretty print with content preview
    print(f"Query: {args.query}")
    print("=" * 80)
    for i, r in enumerate(records, 1):
        m = r.get("meta", {})
        rid = r["id"]
        title = (m.get("title") or m.get("heading") or m.get("section") or "")[:160]
        score_final = r.get("final_score", 0.0)
        score_ce    = r.get("ce_score", 0.0)
        score_rrf   = r.get("rrf_score", 0.0)
        print(f"[{i:02d}] final={score_final:.4f} | ce={score_ce:.4f} | rrf={score_rrf:.4f} "
            f"| {m.get('ticker')} {m.get('fy')} {m.get('form')} | id={rid}")

        if title:
            print(f"     title: {title}")
        # 内容预览
        preview = (r.get("content") or r.get("snippet") or "").replace("\n", " ")
        print("     content:", preview[:args.show_chars])
        print("-" * 80)

if __name__ == "__main__":
    _cli()
