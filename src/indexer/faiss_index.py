# src/indexer/faiss_index.py
import faiss, numpy as np, json
from pathlib import Path
from typing import List, Dict, Any, Optional

class FaissHNSWIndex:
    def __init__(self, dim: int, M: int = 32, ef_search: int = 128):
        self.index = faiss.IndexHNSWFlat(dim, M)
        self.index.hnsw.efSearch = ef_search
        self._ids: List[int] = []             # 顺序ID → metadata 对应
        self._meta: List[Dict[str, Any]] = [] # 与向量一一对应

    def add(self, vecs: np.ndarray, metas: List[Dict[str, Any]]):
        assert vecs.shape[0] == len(metas)
        start = len(self._ids)
        self.index.add(vecs)
        self._ids.extend(range(start, start + vecs.shape[0]))
        self._meta.extend(metas)

    def search(self, q: np.ndarray, top_k: int = 10):
        D, I = self.index.search(q, top_k)  # q: shape [1, dim] 或 [B, dim]
        results = []
        for row_i, idxs in enumerate(I):
            items = []
            for j, idx in enumerate(idxs):
                if idx < 0:  # 空结果保护
                    continue
                items.append({
                    "id": int(idx),
                    "score": float(D[row_i][j]),
                    "meta": self._meta[idx]
                })
            results.append(items)
        return results

    # ---- 持久化（索引 + 元数据）----
    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out_dir / "hnsw.index"))
        with (out_dir / "meta.jsonl").open("w", encoding="utf-8") as f:
            for m in self._meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, dim: int, dir_: Path):
        obj = cls(dim=dim)
        obj.index = faiss.read_index(str(dir_ / "hnsw.index"))
        obj._meta = []
        with (dir_ / "meta.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                obj._meta.append(json.loads(line))
        obj._ids = list(range(len(obj._meta)))
        return obj
