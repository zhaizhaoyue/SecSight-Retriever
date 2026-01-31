from __future__ import annotations
from typing import List, Dict, Any

def pretty_print(hits: List[Dict[str, Any]], k: int = 5):
    for i, h in enumerate(hits[:k], 1):
        m = h.get("meta", {})
        print(f"{i:02d}. score={h.get('score', h.get('score_dense', 0)):.4f} fy={m.get('fy')} fq={m.get('fq')} form={m.get('form')} accno={m.get('accno')}\n     {m.get('source_path')}\n     {h.get('snippet')}\n{'-'*80}")
