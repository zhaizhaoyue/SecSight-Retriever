# audit_meta_vs_source.py
from pathlib import Path
import json, re, sys

CHUNK_RE = re.compile(r'::text::chunk-(\d+)$')

def scan_source_ids(chunked_root: Path) -> set[str]:
    ids = set()
    for p in sorted(chunked_root.rglob("text_chunks.jsonl")):
        with p.open(encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                rid = j.get("id") or j.get("meta",{}).get("chunk_id")
                if rid:
                    ids.add(rid)
    return ids

def audit(index_dir: Path, chunked_root: Path):
    meta_path = index_dir / "meta.jsonl"
    if not meta_path.exists():
        print("missing meta.jsonl", file=sys.stderr); return
    source_ids = scan_source_ids(chunked_root)
    bad_chunknum = 0
    missing_in_source = 0
    total = 0
    for i, line in enumerate(meta_path.open(encoding="utf-8"), 1):
        total += 1
        j = json.loads(line)
        rid = j.get("id")
        if not rid: 
            print(f"[{i}] no id"); continue
        m = CHUNK_RE.search(rid)
        real = int(m.group(1)) if m else None
        stored = j.get("chunk_index") or j.get("meta",{}).get("chunk_index")
        if stored is not None and real is not None and int(stored) != real:
            bad_chunknum += 1
            print(f"[chunk-mismatch] line {i} id={rid} stored={stored} real={real}")
        if rid not in source_ids:
            missing_in_source += 1
            print(f"[missing-source] line {i} id={rid}")
    print(f"TOTAL={total} mismatched_chunknum={bad_chunknum} missing_in_source={missing_in_source}")

if __name__ == "__main__":
    index_dir = Path("data/index")
    chunked_root = Path("data/chunked")
    audit(index_dir, chunked_root)
