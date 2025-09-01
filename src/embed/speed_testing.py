#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import json, time, argparse, itertools
from sentence_transformers import SentenceTransformer

def iter_jsonl(root: Path):
    for p in root.rglob("*.jsonl"):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                t = (obj.get("text") or "").strip()
                if t:
                    yield t

def take_n(it, n):
    return list(itertools.islice(it, n))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk_dir", required=True)
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=256, help="预热样本数")
    ap.add_argument("--measure", type=int, default=2048, help="正式测量样本数")
    args = ap.parse_args()

    print(f"[info] loading model {args.model} on {args.device} ...")
    enc = SentenceTransformer(args.model, device=args.device)

    src = iter_jsonl(Path(args.chunk_dir))
    warm = take_n(src, args.warmup)
    meas = take_n(src, args.measure)

    if len(meas) == 0:
        print("[warn] 没有可用文本（缺少 text 字段）"); return

    # 预热（不计时）
    if warm:
        _ = enc.encode(warm, batch_size=args.batch, normalize_embeddings=True, show_progress_bar=False)

    # 正式计时
    t0 = time.time()
    _ = enc.encode(meas, batch_size=args.batch, normalize_embeddings=True, show_progress_bar=False)
    t1 = time.time()

    sec = t1 - t0
    qps = len(meas) / sec
    print(f"[result] 批量={args.batch} | 样本数={len(meas)} | 用时={sec:.2f}s | 吞吐≈{qps:.1f} chunks/sec")
    print("接下来：总时长 ≈ 总chunks数 ÷ 吞吐")

if __name__ == "__main__":
    main()
