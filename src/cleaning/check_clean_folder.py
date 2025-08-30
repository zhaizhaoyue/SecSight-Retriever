#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

# 设定 clean 根目录
CLEAN_ROOT = Path("data/clean")

# 必须要有的文件清单
REQUIRED_FILES = [
    "calculation_edges.jsonl",
    "calculation_edges.parquet",
    "definition_arcs.jsonl",
    "definition_arcs.parquet",
    "fact.jsonl",
    "fact.parquet",
    "labels_best.jsonl",
    "labels_best.parquet",
    "labels_wide.jsonl",
    "labels_wide.parquet",
    "labels.jsonl",
    "labels.parquet",
    "text_corpus.jsonl",
    "text_corpus.parquet",
]

def check_clean_files(root: Path, required: list[str]) -> None:
    missing = []
    extra = []
    # 遍历 root 下的文件
    existing = {p.name for p in root.rglob("*") if p.is_file()}
    for f in required:
        if f not in existing:
            missing.append(f)
    for f in existing:
        if f not in required:
            extra.append(f)

    print("=== 检查结果 ===")
    if missing:
        print("[缺失文件]")
        for f in missing:
            print(" -", f)
    else:
        print("没有缺失")

    if extra:
        print("\n[额外文件]")
        for f in extra:
            print(" -", f)
    else:
        print("\n没有额外文件")

if __name__ == "__main__":
    check_clean_files(CLEAN_ROOT, REQUIRED_FILES)
