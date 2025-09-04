#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan data/processed/**/calc*.jsonl → build data/mappings/calc.yaml

输出结构（示例）:
rollups:
  us-gaap:Liabilities:
    children:
      - { concept: us-gaap:LiabilitiesCurrent, weight: 1.0, order: 1.0, linkrole: ... }
      - { concept: us-gaap:LiabilitiesNoncurrent, weight: 1.0, order: 2.0, linkrole: ... }
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any
import yaml

def add_child(rollups: Dict[str, Dict[str, Any]],
              parent: str, child: str,
              weight: float|None, order: float|None, linkrole: str|None):
    if not parent or not child:
        return
    entry = rollups.setdefault(parent, {"children": []})
    # 去重：同一 child+linkrole+weight+order 只保留一条
    key = (child, weight, order, linkrole)
    existing = {(c.get("concept"), c.get("weight"), c.get("order"), c.get("linkrole"))
                for c in entry["children"]}
    if key not in existing:
        entry["children"].append({
            "concept": child,
            "weight": float(weight) if weight is not None else None,
            "order": float(order) if order is not None else None,
            "linkrole": linkrole
        })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=Path("data/processed"),
                    help="递归查找 calc*.jsonl")
    ap.add_argument("--pattern", default="calc*.jsonl",
                    help="匹配文件名（默认：calc*.jsonl）")
    ap.add_argument("--output", type=Path, default=Path("data/mappings/calc.yaml"),
                    help="输出 YAML 路径")
    args = ap.parse_args()

    files = list(args.input_root.rglob(args.pattern))
    print(f"[INFO] found {len(files)} calc jsonl files under {args.input_root}")

    rollups: Dict[str, Dict[str, Any]] = {}

    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for ln_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception as e:
                    print(f"[WARN] {fp}:{ln_no} invalid json: {e}")
                    continue

                # 支持两种字段命名：parent/child 或 from/to
                parent = rec.get("parent_concept") or rec.get("from_concept")
                child  = rec.get("child_concept")  or rec.get("to_concept")
                # 计算关系里通常有 weight/order/linkrole
                weight = rec.get("weight")
                order  = rec.get("order")
                linkrole = rec.get("linkrole")

                add_child(rollups, parent, child, weight, order, linkrole)

    # 排序（可选）
    for parent, obj in rollups.items():
        obj["children"].sort(key=lambda x: (
            (x.get("order") is None, x.get("order")),  # 有 order 的优先
            (x.get("concept") or "")
        ))

    # 写 YAML
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fo:
        yaml.safe_dump({"rollups": rollups}, fo, allow_unicode=True, sort_keys=True)

    print(f"[DONE] wrote {args.output} (parents={len(rollups)})")

if __name__ == "__main__":
    main()
