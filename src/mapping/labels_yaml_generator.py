#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan data/processed/**/labels.jsonl → build data/mappings/labels.yaml

- 对每个 concept 聚合各语言的 label_text，写成:
  concept:
    aliases_en:
      - "Net sales"
      - "Sales"
    aliases_zh:
      - "营业收入"
- 递归遍历所有 labels.jsonl
- 去重 / 去空白 / 排序


python src/mapping/labels_yaml_generator.py `
  --input-root data/processed `
  --output data/mappings/labels.yaml

"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Set
import yaml

def normalize_lang(lang: str | None) -> str:
    if not lang:
        return "en"
    return lang.split("-")[0].lower()

def clean_label(s: str | None) -> str:
    if not s:
        return ""
    # 去首尾空白 & 折叠多空格/换行
    return " ".join(s.split())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=Path("data/processed"),
                    help="根目录，递归查找 labels.jsonl")
    ap.add_argument("--pattern", default="labels.jsonl",
                    help="要匹配的文件名（默认 labels.jsonl）")
    ap.add_argument("--output", type=Path, default=Path("data/mappings/labels.yaml"),
                    help="输出 YAML 路径")
    args = ap.parse_args()

    concept_map: Dict[str, Dict[str, Set[str]]] = {}

    files = list(args.input_root.rglob(args.pattern))
    if not files:
        print(f"[WARN] no {args.pattern} found under {args.input_root}")
    else:
        print(f"[INFO] found {len(files)} files")

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

                concept = rec.get("concept")
                label_text = clean_label(rec.get("label_text"))
                lang = normalize_lang(rec.get("lang") or rec.get("lang_safe"))

                if not concept or not label_text:
                    continue

                lang_bucket = concept_map.setdefault(concept, {})
                lang_bucket.setdefault(lang, set()).add(label_text)

    # 转成 YAML 结构：concept -> aliases_<lang>: [ ... ]
    out: Dict[str, Dict[str, list]] = {}
    for concept, lang_dict in concept_map.items():
        entry: Dict[str, list] = {}
        for lang, labels in lang_dict.items():
            key = f"aliases_{lang}"
            # 稳定排序（不区分大小写）
            entry[key] = sorted(labels, key=lambda s: s.casefold())
        out[concept] = entry

    # 写出 YAML
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fo:
        yaml.safe_dump(out, fo, allow_unicode=True, sort_keys=True)

    print(f"[DONE] wrote {args.output} (concepts={len(out)})")

if __name__ == "__main__":
    main()
