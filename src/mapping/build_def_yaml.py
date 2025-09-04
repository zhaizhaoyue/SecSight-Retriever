#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan data/processed/**/def*.jsonl → build data/mappings/def.yaml

根据 arcrole 分类：
- domain-member → domains: <domain> -> members: [member...]
- dimension-domain → dimensions: <dimension> -> domain: <domain>
- 其他/未给 arcrole → parents: <from> -> children: [to...]

支持字段命名：from_concept/to_concept 或 parent_concept/child_concept
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, Set, Tuple, List
import yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=Path("data/processed"),
                    help="递归查找 def*.jsonl")
    ap.add_argument("--pattern", default="def*.jsonl",
                    help="匹配文件名（默认：def*.jsonl）")
    ap.add_argument("--output", type=Path, default=Path("data/mappings/def.yaml"),
                    help="输出 YAML 路径")
    args = ap.parse_args()

    files = list(args.input_root.rglob(args.pattern))
    print(f"[INFO] found {len(files)} def jsonl files under {args.input_root}")

    domains: Dict[str, Dict[str, Any]] = {}     # domain -> {members: [..], linkroles: set}
    dimensions: Dict[str, Dict[str, Any]] = {}  # dimension -> {domain: <...>, linkroles: set}
    parents: Dict[str, Dict[str, Any]] = {}     # parent -> {children: [..], linkroles: set}

    def add_member(domain: str, member: str, linkrole: str|None):
        if not domain or not member:
            return
        obj = domains.setdefault(domain, {"members": [], "linkroles": set()})
        if member not in obj["members"]:
            obj["members"].append(member)
        if linkrole:
            obj["linkroles"].add(linkrole)

    def set_dimension(dim: str, domain: str, linkrole: str|None):
        if not dim or not domain:
            return
        obj = dimensions.setdefault(dim, {"domain": domain, "linkroles": set()})
        # 如果同一维度出现多个 domain（极少见），保留第一个，并记录 linkroles
        if linkrole:
            obj["linkroles"].add(linkrole)

    def add_child(parent: str, child: str, linkrole: str|None):
        if not parent or not child:
            return
        obj = parents.setdefault(parent, {"children": [], "linkroles": set()})
        if child not in obj["children"]:
            obj["children"].append(child)
        if linkrole:
            obj["linkroles"].add(linkrole)

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

                frm = rec.get("from_concept") or rec.get("parent_concept")
                to  = rec.get("to_concept")   or rec.get("child_concept")
                arc = (rec.get("arcrole") or "").lower()
                linkrole = rec.get("linkrole")

                # 分类：常见 arcrole 关键字
                if "domain-member" in arc:
                    # from = domain, to = member
                    add_member(frm, to, linkrole)
                elif "dimension-domain" in arc:
                    # from = dimension, to = domain
                    set_dimension(frm, to, linkrole)
                else:
                    # 其他：视作普通父子关系
                    add_child(frm, to, linkrole)

    # 排序与清理（把 set 转 list）
    for d in domains.values():
        d["members"].sort()
        d["linkroles"] = sorted(d["linkroles"])
    for d in dimensions.values():
        d["linkroles"] = sorted(d["linkroles"])
    for p in parents.values():
        p["children"].sort()
        p["linkroles"] = sorted(p["linkroles"])

    out = {
        "domains": domains,       # Domain -> Members
        "dimensions": dimensions, # Dimension -> Domain
        "parents": parents,       # 普通父子
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fo:
        yaml.safe_dump(out, fo, allow_unicode=True, sort_keys=True)

    print(f"[DONE] wrote {args.output} (domains={len(domains)}, dimensions={len(dimensions)}, parents={len(parents)})")

if __name__ == "__main__":
    main()
