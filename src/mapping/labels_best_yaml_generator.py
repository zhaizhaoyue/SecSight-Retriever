
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan data/processed/**/labels.jsonl → build data/mappings/labels_best.yaml

为每个 concept 选出一个“权威展示名”(preferred_<lang>)：
- 语言优先级：默认 en → zh → 其他（可通过 --lang-priority 调整）
- 角色优先级：默认 standard > terse > total > 其他（可通过 --role-priority 调整）
- 同条件下长度更短优先
- 可选移除尾部 "[Member]"（--strip-member-brackets）
"""

from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, DefaultDict
from collections import defaultdict
import yaml

MEMBER_TAIL_RE = re.compile(r"\s*\[Member\]\s*$", re.IGNORECASE)

def normalize_lang(lang: Optional[str]) -> str:
    if not lang:
        return "en"
    return lang.split("-")[0].lower()

def score_label(
    label_text: str,
    lang: str,
    role: str,
    lang_priority: List[str],
    role_priority: List[str],
) -> Tuple[int, int, int]:
    """
    返回一个可排序的分数（三元组，越大越好；最后对长度取负值以“短优先”）。
    排序 key = (lang_score, role_score, -len(label_text))
    """
    try:
        lang_score = len(lang_priority) - lang_priority.index(lang)
    except ValueError:
        lang_score = 0

    # role 可能有多种写法，统一小写匹配关键字
    r = role.lower() if role else ""
    role_rank = 0
    for i, key in enumerate(role_priority[::-1], start=1):
        if key in r:
            role_rank = i
            break
    # 让“越靠前越高分”
    role_score = role_rank

    return (lang_score, role_score, -len(label_text or ""))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=Path("data/processed"),
                    help="根目录，递归查找 labels.jsonl")
    ap.add_argument("--pattern", default="labels.jsonl",
                    help="匹配的文件名（默认 labels.jsonl）")
    ap.add_argument("--output", type=Path, default=Path("data/mappings/labels_best.yaml"),
                    help="输出 YAML 路径")
    ap.add_argument("--lang-priority", nargs="*", default=["en", "zh"],
                    help="语言优先级（主码，默认：en zh）")
    ap.add_argument("--role-priority", nargs="*", default=["standard", "terse", "total"],
                    help="label_role 优先级关键字，按顺序高→低（默认：standard terse total）")
    ap.add_argument("--strip-member-brackets", action="store_true",
                    help="移除尾部的 '[Member]' 文本（对维度成员展示更友好）")
    args = ap.parse_args()

    files = list(args.input_root.rglob(args.pattern))
    if not files:
        print(f"[WARN] no {args.pattern} found under {args.input_root}")
    else:
        print(f"[INFO] found {len(files)} files")

    # concept -> 收集候选
    # 候选项结构：{lang: [(label_text, role, file_path, line_no), ...]}
    candidates: DefaultDict[str, DefaultDict[str, List[Tuple[str, str, str, int]]]] = defaultdict(lambda: defaultdict(list))

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
                label_text = rec.get("label_text") or ""
                if not concept or not label_text.strip():
                    continue

                lang = normalize_lang(rec.get("lang") or rec.get("lang_safe"))
                role = (rec.get("label_role") or "").strip()

                # 清理
                label_text = " ".join(label_text.split())
                if args.strip_member_brackets:
                    label_text = MEMBER_TAIL_RE.sub("", label_text)

                candidates[concept][lang].append((label_text, role, str(fp), ln_no))

    # 选最佳：按照 lang/role/长度 评分
    best_map: Dict[str, Dict[str, str]] = {}
    for concept, langs in candidates.items():
        # 想支持多语言展示：为每个 lang_priority 挑一条
        entry: Dict[str, str] = {}
        for lang in args.lang_priority:
            items = langs.get(lang, [])
            if not items:
                continue
            # 选择最优
            items_sorted = sorted(
                items,
                key=lambda x: score_label(x[0], lang, x[1], args.lang_priority, args.role_priority),
                reverse=True,
            )
            entry[f"preferred_{lang}"] = items_sorted[0][0]
        # 若优先语言都缺失，尝试任意语言里挑一个总体最优
        if not entry:
            fallback_pool = []
            for lang_k, items in langs.items():
                for it in items:
                    fallback_pool.append((lang_k, *it))  # (lang, label_text, role, file, line)
            if fallback_pool:
                fallback_pool_sorted = sorted(
                    fallback_pool,
                    key=lambda x: score_label(x[1], x[0], x[2], args.lang_priority, args.role_priority),
                    reverse=True,
                )
                chosen = fallback_pool_sorted[0]
                entry[f"preferred_{normalize_lang(chosen[0])}"] = chosen[1]
        if entry:
            best_map[concept] = entry

    # 写 YAML
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fo:
        yaml.safe_dump(best_map, fo, allow_unicode=True, sort_keys=True)

    print(f"[DONE] wrote {args.output} (concepts={len(best_map)})")

if __name__ == "__main__":
    main()
