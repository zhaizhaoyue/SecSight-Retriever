#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_ROOT   = PROJECT_ROOT / "data" / "clean"
REPORT_DIR   = CLEAN_ROOT / "_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_CSV   = REPORT_DIR / "validation_report.csv"

SAFE = lambda x: (x if x is not None else "NA")

def read_jsonl(p: Path) -> pd.DataFrame:
    rows=[]
    with p.open("r", encoding="utf-8") as f:
        for i,l in enumerate(f,1):
            s=l.strip()
            if not s: continue
            try: rows.append(json.loads(s))
            except Exception as e:
                rows.append({"__parse_error__": str(e), "__line__": i})
    return pd.DataFrame(rows)

def maybe_read(path_base: Path, stem: str) -> Optional[pd.DataFrame]:
    for name in (f"{stem}.parquet", f"{stem}.jsonl"):
        p = path_base / name
        if p.exists():
            return pd.read_parquet(p) if p.suffix==".parquet" else read_jsonl(p)
    return None

def pct(x, n): 
    return 0 if n==0 else round(100.0*x/n, 2)

def check_facts(dirpath: Path) -> List[Dict[str,Any]]:
    issues=[]
    facts = maybe_read(dirpath, "fact")
    if facts is None:
        issues.append({"dir":str(dirpath),"type":"facts","severity":"error","msg":"missing fact.{parquet|jsonl}"})
        return issues
    n=len(facts)
    req = ["concept","rag_text","period_label","ticker","form","accno"]
    for c in req:
        if c not in facts.columns:
            issues.append({"dir":str(dirpath),"type":"facts","severity":"error","msg":f"missing column {c}"})
    # empties
    empty_rag = facts["rag_text"].isna().sum() if "rag_text" in facts else n
    if empty_rag>0:
        issues.append({"dir":str(dirpath),"type":"facts","severity":"warn","msg":f"rag_text empty {empty_rag}/{n} ({pct(empty_rag,n)}%)"})
    # fq/fy parsing smoke test
    def to_int(s):
        if pd.isna(s): return None
        m=re.search(r"\d+", str(s))
        return int(m.group()) if m else None
    if "fy" in facts.columns:
        bad_fy = sum(v is None for v in facts["fy"].map(to_int))
        if bad_fy>0:
            issues.append({"dir":str(dirpath),"type":"facts","severity":"info","msg":f"fy non-numeric {bad_fy}/{n} (年度报正常可忽略)"})
    if "fq" in facts.columns:
        # fq 允许为空（10-K），仅提示非空但不可解析的
        mask = facts["fq"].notna()
        bad_fq = sum(v is None for v in facts.loc[mask,"fq"].map(to_int)) if mask.any() else 0
        if bad_fq>0:
            issues.append({"dir":str(dirpath),"type":"facts","severity":"warn","msg":f"fq non-numeric among non-null {bad_fq}/{int(mask.sum())}"})
    # duplicates (same concept+period+accno+value_display)
    key_cols = [c for c in ["concept","period_label","accno","value_display"] if c in facts.columns]
    if key_cols:
        dup = facts.duplicated(subset=key_cols).sum()
        if dup>0:
            issues.append({"dir":str(dirpath),"type":"facts","severity":"warn","msg":f"potential duplicates by {key_cols}: {dup} rows"})

    # labels coverage if available
    labs = maybe_read(dirpath, "labels")
    if labs is not None and "concept" in labs.columns and "label_text" in labs.columns:
        facts_concepts = facts["concept"].dropna().unique()
        covered = pd.Series(facts_concepts).isin(set(labs["concept"].dropna().unique())).sum()
        if covered < len(facts_concepts):
            issues.append({"dir":str(dirpath),"type":"labels","severity":"info","msg":f"label coverage {covered}/{len(facts_concepts)} concepts"})
    return issues

def check_def_calc(dirpath: Path) -> List[Dict[str,Any]]:
    issues=[]
    defs = maybe_read(dirpath, "definition_arcs")
    cal  = maybe_read(dirpath, "calculation_edges")
    facts= maybe_read(dirpath, "fact")

    if defs is None:
        issues.append({"dir":str(dirpath),"type":"def","severity":"info","msg":"definition_arcs.* not found (可选)"})
    else:
        for c in ["from_concept","to_concept","linkrole"]:
            if c not in defs.columns:
                issues.append({"dir":str(dirpath),"type":"def","severity":"warn","msg":f"missing column {c}"})

    if cal is None:
        issues.append({"dir":str(dirpath),"type":"calc","severity":"info","msg":"calculation_edges.* not found (可选)"})
        return issues

    # columns & NaNs
    need = ["parent_concept","child_concept","weight"]
    for c in need:
        if c not in cal.columns:
            issues.append({"dir":str(dirpath),"type":"calc","severity":"error","msg":f"missing column {c}"})
    nan_pct = {c: pct(cal[c].isna().sum(), len(cal)) for c in need if c in cal.columns}
    for c, r in nan_pct.items():
        if r>0:
            issues.append({"dir":str(dirpath),"type":"calc","severity":"warn","msg":f"{c} NaN {r}%"} )

    # child concepts not appearing in facts
    if facts is not None and "concept" in facts.columns:
        factset = set(facts["concept"].dropna().unique())
        missing_children = [c for c in cal["child_concept"].dropna().unique() if c not in factset]
        if missing_children:
            issues.append({"dir":str(dirpath),"type":"calc","severity":"info","msg":f"children not in facts: {len(missing_children)} concepts (样本)"} )

    # lightweight sum check on a small sample of parents
    if facts is not None and {"concept","value_num_clean","period_label"}.issubset(facts.columns):
        # pick up to 10 parents
        parents = list(pd.Series(cal["parent_concept"].dropna().unique()).head(10))
        for pc in parents:
            rows = cal[cal["parent_concept"]==pc]
            children = rows[["child_concept","weight"]].dropna()
            if children.empty: 
                continue
            # for each period, compare
            for per, g in facts.groupby("period_label"):
                parent_val = facts.loc[(facts["concept"]==pc) & (facts["period_label"]==per), "value_num_clean"]
                if parent_val.empty or parent_val.isna().all():
                    continue
                s = 0.0
                ok_any=False
                for _, r in children.iterrows():
                    cc, w = r["child_concept"], r["weight"]
                    vals = facts.loc[(facts["concept"]==cc) & (facts["period_label"]==per), "value_num_clean"]
                    if vals.empty or vals.isna().all():
                        ok_any=True; continue
                    s += float(vals.iloc[0]) * float(w)
                parent=float(parent_val.iloc[0])
                if not np.isfinite(parent) or not np.isfinite(s): 
                    continue
                tol = max(1.0, 0.005*abs(parent))
                if abs(parent - s) > tol:
                    issues.append({"dir":str(dirpath),"type":"calc_check","severity":"info",
                                   "msg":f"sum mismatch {pc} @ {per}: parent={parent:.2f}, sum={s:.2f}, tol={tol:.2f}"})
    return issues

def main():
    records=[]
    # 遍历 data/clean 下每个最末级目录（含 fact.* 的）
    for fact_file in CLEAN_ROOT.rglob("fact.jsonl"):
        dirpath = fact_file.parent
        records += check_facts(dirpath)
        records += check_def_calc(dirpath)

    if not records:
        print("[OK] 未发现问题项。")
        return

    df = pd.DataFrame(records)
    df.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")
    print(f"[DONE] 报告生成：{REPORT_CSV}  |  发现 {len(df)} 条记录")
    print(df.groupby(["type","severity"]).size())

if __name__ == "__main__":
    main()
