from __future__ import annotations
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
from lxml import etree

# ---- config loader (your project config.py) ----
# ---- config loader (prefer project config.py; fallback to config.yaml or defaults) ----
try:
    # 如果项目里有 config.py，就直接用
    from config import cfg_get, get_path, ensure_dir
except Exception:
    # minimal fallback if config.py isn't available
    import os
    try:
        import yaml  # 可选
    except Exception:
        yaml = None

    def _discover_root() -> Path:
        p = Path(__file__).resolve()
        for parent in [p.parent, *p.parents]:
            if (parent / ".git").exists():
                return parent
        return p.parent

    _ROOT = _discover_root()
    cfg_path = _ROOT / "configs" / "config.yaml"
    _CFG: dict[str, Any] = {}

    if yaml is not None and cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            _CFG = yaml.safe_load(f) or {}

    def cfg_get(path: str, default: Any = None) -> Any:
        """
        读取 config.yaml 中的嵌套键（点号路径），取不到就返回 default。
        会把 ${project_root} 展开成实际路径。
        """
        cur: Any = _CFG
        for k in path.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        if isinstance(cur, str):
            return cur.replace("${project_root}", str(_ROOT))
        return cur

    def get_path(path: str, default: Any = None) -> Path:
        """
        返回 Path；如果配置没有该键，提供合理默认：
          - data.standard -> <project>/data/raw_reports/standard
          - data.parsed   -> <project>/data/processed
        """
        v = cfg_get(path, default)
        if v is None:
            if path == "data.standard":
                v = _ROOT / "data" / "raw_reports" / "standard"
            elif path == "data.parsed":
                v = _ROOT / "data" / "processed"
            else:
                raise KeyError(f"missing config key: {path}")
        return Path(str(v).replace("${project_root}", str(_ROOT))).expanduser().resolve()

    def ensure_dir(p: str | Path) -> Path:
        p = Path(p)
        p.mkdir(parents=True, exist_ok=True)
        return p

# 允许点和连字符的 ticker；accno 允许各种 dash；docdate 用“最后 8 位数字”兜底
DASH_CLASS = r"\-\u2010\u2011\u2012\u2013\u2014\u2212"  # -, ‐, ‒, –, —, −
ACCNO_SEARCH_RE = re.compile(
    rf"(?<!\d)(\d{{10}})[{DASH_CLASS}](\d{{2}})[{DASH_CLASS}](\d{{6}})(?!\d)"
)
DATE8_RE = re.compile(r"(?<!\d)(\d{8})(?!\d)")
FNAME_RE = re.compile(r"""
^
(?:US_)?                                   # 可选 US_
(?P<ticker>[A-Z0-9.\-]{1,12})_             # ticker
(?P<year>\d{4})_                           # 年（来自文件名）
(?P<form>10\-K|10\-Q|20\-F|40\-F|8\-K)_    # 表单
(?P<accno>\d{10}\-\d{2}\-\d{6})            # accession
(?:_[A-Za-z0-9\-]+)?                       # 可选 slug，如 brka-20230630 的前缀
(?:_(?P<docdate>\d{8}))?                   # 可选 docdate
(?:_(?P<type>cal|def|pre|lab|htm))?        # 可选类型后缀
\.xml$                                     # 结尾
""", re.X | re.I)

def make_outdir(outroot: Path, ticker, fy, form, accno) -> Path:
    t = (ticker or "UNKNOWN").upper()
    # y 必须是字符串；fy 可能是 float/NaN
    y = str(int(fy)) if (fy is not None and pd.notna(fy)) else "NA"
    f = (form or "NA").upper()
    a = accno or "ACCNO_NA"
    return ensure_dir(outroot / t / y / f"{f}_{a}")

def write_out(df: pd.DataFrame, outdir: Path, stem: str, fmts: list[str]):
    fmts = [f.lower() for f in (fmts or ["parquet","jsonl"])]
    for fmt in fmts:
        if fmt == "parquet":
            df.to_parquet(outdir / f"{stem}.parquet", index=False)
        elif fmt == "csv":
            df.to_csv(outdir / f"{stem}.csv", index=False, encoding="utf-8")
        elif fmt == "jsonl":
            df.to_json(outdir / f"{stem}.jsonl", orient="records", lines=True, force_ascii=False)
        else:
            raise ValueError(f"Unknown format: {fmt}")



def sniff_meta(p: Path) -> Dict[str, Any]:
    name = p.name
    meta: Dict[str, Any] = {
        "ticker": None, "year": None, "fy": None, "form": None, "accno": None,
        "doc_date": None, "file_type": None, "source_path": str(p),
    }

    m = FNAME_RE.search(name)
    if m:
        gd = m.groupdict()
        meta["ticker"]   = (gd.get("ticker") or "").upper() or None
        meta["year"]     = gd.get("year") or None
        meta["form"]     = (gd.get("form") or "").upper() or None
        meta["accno"]    = gd.get("accno") or None
        meta["doc_date"] = gd.get("docdate") or None
        meta["file_type"]= gd.get("type") or None

    # accno 兜底（全路径扫描 + 规范化）
    if not meta["accno"]:
        ma = ACCNO_SEARCH_RE.search(name) or ACCNO_SEARCH_RE.search(str(p))
        if ma:
            meta["accno"] = f"{ma.group(1)}-{ma.group(2)}-{ma.group(3)}"

    # doc_date 兜底：取文件名里出现的“最后一个 8 位数字”
    if not meta["doc_date"]:
        hits = DATE8_RE.findall(name)
        if hits:
            meta["doc_date"] = hits[-1]

    # fy：优先用文件名 year；没有就退 doc_date[:4]
    if meta["year"]:
        try:
            meta["fy"] = int(meta["year"])
        except Exception:
            pass
    if meta.get("fy") is None and meta.get("doc_date"):
        try:
            meta["fy"] = int(meta["doc_date"][:4])
        except Exception:
            pass

    return meta

def parse_presentation(pre_xml: Path) -> pd.DataFrame:
    tree = _parse_xml(pre_xml)
    rows, meta = [], sniff_meta(pre_xml)
    for prelink in tree.findall(".//link:presentationLink", namespaces=NSMAP):
        linkrole = prelink.get(f"{{{NSMAP['xlink']}}}role")
        locs = _loc_map(prelink)
        for arc in prelink.findall(".//link:presentationArc", namespaces=NSMAP):
            src = arc.get(f"{{{NSMAP['xlink']}}}from"); dst = arc.get(f"{{{NSMAP['xlink']}}}to")
            order = arc.get("order")
            if not src or not dst: continue
            parent = locs.get(src); child = locs.get(dst)
            if not parent or not child: continue
            rows.append({"parent_concept": parent, "child_concept": child,
                         "order": float(order) if order else None,
                         "linkrole": linkrole, **meta})
    return pd.DataFrame(rows)

# ---------- core parsers ----------
NSMAP = {
    "link": "http://www.xbrl.org/2003/linkbase",
    "xlink": "http://www.w3.org/1999/xlink",
}

def _parse_xml(path: Path) -> etree._ElementTree:
    parser = etree.XMLParser(recover=True, resolve_entities=False, huge_tree=True)
    return etree.parse(str(path), parser=parser)

def _loc_map(linknode: etree._Element) -> Dict[str, str]:
    """
    Build mapping: locator label -> concept QName (from xlink:href '...#qname').
    """
    locs = {}
    for loc in linknode.findall(".//link:loc", namespaces=NSMAP):
        lab = loc.get(f"{{{NSMAP['xlink']}}}label")
        href = loc.get(f"{{{NSMAP['xlink']}}}href")
        if lab and href:
            # href can be '...#us-gaap_SalesRevenueNet' or '#us-gaap_SalesRevenueNet'
            qname = href.split("#")[-1]
            if ":" in qname:
                norm = qname
            elif "_" in qname:
                pre, local = qname.split("_", 1)
                norm = f"{pre}:{local}"
            else:
                norm = qname
            locs[lab] = norm
    return locs

def parse_labels(lab_xml: Path) -> pd.DataFrame:
    tree = _parse_xml(lab_xml)
    rows: List[Dict[str, Any]] = []
    meta = sniff_meta(lab_xml)

    for lblink in tree.findall(".//link:labelLink", namespaces=NSMAP):
        linkrole = lblink.get(f"{{{NSMAP['xlink']}}}role")
        locs = _loc_map(lblink)

        # label resources
        res_by_label: Dict[str, Dict[str, Any]] = {}
        for lab in lblink.findall(".//link:label", namespaces=NSMAP):
            res_label = lab.get(f"{{{NSMAP['xlink']}}}label")
            role      = lab.get(f"{{{NSMAP['xlink']}}}role")
            lang      = lab.get("{http://www.w3.org/XML/1998/namespace}lang")
            text      = (lab.text or "").strip()
            if res_label:
                res_by_label[res_label] = {"role": role, "lang": lang, "text": text}

        # arcs: from concept locator -> to label resource
        for arc in lblink.findall(".//link:labelArc", namespaces=NSMAP):
            src = arc.get(f"{{{NSMAP['xlink']}}}from")
            dst = arc.get(f"{{{NSMAP['xlink']}}}to")
            if not src or not dst: 
                continue
            concept = locs.get(src)
            res     = res_by_label.get(dst)
            if not concept or not res:
                continue
            rows.append({
                "concept": concept,
                "label_text": res["text"],
                "label_role": res["role"],
                "lang": res["lang"],
                "linkrole": linkrole,
                **meta,
            })
    return pd.DataFrame(rows)

def parse_calculation(cal_xml: Path) -> pd.DataFrame:
    tree = _parse_xml(cal_xml)
    rows: List[Dict[str, Any]] = []
    meta = sniff_meta(cal_xml)

    for callink in tree.findall(".//link:calculationLink", namespaces=NSMAP):
        linkrole = callink.get(f"{{{NSMAP['xlink']}}}role")
        locs = _loc_map(callink)
        for arc in callink.findall(".//link:calculationArc", namespaces=NSMAP):
            src = arc.get(f"{{{NSMAP['xlink']}}}from")
            dst = arc.get(f"{{{NSMAP['xlink']}}}to")
            order = arc.get("order")
            weight = arc.get("weight")
            if not src or not dst: 
                continue
            parent = locs.get(src)
            child  = locs.get(dst)
            if not parent or not child:
                continue
            rows.append({
                "parent_concept": parent,
                "child_concept": child,
                "order": float(order) if order is not None else None,
                "weight": float(weight) if weight is not None else None,
                "linkrole": linkrole,
                **meta,
            })
    return pd.DataFrame(rows)

def parse_definition(def_xml: Path) -> pd.DataFrame:
    tree = _parse_xml(def_xml)
    rows: List[Dict[str, Any]] = []
    meta = sniff_meta(def_xml)

    for deflink in tree.findall(".//link:definitionLink", namespaces=NSMAP):
        linkrole = deflink.get(f"{{{NSMAP['xlink']}}}role")
        locs = _loc_map(deflink)
        for arc in deflink.findall(".//link:definitionArc", namespaces=NSMAP):
            arcrole = arc.get(f"{{{NSMAP['xlink']}}}arcrole")
            src = arc.get(f"{{{NSMAP['xlink']}}}from")
            dst = arc.get(f"{{{NSMAP['xlink']}}}to")
            order = arc.get("order")
            if not src or not dst:
                continue
            frm = locs.get(src)
            to  = locs.get(dst)
            if not frm or not to:
                continue
            rows.append({
                "from_concept": frm,
                "to_concept": to,
                "arcrole": arcrole,
                "order": float(order) if order is not None else None,
                "linkrole": linkrole,
                **meta,
            })
    return pd.DataFrame(rows)

# ---------- file discovery ----------
def find_linkbases(indir: Path):
    labs, cals, defs, pres = [], [], [], []
    for p in indir.rglob("*.xml"):
        n = p.name.lower()
        if n.endswith("_lab.xml"): labs.append(p)
        elif n.endswith("_cal.xml"): cals.append(p)
        elif n.endswith("_def.xml"): defs.append(p)
        elif n.endswith("_pre.xml"): pres.append(p)
    return labs, cals, defs, pres

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="indir",  default=None, help="Directory containing *_lab.xml/_cal.xml/_def.xml")
    ap.add_argument("--out", dest="outdir", default=None, help="Output root directory")
    ap.add_argument(
        "--to",
        dest="fmt",
        nargs="+",
        choices=["parquet","csv","jsonl"],
        default=["parquet","jsonl"]   # 直接点运行就会导出这两种
    )
    args = ap.parse_args()

    indir  = Path(args.indir).resolve() if args.indir else get_path("data.standard")
    outdir = Path(args.outdir).resolve() if args.outdir else get_path("data.parsed")
    outdir = ensure_dir(outdir)
# 读取参数
    fmts = args.fmt  # 这里是一个 list，如 ["parquet","jsonl"]

    # 扫描
    labs, cals, defs, pres = find_linkbases(indir)
    print(f"[scan] labs={len(labs)}, cals={len(cals)}, defs={len(defs)}, pres={len(pres)} in {indir}")

    # --- labels ---
# --- labels ---
    if labs:
        dfs = []
        for f in labs:
            try:
                df = parse_labels(f)
                if not df.empty:
                    dfs.append(df)
                print(f"[ok][lab] {f.name}: {len(df)} labels")
            except Exception as e:
                print(f"[fail][lab] {f.name}: {e}")
        if dfs:
            all_df = pd.concat(dfs, ignore_index=True)
            for (ticker, fy, form, accno), g in all_df.groupby(["ticker","fy","form","accno"], dropna=False):
                out_d = make_outdir(outdir, ticker, fy, form, accno)
                print("[save labels]", out_d)
                write_out(g, out_d, "labels", fmts)
    else:
        print("[warn] no *_lab.xml found")

    # --- calculation ---
    if cals:
        dfs = []
        for f in cals:
            try:
                df = parse_calculation(f)
                if not df.empty:
                    dfs.append(df)
                print(f"[ok][cal] {f.name}: {len(df)} arcs")
            except Exception as e:
                print(f"[fail][cal] {f.name}: {e}")
        if dfs:
            all_df = pd.concat(dfs, ignore_index=True)
            for (ticker, fy, form, accno), g in all_df.groupby(["ticker","fy","form","accno"], dropna=False):
                out_d = make_outdir(outdir, ticker, fy, form, accno)
                print("[save cal]", out_d)
                write_out(g, out_d, "calculation_edges", fmts)   # ← 这里！
    else:
        print("[warn] no *_cal.xml found")

    # --- definition ---
    if defs:
        dfs = []
        for f in defs:
            try:
                df = parse_definition(f)
                if not df.empty:
                    dfs.append(df)
                print(f"[ok][def] {f.name}: {len(df)} arcs")
            except Exception as e:
                print(f"[fail][def] {f.name}: {e}")
        if dfs:
            all_df = pd.concat(dfs, ignore_index=True)
            for (ticker, fy, form, accno), g in all_df.groupby(["ticker","fy","form","accno"], dropna=False):
                out_d = make_outdir(outdir, ticker, fy, form, accno)
                print("[save def]", out_d)
                write_out(g, out_d, "definition_arcs", fmts)     # ← 这里！
    else:
        print("[warn] no *_def.xml found")


# --- presentation（可选）---
# if pres:
#   同上：parse_presentation -> groupby -> write_out(..., "presentation_edges", fmts)


    print("[done] linkbase extraction finished.")

if __name__ == "__main__":
    main()
