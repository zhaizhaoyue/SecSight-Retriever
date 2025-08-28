#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
# ---------------- filename regex ----------------
DASH_CLASS = r"\-\u2010\u2011\u2012\u2013\u2014\u2212"  # -, ‐, -, ‒, –, —, −(math minus)
ACCNO_SEARCH_RE = re.compile(
    rf"(?<!\d)(\d{{10}})[{DASH_CLASS}](\d{{2}})[{DASH_CLASS}](\d{{6}})(?!\d)"
)
FORM_RE  = re.compile(r"\b(10-K|10-Q|20-F|40-F|8-K)\b", re.I)
DATE8_RE = re.compile(r"(?<!\d)(\d{8})(?!\d)")

# ---------------- try import config ----------------
def _try_import_cfg():
    try:
        from config import cfg_get as _cfg_get, path_join as _path_join
        return _cfg_get, _path_join
    except Exception:
        try:
            import yaml
        except Exception:
            yaml = None

        def _project_root_guess() -> Path:
            p = Path(__file__).resolve()
            for parent in p.parents:
                if (parent / ".git").exists():
                    return parent
            return p.parent

        _PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", _project_root_guess()))
        _CFG_PATH = _PROJECT_ROOT / "configs" / "config.yaml"

        _CFG_RAW: Dict[str, Any] = {}
        if yaml and _CFG_PATH.exists():
            with open(_CFG_PATH, "r", encoding="utf-8") as f:
                _CFG_RAW = yaml.safe_load(f) or {}

        def _expand(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _expand(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_expand(x) for x in obj]
            if isinstance(obj, str):
                return obj.replace("${project_root}", str(_PROJECT_ROOT))
            return obj

        _CFG = _expand(_CFG_RAW)

        def _cfg_get(path: str, default: Any = None) -> Any:
            cur: Any = _CFG
            for key in path.split("."):
                if not isinstance(cur, dict) or key not in cur:
                    return default
                cur = cur[key]
            return cur

        def _path_join(*parts: str | os.PathLike) -> Path:
            return Path(*parts).expanduser().resolve()

        return _cfg_get, _path_join

_cfg_get, _path_join = _try_import_cfg()

# ---------------- arelle import ----------------
try:
    from arelle import Cntlr, FileSource
except Exception as e:
    raise SystemExit("请先安装 arelle-release: pip install arelle-release\n" + str(e))

# ---------------- filename regex ----------------
ACCNO_RE = re.compile(r"\b\d{10}-\d{2}-\d{6}\b")


def sniff_meta(p: Path) -> Dict[str, Any]:
    name = p.name
    base = p.stem
    meta = {
        "ticker": None, "year": None, "form": None,
        "accno": None, "doc_date": None, "fy": None, "fq": None,
        "source_path": str(p),
    }

    # 1) accno / form / docdate
    m_acc = ACCNO_SEARCH_RE.search(name)
    if not m_acc:
        # 再试整个路径（有时上游会把 accno 放到父目录名里）
        m_acc = ACCNO_SEARCH_RE.search(str(p))
    if m_acc:
        meta["accno"] = f"{m_acc.group(1)}-{m_acc.group(2)}-{m_acc.group(3)}"

    m_form = FORM_RE.search(name)
    if m_form:
        meta["form"] = m_form.group(1).upper()

    dates = list(DATE8_RE.finditer(name))
    if dates:
        meta["doc_date"] = dates[-1].group(1)

    # 2) 基于位置推断 ticker/year，允许连字符和点
    tokens = base.split("_")
    if tokens and tokens[0].upper() == "US":
        tokens = tokens[1:]

    def looks_ticker(tok: str) -> bool:
        # 允许字母数字、点和连字符，最多 12 位
        return bool(re.fullmatch(r"[A-Z0-9.\-]{1,12}", tok))

    if len(tokens) >= 1 and looks_ticker(tokens[0].upper()):
        meta["ticker"] = tokens[0].upper()

    if len(tokens) >= 2 and tokens[1].isdigit() and len(tokens[1]) == 4:
        meta["year"] = tokens[1]

    # 3) 派生 fy/fq/year
    if not meta.get("year") and meta.get("doc_date"):
        meta["year"] = meta["doc_date"][:4]

    if meta.get("year") and meta["year"].isdigit():
        meta["fy"] = int(meta["year"])

    if meta.get("doc_date"):
        try:
            mm = int(meta["doc_date"][4:6])
            meta["fq"] = {3:"Q1",6:"Q2",9:"Q3",12:"Q4"}.get(mm)
        except Exception:
            pass

    return meta



FORM_SET = {"10-K","10-Q","20-F","40-F","8-K"}

def sniff_from_parents(p: Path) -> Dict[str, Any]:
    parts = list(p.parts)
    up = [s.upper() for s in parts]
    out = {"ticker": None, "form": None, "accno": None}
    for i, seg in enumerate(up):
        if seg in FORM_SET:
            if i > 0 and re.fullmatch(r"[A-Z0-9.\-]{1,12}", up[i-1]):
                out["ticker"] = up[i-1]
            # 先直接试下一个段是不是 accno；否则扫描整段
            if i + 1 < len(parts):
                seg_next = parts[i+1]
                m_acc = ACCNO_SEARCH_RE.fullmatch(seg_next) or ACCNO_SEARCH_RE.search(seg_next)
                if m_acc:
                    out["accno"] = f"{m_acc.group(1)}-{m_acc.group(2)}-{m_acc.group(3)}"
            out["form"] = seg
            break
    if not out["accno"]:
        for seg in parts:
            m_acc = ACCNO_SEARCH_RE.fullmatch(seg) or ACCNO_SEARCH_RE.search(seg)
            if m_acc:
                out["accno"] = f"{m_acc.group(1)}-{m_acc.group(2)}-{m_acc.group(3)}"
                break
    return out

def enrich_meta_from_dei(model_xbrl, meta: Dict[str, Any]) -> Dict[str, Any]:
    import re

    def qname_localname(qn) -> Optional[str]:
        # qn 可能是 arelle.ModelValue.QName；优先用 localName
        if hasattr(qn, "localName") and qn.localName:
            return qn.localName
        # 退路：从字符串里解析，兼容 "{ns}Local" 或 "prefix:Local"
        if qn is not None:
            s = str(qn)
            if "}" in s and s.startswith("{"):
                return s.split("}", 1)[1]
            if ":" in s:
                return s.split(":", 1)[1]
            return s
        return None

    def get_dei(local: str) -> Optional[str]:
        for f in getattr(model_xbrl, "facts", []):
            qn = getattr(f, "qname", None) or getattr(getattr(f, "concept", None), "qname", None)
            ln = qname_localname(qn)
            if ln == local:
                v = getattr(f, "xValue", None)
                if v is None:
                    v = getattr(f, "value", None)
                if v is not None:
                    return str(v).strip()
        return None

    ticker = get_dei("TradingSymbol")
    form   = get_dei("DocumentType")
    fy_s   = get_dei("DocumentFiscalYearFocus")
    fq_s   = get_dei("DocumentFiscalPeriodFocus")
    dped   = get_dei("DocumentPeriodEndDate")  # YYYY-MM-DD

    if not meta.get("ticker") and ticker:
        meta["ticker"] = ticker.upper()

    if not meta.get("form") and form:
        fup = form.upper()
        m = re.search(r"(10-K|10-Q|20-F|40-F|8-K)", fup)   # 兼容 "10-Q/A" 等
        if m:
            meta["form"] = m.group(1)

    if not meta.get("doc_date") and dped and re.fullmatch(r"\d{4}-\d{2}-\d{2}", dped):
        meta["doc_date"] = dped.replace("-", "")

    # FY / FQ
    if not meta.get("fy"):
        if fy_s and fy_s.isdigit():
            meta["fy"] = int(fy_s)
        elif meta.get("doc_date"):
            meta["fy"] = int(meta["doc_date"][:4])

    if not meta.get("fq"):
        if fq_s:
            q = fq_s.upper()
            if q in {"Q1","Q2","Q3","Q4"}:
                meta["fq"] = q
            elif q in {"FY","ANNUAL"}:
                meta["fq"] = None
        elif meta.get("doc_date"):
            try:
                mm = int(meta["doc_date"][4:6])
                meta["fq"] = {3:"Q1",6:"Q2",9:"Q3",12:"Q4"}.get(mm)
            except Exception:
                pass

    # ★ 确保 year 有值（目录用的是 year）
    if not meta.get("year") and meta.get("fy"):
        meta["year"] = str(meta["fy"])

    return meta



# ---------------- arelle helpers ----------------
def load_instance(ctrl: Cntlr.Cntlr, file_path: str):
    mm = ctrl.modelManager
    try:
        model_xbrl = mm.load(file=file_path)
        if model_xbrl: return model_xbrl
    except TypeError: pass
    fs = FileSource.openFileSource(file_path, ctrl)
    model_xbrl = mm.load(fs)
    fs.close()
    return model_xbrl

def fact_qname(f): 
    qn = getattr(getattr(f,"concept",None),"qname",None) or getattr(f,"qname",None)
    return str(qn) if qn else ""

def fact_value_num(f):
    xv = getattr(f,"xValue",None)
    return xv if xv is not None and not isinstance(xv,str) else None

def fact_period(f):
    ctx = getattr(f,"context",None)
    if ctx is None: return None,None,None
    to_iso = lambda dt: (dt.isoformat()[:10] if hasattr(dt,"isoformat") else str(dt)) if dt else None
    if getattr(ctx,"startDatetime",None) or getattr(ctx,"endDatetime",None) or getattr(ctx,"instantDatetime",None):
        return to_iso(ctx.startDatetime), to_iso(ctx.endDatetime), to_iso(ctx.instantDatetime)
    return ctx.startDate, ctx.endDate, ctx.instantDate

def iter_instance_files(indir: Path) -> List[Path]:
    cands = []
    for p in indir.rglob("*"):
        if not p.is_file(): continue
        name = p.name.lower()
        if any(name.endswith(s) for s in ["_lab.xml","_def.xml","_pre.xml","_cal.xml"]): continue
        if name.endswith((".xml",".htm",".html","_htm.xml")):
            cands.append(p)
    return cands

# ---------------- main parse ----------------
def parse_one(file_path: Path) -> pd.DataFrame:
    ctrl = Cntlr.Cntlr(logFileName=None)
    mx = load_instance(ctrl, str(file_path))
    rows = []
# 1) 文件名
    base_meta = sniff_meta(file_path)

# 2) 父目录兜底
    pmeta = sniff_from_parents(file_path)
    for k, v in pmeta.items():
        if not base_meta.get(k) and v:
            base_meta[k] = v

# 3) DEI 兜底（需要在已加载 model_xbrl 后执行）
    base_meta = enrich_meta_from_dei(mx, base_meta)

# （可选调试）若仍缺失，打印一次文件名
    if not base_meta.get("ticker") or not base_meta.get("form") or not base_meta.get("accno"):
        print(f"[dbg] meta incomplete -> {file_path.name} -> {base_meta}")

    for f in mx.facts:
        if getattr(f,"isNil",False): continue
        ps,pe,inst = fact_period(f)
        row = {
            "concept": fact_qname(f),
            "value_raw": getattr(f,"value",None),
            "value_num": fact_value_num(f),
            "period_start": ps, "period_end": pe, "instant": inst,
            **base_meta
        }
        rows.append(row)
    mx.close()
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default=None)
    ap.add_argument("--out", dest="outdir", default=None)
    ap.add_argument(
    "--to",
    dest="fmt",
    nargs="+",                                 # 支持多选：--to parquet jsonl
    choices=["parquet","csv","jsonl"],
    default=["parquet","jsonl"]                # ★ 默认：点击运行就导出 parquet + jsonl
    )
    args = ap.parse_args()

    raw_in  = _cfg_get("data.raw_reports.standard", "${project_root}/data/raw_reports/standard")
    raw_out = _cfg_get("data.processed", "${project_root}/data/processed")

    indir  = Path(args.indir).resolve() if args.indir else Path(raw_in.replace("${project_root}", str(Path(__file__).resolve().parents[2]))).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else Path(raw_out.replace("${project_root}", str(Path(__file__).resolve().parents[2]))).resolve()

    files = iter_instance_files(indir)
    if not files:
        raise SystemExit(f"No instance files found in {indir}")

    for fp in files:
        try:
            df = parse_one(fp)
        except Exception as e:
            print(f"[fail] {fp.name}: {e}")
            continue
        if df.empty:
            continue

    # ✅ 从解析后的 DataFrame 拿元数据（已经过 DEI/父目录兜底）
        def first_nonnull(df, col, default=None):
            if col in df.columns:
                s = df[col].dropna()
                if not s.empty:
                    return str(s.iloc[0])
            return default

        # 在保存循环里改成这样：
        ticker = first_nonnull(df, "ticker", "UNKNOWN").upper()

        year = first_nonnull(df, "year", None)
        if not year:
            fy = first_nonnull(df, "fy", None)
            year = str(fy) if fy else "NA"

        form = first_nonnull(df, "form", None)
        if not form:
            # 再试一次用文件名快速猜
            m_form = FORM_RE.search(Path(fp).name)
            form = m_form.group(1).upper() if m_form else "NA"
        else:
            form = form.upper()

        accno = first_nonnull(df, "accno", None) or "ACCNO_NA"

        out_dir = outdir / ticker / year / f"{form}_{accno}"
        out_dir.mkdir(parents=True, exist_ok=True)

        fmts = [f.lower() for f in (args.fmt or ["parquet","jsonl"])]
        for fmt in fmts:
            if fmt == "parquet":
                out_path = out_dir / "facts.parquet"
                df.to_parquet(out_path, index=False)
            elif fmt == "csv":
                out_path = out_dir / "facts.csv"
                df.to_csv(out_path, index=False, encoding="utf-8")
            elif fmt == "jsonl":
                out_path = out_dir / "facts.jsonl"
                df.to_json(out_path, orient="records", lines=True, force_ascii=False)
            else:
                raise ValueError(f"Unknown format: {fmt}")

            print(f"[ok] saved {len(df)} facts -> {out_path}")

if __name__=="__main__":
    main()
