import json
import argparse
from pathlib import Path
from typing import Iterable, Optional, Dict, Any, List
import re
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_INPUT_DIR   = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR  = PROJECT_ROOT / "data" / "clean"


def _to_int_or_none(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    m = re.search(r"\d+", s)
    return int(m.group()) if m else None

# 2) period_label 生成：用上规范化后的 fy/fq
def _mk_period_label(row: pd.Series) -> str:
    fy_i = _to_int_or_none(row.get("fy"))
    fq_i = _to_int_or_none(row.get("fq"))
    inst = row.get("instant")

    if pd.notna(inst):
        return f"FY{fy_i} instant {inst}" if fy_i is not None else f"instant {inst}"

    a = row.get("period_start")
    b = row.get("period_end")
    if pd.notna(a) or pd.notna(b):
        a = str(a) if pd.notna(a) else "?"
        b = str(b) if pd.notna(b) else "?"
        if (fy_i is not None) and (fq_i is not None):
            return f"FY{fy_i} Q{fq_i} {a}→{b}"
        if fy_i is not None:
            return f"FY{fy_i} {a}→{b}"
        return f"{a}→{b}"

    return "period:unknown"
# -------------------------
# Root discovery
# -------------------------
def find_default_root() -> Path:
    """从脚本位置逐级 parent 寻找 data/processed；找不到则回退 CWD/data/processed；再不行用 CWD。"""
    start = Path(__file__).resolve()
    for base in [start.parent, *start.parents]:
        dp = base / "data" / "processed"
        if dp.exists() and dp.is_dir():
            return dp
    cwd = Path.cwd()
    dp = cwd / "data" / "processed"
    return dp if dp.exists() and dp.is_dir() else cwd


# -------------------------
# IO helpers
# -------------------------
def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".jsonl":
        recs: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except Exception as e:
                    print(f"    [WARN] {path.name} 第{i}行 JSON 解析失败：{e}")
        return pd.DataFrame(recs)
    if suf == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input type: {path.name}")


def try_write_parquet(df: pd.DataFrame, out_path: Path) -> bool:
    for engine in ("pyarrow", "fastparquet"):
        try:
            df.to_parquet(out_path, index=False, engine=engine)
            return True
        except Exception:
            continue
    print(f"    [WARN] 未安装 pyarrow/fastparquet，跳过写入 {out_path.name}。"
          f" 可执行：pip install pyarrow   或   pip install fastparquet")
    return False


def save_jsonl(df: pd.DataFrame, out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# -------------------------
# Labels (optional)
# -------------------------
ROLE_PRI = [
    "http://www.xbrl.org/2003/role/label",        # standard
    "http://www.xbrl.org/2003/role/terseLabel",   # terse
    "http://www.xbrl.org/2003/role/verboseLabel", # verbose
]

def read_labels_if_any(dirpath: Path) -> Optional[pd.DataFrame]:
    for name in ("labels.jsonl", "labels.parquet"):
        p = dirpath / name
        if p.exists():
            df = read_table(p)
            if not df.empty and "concept" in df.columns:
                return df
    return None

def build_preferred_labels(labels_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if labels_df is None or labels_df.empty or "concept" not in labels_df.columns:
        return None
    def _role_rank(r: Any) -> int:
        try:
            return ROLE_PRI.index(r)
        except Exception:
            return len(ROLE_PRI)
    labels_df = labels_df.copy()
    labels_df["role_rank"] = labels_df.get("label_role", "").apply(_role_rank)
    labels_df["lang_rank"] = labels_df.get("lang", "").apply(lambda x: 0 if str(x).lower().startswith("en") else 1)
    return (labels_df.sort_values(["concept", "role_rank", "lang_rank"])
            .groupby("concept", as_index=False).first()[["concept", "label_text"]])


# -------------------------
# Cleaning core
# -------------------------
DEF_COLS = ["concept","value_raw","value_num","period_start","period_end","instant",
            "ticker","year","fy","fq","form","accno","doc_date","source_path"]

def _to_float(x: Any) -> float:
    try:
        if isinstance(x, str) and x.strip().lower() in ("true","false"):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def _norm_text(x: Any) -> Any:
    if isinstance(x, str):
        xl = x.strip()
        if xl.lower() in ("true","false"):
            return xl.lower()
        return xl
    return x


def _fmt_value(r: pd.Series) -> str:
    v = r.get("value_num_clean")
    if pd.notna(v):
        if abs(v) >= 1e9: return f"{v/1e9:.3f} B"
        if abs(v) >= 1e6: return f"{v/1e6:.3f} M"
        if abs(v) >= 1e3: return f"{v/1e3:.3f} K"
        return f"{v:.6g}"
    raw = r.get("value_raw_clean")
    return str(raw) if pd.notna(raw) and str(raw).strip() else ""

def _mk_rag_text(r: pd.Series) -> str:
    label = r["label_text"] if pd.notna(r.get("label_text")) else r.get("concept")
    val   = r["value_display"]
    per   = r["period_label"]
    tick  = r.get("ticker")
    form  = r.get("form")
    accno = r.get("accno")
    meta  = f"{tick} {form} accno={accno}" if pd.notna(accno) else f"{tick} {form}"
    return f"{label}: {val} ({per}; {meta})"

def clean_one_facts_table(facts_df: pd.DataFrame, labels_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if facts_df is None or facts_df.empty:
        print("    [SKIP] 空的 facts，跳过")
        return pd.DataFrame()

    # 缺列补齐
    facts_df = facts_df.copy()
    for c in DEF_COLS:
        if c not in facts_df.columns:
            facts_df[c] = pd.NA

    # 标签优选
    pref_labels = build_preferred_labels(labels_df)

    # 派生列
    facts_df["value_num_clean"] = facts_df["value_num"].apply(_to_float)
    facts_df["value_raw_clean"] = facts_df["value_raw"].apply(_norm_text)
    facts_df["period_label"]    = facts_df.apply(_mk_period_label, axis=1)
    facts_df["fy_norm"] = facts_df.get("fy", np.nan).apply(_to_int_or_none)
    facts_df["fq_norm"] = facts_df.get("fq", np.nan).apply(_to_int_or_none)

# 如果你在其它地方也拼 period，用 fy_norm / fq_norm 更稳妥


    if pref_labels is not None:
        facts_df = facts_df.merge(pref_labels, on="concept", how="left")
    else:
        facts_df["label_text"] = pd.NA

    keep = [
        "concept","label_text",
        "value_raw_clean","value_num_clean",
        "period_start","period_end","instant","period_label",
        "ticker","year","fy","fq","form","accno","doc_date","source_path",
    ]
    df = facts_df[keep].drop_duplicates()

    df["value_display"] = df.apply(_fmt_value, axis=1)
    df["rag_text"]      = df.apply(_mk_rag_text, axis=1)

    # 列顺序
    cols = [
        "concept","label_text",
        "value_raw_clean","value_num_clean","value_display","rag_text",
        "period_start","period_end","instant","period_label",
        "ticker","year","fy","fq","form","accno","doc_date","source_path",
    ]
    return df[cols]


# -------------------------
# Scan inputs & out paths
# -------------------------
def iter_inputs(root: Path) -> Iterable[Path]:
    # 避免重复：先 jsonl 再 parquet
    yield from root.rglob("facts.jsonl")
    yield from root.rglob("facts.parquet")

out_root = Path(r"C:\Users\max_s\Desktop\新建文件夹 (2)\Multi-Modal-LLM-Research-Assistant-for-Finance\data\clean")

def compute_out_dir(facts_path: Path, root: Path) -> Path:
    """
    根据 processed 下的相对路径，映射到 clean 下
    """
    rel = facts_path.parent.relative_to(root)  # AAPL/2023/10-K_xxx
    return out_root / rel


# -------------------------
# Main
# -------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Batch clean facts for LLM+RAG (processed → clean, same structure)")
    ap.add_argument("--root", type=str, default=None,
                    help="输入根目录（默认使用 PROJECT_ROOT/data/processed）")
    ap.add_argument("--dry-run", action="store_true", help="只扫描不写文件")
    args = ap.parse_args()

    # 1) 解析根目录：默认 data/processed
    root = Path(args.root).expanduser().resolve() if args.root else DEFAULT_INPUT_DIR.resolve()
    out_root = DEFAULT_OUTPUT_DIR.resolve()

    if not root.exists():
        print(f"[INFO] 输入根目录不存在：{root}")
        return

    # 2) 扫描输入（facts.jsonl / facts.parquet）
    inputs = list(iter_inputs(root))
    if not inputs:
        print(f"[INFO] 在 {root} 下未找到 facts.jsonl / facts.parquet")
        return

    print(f"[INFO] 输入根目录：{root}")
    print(f"[INFO] 输出根目录：{out_root}（与 processed 结构完全一致）")
    print(f"[INFO] 待处理文件数：{len(inputs)}")

    # 3) 主循环
    for i, facts_path in enumerate(inputs, 1):
        try:
            print(f"[{i}/{len(inputs)}] 清洗：{facts_path}")

            facts_df  = read_table(facts_path)
            labels_df = read_labels_if_any(facts_path.parent)
            cleaned   = clean_one_facts_table(facts_df, labels_df)
            if cleaned.empty:
                print("    [SKIP] 空结果，跳过")
                continue

            # ——关键：将 processed 的相对路径映射到 clean——
            try:
                rel = facts_path.parent.relative_to(root)     # e.g. AAPL/2023/10-Q_xxx
                out_dir = out_root / rel
            except ValueError:
                # 万一不在 root 下，降级为字符串替换（更鲁棒）
                out_dir = Path(str(facts_path.parent).replace(
                    str(DEFAULT_INPUT_DIR), str(DEFAULT_OUTPUT_DIR)
                ))

            if args.dry_run:
                print(f"    -> dry-run：rows={len(cleaned)}  out_dir={out_dir}")
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            pq = out_dir / "fact.parquet"
            jl = out_dir / "fact.jsonl"

            ok = try_write_parquet(cleaned, pq)   # 无 pyarrow/fastparquet 会给出友好提示
            save_jsonl(cleaned, jl)

            print(f"    -> 输出：{'fact.parquet, ' if ok else ''}fact.jsonl  rows={len(cleaned)}  dir={out_dir}")

        except Exception as e:
            print(f"[WARN] 处理失败：{facts_path}\n{e}")

    print("[DONE] 全部完成。")



if __name__ == "__main__":
    main()
