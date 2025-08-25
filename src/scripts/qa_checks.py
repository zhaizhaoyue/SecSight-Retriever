from pathlib import Path
import json, csv
import pandas as pd

ROOT = Path("data/processed")
QA   = Path("data/qa"); QA.mkdir(parents=True, exist_ok=True)

def numeric_ratio(df: pd.DataFrame) -> float:
    if df.empty: return 0.0
    total = df.size
    num = 0
    for v in df.to_numpy().ravel():
        try:
            float(str(v).replace(",","").replace("%",""))
            num += 1
        except Exception:
            pass
    return round(num / max(total,1), 4)

def run():
    rows = []
    for p in ROOT.rglob("text.jsonl"):
        parts = p.parts
        # .../processed/{ticker}/{year}/{form_acc}/text.jsonl
        ticker, year = parts[-4], parts[-3]
        form_acc = parts[-2]
        text_lines = sum(1 for _ in p.open(encoding="utf-8", errors="ignore"))
        tab_pq = p.parent / "tables.parquet"
        table_count = 0
        num_ratio = 0.0
        if tab_pq.exists():
            df = pd.read_parquet(tab_pq)
            table_count = df["__table_id__"].nunique() if "__table_id__" in df.columns else (1 if not df.empty else 0)
            num_ratio = numeric_ratio(df)
        rows.append({
            "ticker": ticker, "year": year, "form_acc": form_acc,
            "text_lines": text_lines, "table_count": table_count,
            "numeric_ratio": num_ratio
        })
    out_csv = QA / "qa_report.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()) if rows else 
                           ["ticker","year","form_acc","text_lines","table_count","numeric_ratio"])
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"✅ 输出：{out_csv}")

if __name__ == "__main__":
    run()
