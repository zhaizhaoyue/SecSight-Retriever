from pathlib import Path
import re, csv

STD = Path("data/raw_reports/standard")
QA  = Path("data/qa"); QA.mkdir(parents=True, exist_ok=True)

def parse_std_name(p: Path):
    # US_{ticker}_{year}_{form}_{accno}.{ext}
    m = re.match(r"US_(.+?)_(\d{4})_(10-[KQ])_(.+)\.(.+)$", p.name, re.I)
    if not m: 
        return None
    ticker, year, form, accno, ext = m.groups()
    return dict(ticker=ticker, year=int(year), form=form, accno=accno, ext=ext)

def run():
    rows = []
    for f in STD.glob("*.*"):
        info = parse_std_name(f)
        if not info: 
            continue
        rows.append({**info, "path": str(f)})
    out_csv = QA / "download_report.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["ticker","form","year","ext","accno","path"])
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["ticker"], x["form"], -x["year"])):
            w.writerow(r)
    print(f"✅ 输出：{out_csv}")

if __name__ == "__main__":
    run()
