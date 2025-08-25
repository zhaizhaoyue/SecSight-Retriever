from pathlib import Path
import csv, argparse, datetime as dt
from sec_edgar_downloader import Downloader

QA_DIR = Path("data/qa")
QA_DIR.mkdir(parents=True, exist_ok=True)

def year_bounds(y: int):
    return f"{y}-01-01", f"{y+1}-01-01"

def run(email: str, outdir: str, company_name: str = "FinanceLLMAssistant",
        limit_10k: int | None = None, limit_10q: int | None = None):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    dl = Downloader(company_name, email, str(out))

    ok_log  = (QA_DIR / "download_success.csv").open("a", encoding="utf-8", newline="")
    fail_log= (QA_DIR / "failed_downloads.csv").open("a", encoding="utf-8", newline="")
    import csv as _csv
    okw   = _csv.writer(ok_log)
    failw = _csv.writer(fail_log)


    if ok_log.tell() == 0:
        okw.writerow(["ticker","form","mode","years_or_limit","timestamp"])
    if fail_log.tell() == 0:
        failw.writerow(["ticker","form","year","error","timestamp"])

    ts = dt.datetime.now().isoformat(timespec="seconds")

    with open("data/companies.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("source","").upper() != "EDGAR") or (row.get("market","").upper() != "US"):
                continue
            ticker = row["ticker"].strip()
            forms  = [x.strip() for x in row["form_types"].split("|") if x.strip()]
            years  = [int(x) for x in row["years"].split("|") if x.strip().isdigit()]

            for form in forms:
                try:

                    if form == "10-K" and limit_10k:
                        print(f"[downloading] {ticker} {form} {y} ...")

                        dl.get("10-K", ticker, limit=limit_10k, download_details=True)
                        okw.writerow([ticker, form, f"limit={limit_10k}", "", ts])
                        continue
                    if form == "10-Q" and limit_10q:
                        print(f"[downloading] {ticker} {form} {y} ...")

                        dl.get("10-Q", ticker, limit=limit_10q, download_details=True)
                        okw.writerow([ticker, form, f"limit={limit_10q}", "", ts])
                        continue

                    for y in years:
                        after, before = year_bounds(y)
                        dl.get(form, ticker, after=after, before=before, download_details=True)
                        okw.writerow([ticker, form, "year-range", f"{y}", ts])
                except Exception as e:
                    yinfo = years if (not limit_10k and not limit_10q) else ""
                    failw.writerow([ticker, form, yinfo, repr(e), ts])
                    print(f"[fail] {ticker} {form}: {e}")

    ok_log.close(); fail_log.close()
    print("✅ download_from_csv: 完成（详见 data/qa/）")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", default="zhaizhaoyue520@gmail.com", help="zhaizhaoyue520@gmail.com")
    ap.add_argument("--outdir", default="data/raw_reports", help="下载目录")
    ap.add_argument("--company-name", default="FinanceLLMAssistant", help="UA公司名")
    ap.add_argument("--limit-10k", type=int, default=None, help="10-K 最近N份（可选）")
    ap.add_argument("--limit-10q", type=int, default=None, help="10-Q 最近N份（可选）")
    args = ap.parse_args()
    run(args.email, args.outdir, args.company_name, args.limit_10k, args.limit_10q)
