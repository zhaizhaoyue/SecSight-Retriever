from pathlib import Path
import csv, argparse, datetime as dt
from typing import Optional
import sys, platform, time, random, re, json
import requests
from bs4 import BeautifulSoup
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from sec_edgar_downloader import Downloader

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
QA_DIR = ROOT_DIR / "data" / "qa"
QA_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_OUTDIR = ROOT_DIR / "data" / "raw_reports"
COMPANIES_CSV = ROOT_DIR / "data" / "companies.csv"

USER_AGENT_TMPL = "{company} ({email})"

TICKER_CIK_HARDCODE = {
    "BRK-A": "0001067983",
    "BRK.B": "0001067983",
    "BRK-B": "0001067983",
    "BRK.A": "0001067983",
}

AGENT_CIKS = {
    "0000950170",  # common filing agent
    "0001564590",  # common filing agent seen in logs
    "0001193125",  # common filing agent
    "0001047469",  # common filing agent
}

# ------------------------------
# Helper utilities for HTTP sessions and requests
# ------------------------------
def _requests_session():
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s

def _ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def _fetch_json(url: str, headers: dict, session=None):
    session = session or _requests_session()
    try:
        r = session.get(url, headers=headers, timeout=30)
        print(f"[http] GET {url} -> {r.status_code}")
        if r.status_code == 200:
            return r.json()
        else:
            body = r.text[:200] if r.text else ""
            if body:
                print(f"[http] body[:200]: {body}")
    except Exception as e:
        print(f"[http] EXC {url}: {e}")
    return None

def _download_file(url: str, dst: Path, headers: dict, session=None) -> bool:
    session = session or _requests_session()
    try:
        r = session.get(url, headers=headers, timeout=60)
        print(f"[http] GET {url} -> {r.status_code} -> {dst}")
        if r.status_code == 200:
            _ensure_dir(dst)
            dst.write_bytes(r.content)
            return True
    except Exception as e:
        print(f"[http] EXC {url}: {e}")
    return False

# ------------------------------
# Metadata helpers and directory parsing
# ------------------------------
def _read_metadata(folder: Path) -> dict:
    mfile = folder / "metadata.json"
    if mfile.exists():
        try:
            return json.loads(mfile.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return {}
    return {}

# Cache SEC ticker-to-CIK mapping once
_SEC_TICKERS_CACHE = None

def _resolve_cik_from_ticker(ticker: str, company_name: str, email: str) -> Optional[str]:
    global _SEC_TICKERS_CACHE
    if not ticker:
        return None
    key0 = ticker.upper()
    if key0 in TICKER_CIK_HARDCODE:
        return TICKER_CIK_HARDCODE[key0]
    key = key0.replace("-", ".")
    if key in TICKER_CIK_HARDCODE:
        return TICKER_CIK_HARDCODE[key]

    # Check hard-coded mapping first
    t_upper = ticker.upper()
    if t_upper in TICKER_CIK_HARDCODE:
        return TICKER_CIK_HARDCODE[t_upper]
    t_norm = t_upper.replace("-", ".")
    if t_norm in TICKER_CIK_HARDCODE:
        return TICKER_CIK_HARDCODE[t_norm]

    try:
        if _SEC_TICKERS_CACHE is None:
            sess = _requests_session()
            headers = {
                "User-Agent": USER_AGENT_TMPL.format(company=company_name, email=email),
                "Accept": "application/json,*/*"
            }
            # Try official company_tickers.json first
            url = "https://www.sec.gov/files/company_tickers.json"
            j = _fetch_json(url, headers=headers, session=sess)
            if j:
                _SEC_TICKERS_CACHE = {}
                for v in j.values():
                    _SEC_TICKERS_CACHE[str(v.get("ticker","")).upper()] = str(v.get("cik_str","")).zfill(10)
                    _SEC_TICKERS_CACHE[str(v.get("ticker","")).upper().replace("-", ".")] = str(v.get("cik_str","")).zfill(10)
            else:
                _SEC_TICKERS_CACHE = {}

        # Index with both raw and normalized keys
        for k in (t_upper, t_norm):
            v = _SEC_TICKERS_CACHE.get(k)
            if v:
                return v
    except Exception:
        pass
    return None


def _extract_issuer_cik(meta: dict, folder: Path, *, ticker: Optional[str], company_name: str, email: str) -> Optional[str]:
    """Resolve issuer CIKs while excluding common filing agents; fall back to ticker lookup when required."""
    # 1) metadata.json when present
    candidate_paths = [
        ("companyData","cik"),
        ("companyInfo","cik"),
        ("entity","cik"),
        ("primaryIssuer","cik"),
        ("issuerCik",),
        ("cik",),
        ("cik_str",),
        ("request","cik"),
    ]
    for path in candidate_paths:
        try:
            v = meta
            for k in path:
                v = v[k]
            v = str(v).strip().zfill(10)
            if v.isdigit() and v not in AGENT_CIKS:
                return v
        except Exception:
            pass

    # 2) Parse filing-details.html for metadata
    try:
        htm = folder / "filing-details.html"
        if htm.exists():
            txt = htm.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"CIK[:\s]+0*([0-9]{1,10})", txt, flags=re.I)
            if m:
                v = m.group(1).zfill(10)
                if v not in AGENT_CIKS:
                    return v
    except Exception:
        pass

    # 3) Reverse lookup via ticker (most reliable)
    if ticker:
        v = _resolve_cik_from_ticker(ticker, company_name, email)
        if v and v not in AGENT_CIKS:
            return v

    # 4) Fallback: use accession prefix unless blacklisted
    accno = str(meta.get("accessionNumber") or meta.get("accession") or folder.name).strip()
    m = re.match(r"^(\d{10})-\d{2}-\d{6}$", accno)
    if m:
        cand = m.group(1).zfill(10)
        if cand not in AGENT_CIKS:
            return cand

    return None



def _cik_from_accno(accno: str) -> Optional[str]:
    # "0000320193-24-000010" -> "0000320193"
    m = re.match(r"^(\d{10})-\d{2}-\d{6}$", accno)
    return m.group(1) if m else None

def _list_dir_html(url: str, headers: dict, session=None):
    session = session or _requests_session()
    try:
        r = session.get(url, headers=headers, timeout=30)
        print(f"[http] DIR {url} -> {r.status_code}")
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "lxml")
        items = []
        for a in soup.select("a[href]"):
            href = a.get("href", "")
            if not href or href in ("./", "../") or href.startswith("?"):
                continue
            items.append((href, href.endswith("/")))
        return items
    except Exception as e:
        print(f"[http] EXC dir-html {url}: {e}")
        return []

def _looks_like_xbrl(name: str) -> bool:
    n = name.lower()
    if n.endswith((".xml", ".xsd")):
        # Prefer keeping everything and prioritize keyword matches
        return True
    return False

def _count_xml(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in {".xml", ".xsd"})

def _harvest_filing_dirs(root: Path, ticker: str, form: str) -> list[Path]:
    base = root / "sec-edgar-filings" / ticker / form
    return [p for p in base.glob("*") if p.is_dir()]

# ------------------------------
# XBRL downloads with index.json and HTML fallback
# ------------------------------
def _download_xbrl_for_folder(folder: Path, company_name: str, email: str, sleep_sec: float = 0.4, ticker: Optional[str] = None) -> int:
    """Download XBRL assets for a filing directory, using index.json first and falling back to crawling HTML listings."""
    meta  = _read_metadata(folder)
    accno = (meta.get("accessionNumber") or meta.get("accession") or folder.name).strip()

    cik = _extract_issuer_cik(meta, folder, ticker=ticker, company_name=company_name, email=email)
    if not cik:
        print(f"[xbrl] skip (no issuer CIK): folder={folder.name} accno={accno} ticker={ticker}")
        return 0

    headers = {
        "User-Agent": USER_AGENT_TMPL.format(company=company_name, email=email),
        "Accept": "application/json, text/html, */*",
        "Connection": "close",
    }
    session   = _requests_session()
    acc_key   = accno.replace("-", "")
    base_root = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_key}/"

    print(f"[xbrl] issuer_cik={cik} accno={accno} base={base_root}")
    print(f"[xbrl] index: {base_root}index.json")

    idx = _fetch_json(base_root + "index.json", headers=headers, session=session)
    targets: list[tuple[str, bool]] = []

    def walk_json(items, prefix=""):
        for it in items:
            name = it.get("name","")
            href = it.get("href", name)
            if not name and not href:
                continue
            is_dir = (it.get("type") == "dir") or str(href).endswith("/")
            if is_dir:
                sub = _fetch_json(base_root + href + "index.json", headers=headers, session=session)
                if sub and "directory" in sub:
                    walk_json(sub["directory"].get("item", []), prefix + href)
                else:
                    for h, d in _list_dir_html(base_root + href, headers=headers, session=session):
                        if d:
                            for h2, d2 in _list_dir_html(base_root + href + h, headers=headers, session=session):
                                targets.append((href + h + h2, d2))
                        else:
                            targets.append((href + h, False))
            else:
                targets.append((href, False))

    if idx and "directory" in idx and "item" in idx["directory"]:
        walk_json(idx["directory"]["item"])
    else:
        print("[xbrl] index.json unavailable -> fallback to HTML listing")
        def walk_html(dir_url: str, rel=""):
            for href, is_dir in _list_dir_html(dir_url, headers=headers, session=session):
                if is_dir:
                    walk_html(dir_url + href, rel + href)
                else:
                    targets.append((rel + href, False))
        walk_html(base_root, "")

    # Download .xml/.xsd assets
    count = 0
    for rel, is_dir in targets:
        if is_dir:
            continue
        name = rel.split("/")[-1].lower()
        if name.endswith((".xml", ".xsd")):
            url = base_root + rel
            dst = folder / rel
            if not dst.exists():
                if _download_file(url, dst, headers=headers, session=session):
                    count += 1
                    time.sleep(sleep_sec + random.random() * 0.2)
    print(f"[xbrl] downloaded {count} xml/xsd -> {folder}")
    return count


# ------------------------------
# Misc helpers
# ------------------------------
def year_bounds(y: int):
    return f"{y}-01-01", f"{y+1}-01-01"

def _supports_builtin_xbrl(dl: Downloader) -> bool:
    try:
        return "download_xbrl" in dl.get.__code__.co_varnames
    except Exception:
        return False

# ------------------------------
# Main workflow
# ------------------------------
def run(email: str,
        outdir: str,
        company_name: str = "FinanceLLMAssistant",
        limit_10k: int | None = None,
        limit_10q: int | None = None,
        xbrl: bool = True,
        sleep: float = 0.4,
        companies_csv: Path | str = COMPANIES_CSV):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    companies_path = Path(companies_csv)
    if not companies_path.exists():
        raise FileNotFoundError(f"companies csv not found: {companies_path}")
    print("[info] CWD:", Path.cwd())
    print("[info] Python:", sys.executable, platform.python_version())
    print("[info] download root:", out.resolve())

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

    supports_builtin_xbrl = _supports_builtin_xbrl(dl)
    with open(companies_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("source","").upper() != "EDGAR") or (row.get("market","").upper() != "US"):
                continue
            ticker = row["ticker"].strip()
            forms  = [x.strip() for x in row["form_types"].split("|") if x.strip()]
            years  = [int(x) for x in row["years"].split("|") if x.strip().isdigit()]

            for form in forms:
                try:
                    # ---- Limited mode ----
                    if form == "10-K" and limit_10k:
                        print(f"[downloading] {ticker} {form} last {limit_10k} ...")
                        if xbrl and supports_builtin_xbrl:
                            dl.get("10-K", ticker, limit=limit_10k, download_details=True, download_xbrl=True)
                        else:
                            dl.get("10-K", ticker, limit=limit_10k, download_details=True)
                        okw.writerow([ticker, form, f"limit={limit_10k}", "", ts])

                        # Second-stage validation plus fallback
                        for folder in _harvest_filing_dirs(out, ticker, "10-K"):
                            xmln = _count_xml(folder)
                            if xbrl and xmln == 0:
                                _download_xbrl_for_folder(folder, company_name, email, sleep_sec=sleep, ticker=ticker)
                                xmln = _count_xml(folder)
                            print(f"[ok] wrote -> {folder} (xml/xsd={xmln})")
                        continue

                    if form == "10-Q" and limit_10q:
                        print(f"[downloading] {ticker} {form} last {limit_10q} ...")
                        if xbrl and supports_builtin_xbrl:
                            dl.get("10-Q", ticker, limit=limit_10q, download_details=True, download_xbrl=True)
                        else:
                            dl.get("10-Q", ticker, limit=limit_10q, download_details=True)
                        okw.writerow([ticker, form, f"limit={limit_10q}", "", ts])

                        for folder in _harvest_filing_dirs(out, ticker, "10-Q"):
                            xmln = _count_xml(folder)
                            if xbrl and xmln == 0:
                                _download_xbrl_for_folder(folder, company_name, email, sleep_sec=sleep, ticker=ticker)
                                xmln = _count_xml(folder)
                            print(f"[ok] wrote -> {folder} (xml/xsd={xmln})")
                        continue

                    # ---- Year-range mode ----
                    for y in years:
                        after, before = year_bounds(y)
                        print(f"[downloading] {ticker} {form} in {y} ...")
                        if xbrl and supports_builtin_xbrl:
                            dl.get(form, ticker, after=after, before=before, download_details=True, download_xbrl=True)
                        else:
                            dl.get(form, ticker, after=after, before=before, download_details=True)
                        okw.writerow([ticker, form, "year-range", f"{y}", ts])

                        for folder in _harvest_filing_dirs(out, ticker, form):
                            xmln = _count_xml(folder)
                            if xbrl and xmln == 0:
                                _download_xbrl_for_folder(folder, company_name, email, sleep_sec=sleep, ticker=ticker)
                                xmln = _count_xml(folder)
                            print(f"[ok] wrote -> {folder} (xml/xsd={xmln})")

                except Exception as e:
                    yinfo = years if (not limit_10k and not limit_10q) else ""
                    failw.writerow([ticker, form, yinfo, repr(e), ts])
                    print(f"[fail] {ticker} {form}: {e}")

    ok_log.close(); fail_log.close()
    print("[ok] download_from_csv: completed (see data/qa/)")

# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", default="zhaizhaoyue520@gmail.com", help="Contact email used in the User-Agent header")
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Download root directory")
    ap.add_argument("--company-name", default="FinanceLLMAssistant", help="Company name to embed in User-Agent")
    ap.add_argument("--limit-10k", type=int, default=None, help="Download the last N 10-K filings")
    ap.add_argument("--limit-10q", type=int, default=None, help="Download the last N 10-Q filings")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--xbrl", dest="xbrl", action="store_true", help="Enable XBRL download")
    g.add_argument("--no-xbrl", dest="xbrl", action="store_false", help="Disable XBRL download")
    ap.set_defaults(xbrl=True)

    ap.add_argument("--sleep", type=float, default=0.4, help="Minimum sleep between XBRL downloads (seconds)")
    args = ap.parse_args()

    run(
        args.email,
        args.outdir,
        args.company_name,
        args.limit_10k,
        args.limit_10q,
        xbrl=args.xbrl,
        sleep=args.sleep,
    )
