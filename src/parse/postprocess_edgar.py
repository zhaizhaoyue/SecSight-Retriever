from pathlib import Path
import shutil, re, json
from bs4 import BeautifulSoup

DEFAULT_BASE_DIR = Path("data/raw_reports/sec-edgar-filings")
DEFAULT_OUTPUT_DIR = Path("data/raw_reports/standard")


def pick_main_html(folder: Path):
    """Pick the primary HTML file from a filing directory."""
    htmls = list(folder.glob("*.htm")) + list(folder.glob("*.html"))
    htmls = [p for p in htmls if "filing-details" not in p.name.lower()]
    if htmls:
        htmls.sort(key=lambda p: p.stat().st_size, reverse=True)
        return htmls[0]

    fd = folder / "filing-details.html"
    if fd.exists():
        return fd

    fs = folder / "full-submission.txt"
    if fs.exists():
        return fs

    any_html = list(folder.glob("*.htm")) + list(folder.glob("*.html"))
    if any_html:
        return any_html[0]

    any_txt = list(folder.glob("*.txt"))
    if any_txt:
        return any_txt[0]

    return None


def get_filing_date_and_year(folder: Path, main_file: Path):
    """Extract filing date and year."""
    meta = folder / "metadata.json"
    if meta.exists():
        try:
            m = json.loads(meta.read_text(encoding="utf-8", errors="ignore"))
            fd = m.get("filedAt") or m.get("filingDate")
            if fd and re.match(r"\d{4}-\d{2}-\d{2}", fd):
                return fd, fd[:4]
        except Exception:
            pass

    if main_file and main_file.suffix.lower() in {".htm", ".html"}:
        try:
            soup = BeautifulSoup(main_file.read_text(encoding="utf-8", errors="ignore"), "lxml")
            text = soup.get_text(" ", strip=True)
            m = re.search(r"Filing Date[:\s]+(\d{4}-\d{2}-\d{2})", text)
            if m:
                fd = m.group(1)
                return fd, fd[:4]
        except Exception:
            pass

    parts = folder.name.split("-")
    if len(parts) >= 3 and len(parts[1]) == 2:
        return None, "20" + parts[1]

    return None, "0000"


def pick_all_xmls(folder: Path):
    """Return all XML/XBRL documents in the filing directory, excluding FilingSummary.xml."""
    xmls = list(folder.glob("*.xml")) + list(folder.glob("*.xbrl"))
    return [x for x in xmls if x.name.lower() != "filingsummary.xml"]


def run(base_dir: Path | str = DEFAULT_BASE_DIR, out_dir: Path | str = DEFAULT_OUTPUT_DIR) -> int:
    """Normalize raw EDGAR download structure into the standard directory."""
    base = Path(base_dir)
    out = Path(out_dir)

    if not base.exists():
        print(f"[error] not found: {base}")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    count = 0

    for ticker_dir in base.iterdir():
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name

        for form_dir in ticker_dir.iterdir():
            if not form_dir.is_dir():
                continue
            form = form_dir.name

            for acc_dir in form_dir.iterdir():
                if not acc_dir.is_dir():
                    continue

                # 1. Copy the primary HTML file
                main_file = pick_main_html(acc_dir)
                if not main_file:
                    print(f"[skip] no main file: {acc_dir}")
                    continue

                filing_date, year = get_filing_date_and_year(acc_dir, main_file)
                accno = acc_dir.name.replace("/", "_")

                ext = main_file.suffix.lower().lstrip(".") or "txt"
                dst = out / f"US_{ticker}_{year}_{form}_{accno}.{ext}"
                shutil.copy2(main_file, dst)
                count += 1
                print(f"[ok-html] {dst}")

                # 2. Copy XML/XBRL files (excluding FilingSummary)
                for xml_file in pick_all_xmls(acc_dir):
                    ext_xml = xml_file.suffix.lower().lstrip(".") or "xml"
                    dst_xml = out / f"US_{ticker}_{year}_{form}_{accno}_{xml_file.stem}.{ext_xml}"
                    shutil.copy2(xml_file, dst_xml)
                    count += 1
                    print(f"[ok-xml] {dst_xml}")

    print(f"[ok] copied and renamed {count} file(s)")
    return count


if __name__ == "__main__":
    run()
