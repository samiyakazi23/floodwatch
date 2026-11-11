"""Collect CWC daily flood situation reports (monsoon 2021-2026) and save their text.

CWC publishes a daily PDF listing the gauge sites in flood (Above Normal / Severe / Extreme).
File names changed over the years, so several patterns are tried per day.
Output: data/cwc/text/YYYY-MM-DD.txt (pdftotext -layout) and data/cwc/index.csv
"""
import csv, datetime as dt, subprocess, time, pathlib, urllib.request, urllib.error, sys, shutil, tempfile

BASE = "https://cwc.gov.in/sites/default/files/"
OUT = pathlib.Path("data/cwc/text"); OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (research; FloodWatch flood-forecast verification)"}

def patterns(d: dt.date):
    a, b, c = d.strftime("%d%m%Y"), d.strftime("%d.%m.%Y"), d.strftime("%d%m%y")
    # dd.mm.yyyy first: the form CWC used in 2024 (dfsitrepca-08.09.2024.pdf, dfsra-21.07.2024.pdf)
    stems = []
    for pre in ("dfsitrepca", "dfsra"):
        stems += [f"{pre}-{b}", f"{pre}{b}", f"{pre}-{a}", f"{pre}{a}", f"{pre}-{c}", f"{pre}{c}"]
    for stem in stems:
        yield stem + ".pdf"
        yield stem + "-1.pdf"
        yield stem + "-0.pdf"

errors = {}
first_error = {}

try:  # python.org builds on macOS ship without CA certificates; certifi fixes CERTIFICATE_VERIFY_FAILED
    import ssl, certifi
    CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    CTX = None

def pdf_text(pdf: bytes) -> str:
    """pdftotext -layout if installed (CI), otherwise pypdf (pip install pypdf) so no Homebrew is needed."""
    if shutil.which("pdftotext"):
        p = pathlib.Path(tempfile.gettempdir()) / "cwc_r.pdf"; p.write_bytes(pdf)
        return subprocess.run(["pdftotext", "-layout", str(p), "-"], capture_output=True, text=True).stdout
    import io, pypdf
    pages = pypdf.PdfReader(io.BytesIO(pdf)).pages
    return "\n".join(pg.extract_text(extraction_mode="layout") or "" for pg in pages)

def get(url):
    """Return PDF bytes, or None. Non-404 failures are counted so a blocked run is visible."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30, context=CTX) as r:
            data = r.read()
            if data[:4] == b"%PDF":
                return data
            errors["not a PDF (HTML page?)"] = errors.get("not a PDF (HTML page?)", 0) + 1
    except urllib.error.HTTPError as e:
        if e.code != 404:
            errors[f"HTTP {e.code}"] = errors.get(f"HTTP {e.code}", 0) + 1
    except Exception as e:
        k = type(e).__name__
        errors[k] = errors.get(k, 0) + 1
        first_error.setdefault(k, str(getattr(e, "reason", e)))
    return None

if len(sys.argv) > 1 and sys.argv[1] == "--test":  # quick check on one report that is known to exist
    pdf = get(BASE + "dfsitrepca-08.09.2024.pdf")
    if pdf:
        t = pdf_text(pdf)
        print("OK: downloaded and read a CWC report.", "Rows mention Assam:", "Assam" in t)
    else:
        print("FAILED:", errors, first_error)
    sys.exit(0 if pdf else 1)

rows = []
YEARS = [int(sys.argv[1])] if len(sys.argv) > 1 else range(2021, 2027)
for year in YEARS:
    d, end = dt.date(year, 5, 1), dt.date(year, 10, 31)
    while d <= end:
        hit = None
        for name in patterns(d):
            pdf = get(BASE + name)
            time.sleep(0.15)
            if pdf:
                hit = name
                txt = pdf_text(pdf)
                (OUT / f"{d}.txt").write_text(txt)
                break
        rows.append({"date": str(d), "file": hit or ""})
        print(d, hit, flush=True)
        d += dt.timedelta(days=1)
with open(f"data/cwc/index_{YEARS[0] if len(YEARS) == 1 else 'all'}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["date", "file"]); w.writeheader(); w.writerows(rows)
found = sum(1 for r in rows if r["file"])
print("found", found, "of", len(rows))
print("non-404 failures:", errors or "none", first_error or "")
if found == 0:
    sys.exit("No reports downloaded - CWC is probably blocking this machine. Run it from an Indian network.")
