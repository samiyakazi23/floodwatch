"""Collect CWC daily flood situation reports (monsoon 2021-2026) and save their text.

CWC publishes a daily PDF listing the gauge sites in flood (Above Normal / Severe / Extreme).
File names changed over the years, so several patterns are tried per day.
Output: data/cwc/text/YYYY-MM-DD.txt (pdftotext -layout) and data/cwc/index.csv
"""
import csv, datetime as dt, subprocess, time, pathlib, urllib.request

BASE = "https://cwc.gov.in/sites/default/files/"
OUT = pathlib.Path("data/cwc/text"); OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (research; FloodWatch flood-forecast verification)"}

def patterns(d: dt.date):
    a, b, c = d.strftime("%d%m%Y"), d.strftime("%d.%m.%Y"), d.strftime("%d%m%y")
    for stem in (f"dfsitrepca-{a}", f"dfsitrepca{a}", f"dfsitrepca-{b}", f"dfsitrepca{b}", f"dfsitrepca{c}", f"dfsitrepca-{c}"):
        yield stem + ".pdf"
        yield stem + "-1.pdf"
        yield stem + "-0.pdf"

def get(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
            data = r.read()
            return data if data[:4] == b"%PDF" else None
    except Exception:
        return None

rows = []
for year in range(2021, 2027):
    d, end = dt.date(year, 5, 1), dt.date(year, 10, 31)
    while d <= end:
        hit = None
        for name in patterns(d):
            pdf = get(BASE + name)
            time.sleep(0.3)
            if pdf:
                hit = name
                p = pathlib.Path("/tmp/r.pdf"); p.write_bytes(pdf)
                txt = subprocess.run(["pdftotext", "-layout", str(p), "-"], capture_output=True, text=True).stdout
                (OUT / f"{d}.txt").write_text(txt)
                break
        rows.append({"date": str(d), "file": hit or ""})
        print(d, hit, flush=True)
        d += dt.timedelta(days=1)
with open("data/cwc/index.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["date", "file"]); w.writeheader(); w.writerows(rows)
print("found", sum(1 for r in rows if r["file"]), "of", len(rows))
