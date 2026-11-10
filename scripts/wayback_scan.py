"""Find which daily ASDMA flood reports (2015-2020, Jun-Oct) the Internet Archive saved."""
import json, time, datetime as dt
from pathlib import Path
import requests
out = Path("data/asdma_archive/wayback_index.json")
found = json.loads(out.read_text()) if out.exists() else {}
for year in range(2015, 2021):
    d = dt.date(year, 6, 1)
    while d <= dt.date(year, 10, 31):
        key = d.strftime("%d.%m.%Y")
        if key not in found:
            url = f"asdma.gov.in/pdf/flood_report/{year}/Daily_Flood_Report_{key}.pdf"
            for attempt in range(3):
                try:
                    r = requests.get("https://archive.org/wayback/available", params={"url": url}, timeout=25)
                    s = r.json().get("archived_snapshots", {}).get("closest")
                    found[key] = s["url"] if s else None
                    break
                except Exception:
                    time.sleep(3)
            out.write_text(json.dumps(found, indent=0))
            time.sleep(0.5)
        d += dt.timedelta(days=1)
by_year = {}
for k, v in found.items():
    by_year.setdefault(k[-4:], [0, 0]); by_year[k[-4:]][0] += 1; by_year[k[-4:]][1] += v is not None
print({y: f"{a[1]} of {a[0]} days archived" for y, a in sorted(by_year.items())})
