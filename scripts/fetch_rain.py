"""Daily rainfall 1997-2026 at catchment points from NASA POWER (MERRA-2 / IMERG-based PRECTOTCORR, mm/day).
Upstream Arunachal / Bhutan / Meghalaya rain reaches Assam's rivers days later; local rain drives
floods the main-river gauges miss."""
import json, time
from pathlib import Path
import requests

URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POINTS = {
    "Pasighat": (28.07, 95.33), "Along": (28.17, 94.80), "Ziro": (27.55, 93.83), "Itanagar": (27.08, 93.60),
    "Tawang": (27.59, 91.87), "Gelephu": (26.90, 90.50), "Shillong": (25.58, 91.89), "Haflong": (25.17, 93.02),
    "Dibrugarh": (27.48, 94.91), "Jorhat": (26.75, 94.20), "Tezpur": (26.63, 92.80), "Guwahati": (26.14, 91.74),
    "Barpeta": (26.32, 91.00), "Dhubri": (26.02, 89.98), "Silchar": (24.82, 92.80),
}
out = Path("data/rain"); out.mkdir(parents=True, exist_ok=True)
for name, (lat, lon) in POINTS.items():
    f = out / f"{name}.json"
    if f.exists():
        continue
    for attempt in range(5):
        try:
            r = requests.get(URL, params=dict(parameters="PRECTOTCORR", community="AG", latitude=lat, longitude=lon,
                                              start="19970101", end="20261108", format="JSON"), timeout=180)
            r.raise_for_status()
            d = r.json()["properties"]["parameter"]["PRECTOTCORR"]
            f.write_text(json.dumps({"lat": lat, "lon": lon, "time": [f"{k[:4]}-{k[4:6]}-{k[6:]}" for k in d], "rain": [None if v < 0 else v for v in d.values()]}))
            print(name, "ok", len(d), flush=True)
            break
        except Exception as e:
            print(name, "retry", e.__class__.__name__, flush=True); time.sleep(10 * (attempt + 1))
    time.sleep(1)
