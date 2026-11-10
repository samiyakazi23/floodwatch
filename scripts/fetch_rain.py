"""Daily rainfall (ERA5 via Open-Meteo archive) at catchment points, 1997 onward.
Upstream Arunachal/Bhutan/Meghalaya rain reaches Assam's rivers days later,
which is where forecast skill beyond 1-2 days comes from."""
import json, time
from pathlib import Path
import requests

URL = "https://archive-api.open-meteo.com/v1/archive"
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
    for attempt in range(8):
        try:
            r = requests.get(URL, params=dict(latitude=lat, longitude=lon, start_date="1997-01-01", end_date="2026-11-05",
                                              daily="precipitation_sum", timezone="Asia/Kolkata"), timeout=120)
            if r.status_code == 200:
                d = r.json()["daily"]
                f.write_text(json.dumps({"lat": lat, "lon": lon, "time": d["time"], "rain": d["precipitation_sum"]}))
                print(name, "ok", len(d["time"]), flush=True)
                break
            print(name, r.status_code, r.text[:120], flush=True)
        except requests.RequestException as e:
            print(name, "err", e.__class__.__name__, flush=True)
        time.sleep(min(30 * 2 ** attempt, 600))
    time.sleep(20)
