"""Snap each named river station to the GloFAS grid cell carrying the main channel
(the cell with the highest flow in a small neighbourhood), then save its full daily history."""
import itertools, json, time
from pathlib import Path
import numpy as np, requests

URL = "https://flood-api.open-meteo.com/v1/flood"


def get(params, timeout=120):
    for attempt in range(6):
        try:
            r = requests.get(URL, params=params, timeout=timeout)
            if r.status_code == 429:
                raise requests.HTTPError("rate limited")
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.HTTPError, requests.Timeout) as e:
            wait = 5 * 2 ** attempt
            print(f"  retry in {wait}s ({e.__class__.__name__})", flush=True)
            time.sleep(wait)
    raise RuntimeError("giving up")
# Central Water Commission stations on the Brahmaputra and major tributaries (approx. coordinates)
STATIONS = {
    "Dibrugarh":   (27.48, 94.91, "Brahmaputra"),
    "Neamatighat": (26.82, 94.25, "Brahmaputra"),
    "Tezpur":      (26.60, 92.80, "Brahmaputra"),
    "Guwahati":    (26.18, 91.74, "Brahmaputra"),
    "Goalpara":    (26.17, 90.62, "Brahmaputra"),
    "Dhubri":      (26.02, 89.98, "Brahmaputra"),
    "Badatighat":  (27.08, 94.10, "Subansiri"),
    "Jia Bharali": (26.80, 92.85, "Jia Bharali"),
    "Numaligarh":  (26.60, 93.73, "Dhansiri"),
    "Dharamtul":   (26.20, 92.38, "Kopili"),
    "Beki Road":   (26.40, 90.95, "Beki/Manas"),
    "Silchar":     (24.82, 92.80, "Barak"),
}
out = Path("data/river"); out.mkdir(parents=True, exist_ok=True)
meta = json.loads((out / "stations.json").read_text()) if (out / "stations.json").exists() else {}
for name, (lat, lon, river) in STATIONS.items():
    if name in meta and (out / f"{name}.json").exists():
        continue
    offs = np.round(np.arange(-0.15, 0.151, 0.05), 2)
    pts = [(round(lat + a, 3), round(lon + b, 3)) for a, b in itertools.product(offs, offs)]
    res = get(dict(latitude=",".join(str(p[0]) for p in pts), longitude=",".join(str(p[1]) for p in pts),
                   daily="river_discharge", start_date="2022-05-15", end_date="2022-08-31"))
    best = max(res, key=lambda d: max([x for x in d["daily"]["river_discharge"] if x is not None] or [0]))
    clat, clon = best["latitude"], best["longitude"]
    time.sleep(1)
    h = get(dict(latitude=clat, longitude=clon, daily="river_discharge", start_date="1984-01-01", end_date="2026-11-08"))
    (out / f"{name}.json").write_text(json.dumps({"time": h["daily"]["time"], "q": h["daily"]["river_discharge"]}))
    q = [x for x in h["daily"]["river_discharge"] if x is not None]
    meta[name] = {"river": river, "station_lat": lat, "station_lon": lon, "cell_lat": clat, "cell_lon": clon, "n_days": len(q), "max_q": max(q)}
    print(name, meta[name], flush=True)
    (out / "stations.json").write_text(json.dumps(meta, indent=2))
    time.sleep(1)
(out / "stations.json").write_text(json.dumps(meta, indent=2))
