"""Daily live forecast: fetch the latest river flow, forecast 1-7 days ahead, write docs/data/live.json.

Runs every morning on GitHub Actions (.github/workflows/daily.yml). Uses only flow
observed up to today, the same inputs the backtested model saw.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from floodwatch.forecast import build_features  # noqa: E402

URL = "https://flood-api.open-meteo.com/v1/flood"
HISTORY_DAYS = 45  # enough for the 30-day rolling features


def fetch(lat: float, lon: float, start: str, end: str) -> pd.Series:
    for attempt in range(5):
        try:
            r = requests.get(URL, params=dict(latitude=lat, longitude=lon, daily="river_discharge", start_date=start, end_date=end), timeout=60)
            r.raise_for_status()
            d = r.json()["daily"]
            return pd.Series(d["river_discharge"], index=pd.to_datetime(d["time"]), dtype=float)
        except requests.RequestException:
            time.sleep(5 * 2 ** attempt)
    raise RuntimeError(f"could not fetch flow at {lat},{lon}")


def main() -> None:
    meta = json.loads((ROOT / "models/forecast_meta.json").read_text())
    stations = json.loads((ROOT / "data/river/stations.json").read_text())
    today = date.today()
    start = (today - timedelta(days=HISTORY_DAYS)).isoformat()
    # "today" in GloFAS terms = the latest day with a value; the API also returns forecast days, which we ignore
    flows = pd.DataFrame({s: fetch(stations[s]["cell_lat"], stations[s]["cell_lon"], start, today.isoformat()) for s in meta["stations"]})
    flows = flows[flows.index <= pd.Timestamp(today)].interpolate(limit=3).dropna(how="all")
    feats = build_features(flows[meta["stations"]])
    last = feats["date"].max()
    now = feats[feats.date == last].copy()
    thr = pd.Series(meta["thresholds"])
    now["ratio"] = np.log(now["q_raw"] / now["station_name"].map(thr))
    now["flood_now"] = (now["q_raw"] > now["station_name"].map(thr)).astype(int)
    out = {"issued": str(last.date()), "generated": today.isoformat(), "trained_through": meta["trained_through"], "stations": {}}
    for _, row in now.iterrows():
        s = row["station_name"]
        rec = {"q": round(float(row["q_raw"])), "thr": round(float(thr[s])), "p": {}, "cut": {}}
        for h in meta["horizons"]:
            model = lgb.Booster(model_file=str(ROOT / f"models/forecast_h{h}.txt"))
            rec["p"][str(h)] = round(float(model.predict(pd.DataFrame([row[meta["features"]]]).astype(float))[0]), 3)
            rec["cut"][str(h)] = round(meta["cutoffs"][str(h)], 3)
        out["stations"][s] = rec
    dest = ROOT / "docs/data/live.json"
    dest.write_text(json.dumps(out, indent=1))
    # keep an append-only log of every issued forecast, for scoring the 2027 monsoon later
    log = ROOT / "docs/data/live_log.jsonl"
    with open(log, "a") as f:
        f.write(json.dumps(out) + "\n")
    warned = [s for s, r in out["stations"].items() if r["p"]["3"] >= r["cut"]["3"]]
    print(f"forecast issued for {out['issued']}; 3-day warnings at: {', '.join(warned) or 'none'}")


if __name__ == "__main__":
    main()
