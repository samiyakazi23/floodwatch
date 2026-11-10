"""Validate forecasts against REAL flood impacts: ASDMA's district reports.

Ground truth: population reported affected by floods, per district and month
(ASDMA FRIMS reports, compiled by CivicDataLab's IDS-DRR project), Apr 2021 - Sep 2026.

Unit of evaluation: district-month in the monsoon (May-Oct). A district-month
counts as "flooded" when ASDMA reported at least MIN_AFFECTED people affected.

A forecast "warned" that district-month if, on any day of the month, the 3-day
forecast for the district's river station crossed its alert cut-off.

Baselines:
  climatology   flag every June and July (the usual peak months)
  river_above   the river was actually above warning level that month: an
                observation, not a forecast, so a ceiling for river-only methods
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

MIN_AFFECTED = 1000
H = 3
RENAME = {"KAMRUP METRO": "Kamrup Metropolitan", "BAJALI": "Barpeta", "TAMULPUR": "Baksa"}  # new districts -> parent
BARAK_VALLEY = {"Cachar", "Karimganj", "Hailakandi"}
# geography the nearest-centroid rule gets wrong: Majuli is the river island opposite Jorhat;
# Biswanath sits on the north bank between Tezpur and the Subansiri
OVERRIDES = {"Majuli": "Neamatighat", "Biswanath": "Tezpur"}


def centroid(geom: dict) -> tuple[float, float]:
    """Mean of the outer-ring vertices of every polygon part (good enough to pick the nearest river station)."""
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    pts = np.vstack([np.asarray(poly[0], dtype=float)[:, :2] for poly in polys])
    return float(pts[:, 1].mean()), float(pts[:, 0].mean())


def district_station_map() -> dict[str, str]:
    stations = json.loads(Path("data/river/stations.json").read_text())
    geo = json.loads(Path("data/geo/assam_districts.geojson").read_text())
    out = {}
    for f in geo["features"]:
        name = f["properties"]["dtname"]
        lat, lon = centroid(f["geometry"])
        if name in BARAK_VALLEY:
            out[name] = "Silchar"
            continue
        if name in OVERRIDES:
            out[name] = OVERRIDES[name]
            continue
        cands = {s: m for s, m in stations.items() if s != "Silchar"}
        out[name] = min(cands, key=lambda s: math.hypot(cands[s]["station_lat"] - lat, (cands[s]["station_lon"] - lon) * math.cos(math.radians(lat))))
    return out


def load_truth() -> pd.DataFrame:
    d = pd.read_csv("data/asdma/district_month_affected.csv")
    d["district"] = d["district"].map(lambda x: RENAME.get(x, x.title()))
    d = d.groupby(["district", "year", "month"], as_index=False)["pop_affected"].sum()
    return d[d.month.between(5, 10)]


def score(y: np.ndarray, w: np.ndarray) -> dict:
    tp = int((y & w).sum()); fp = int((~y & w).sum()); fn = int((y & ~w).sum())
    return {"hit_rate": round(tp / max(tp + fn, 1), 3), "false_alarm_ratio": round(fp / max(tp + fp, 1), 3),
            "csi": round(tp / max(tp + fp + fn, 1), 3), "flooded": int(y.sum()), "warned": int(w.sum())}


def validate(preds_path: str = "reports/backtest_predictions.csv.gz") -> dict:
    preds = pd.read_csv(preds_path, parse_dates=["date"])
    preds["year"], preds["month"] = preds.date.dt.year, preds.date.dt.month
    preds["warn"] = preds[f"p_{H}"] >= preds[f"cut_{H}"]
    preds["above"] = preds["q_raw"] > preds["threshold"]
    st_month = preds.groupby(["station_name", "year", "month"]).agg(warn=("warn", "any"), above=("above", "any")).reset_index()

    dmap = district_station_map()
    truth = load_truth()
    truth = truth[truth.district.isin(dmap)]
    truth["station_name"] = truth.district.map(dmap)
    m = truth.merge(st_month, on=["station_name", "year", "month"], how="inner")
    y = (m.pop_affected >= MIN_AFFECTED).to_numpy()
    clim = m.month.isin([6, 7]).to_numpy()
    res = {
        "unit": "district-month, May-Oct 2021-2026",
        "truth": f"ASDMA reported >= {MIN_AFFECTED} people affected",
        "n": int(len(m)),
        "model_3day_forecast": score(y, m.warn.to_numpy()),
        "baseline_climatology_jun_jul": score(y, clim),
        "ceiling_river_above_warning_observed": score(y, m.above.to_numpy()),
        "per_year_model_csi": {int(yr): score((g.pop_affected >= MIN_AFFECTED).to_numpy(), g.warn.to_numpy())["csi"] for yr, g in m.groupby("year")},
        "district_station_map": dmap,
    }
    # people in district-months that were warned vs missed
    res["share_of_affected_people_in_warned_months"] = round(float(m.loc[m.warn, "pop_affected"].sum() / max(m.pop_affected.sum(), 1)), 3)
    return res


if __name__ == "__main__":
    r = validate()
    Path("reports/asdma_validation.json").write_text(json.dumps(r, indent=2))
    print(json.dumps({k: v for k, v in r.items() if k != "district_station_map"}, indent=1))
