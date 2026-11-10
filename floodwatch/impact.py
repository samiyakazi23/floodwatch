"""Impact-based flood forecasting: predict where ASDMA will REPORT people affected,
not just where a river crosses a line.

Unit: district-month in the monsoon (May-Oct), 2021-2026, 33 districts.
Target: ASDMA reported >= MIN_AFFECTED people affected in that district-month.

Features (all available before the month's impacts are known):
  * the river forecast signals for the district's station, taken from the
    walk-forward backtest (so they are genuinely out-of-sample):
    max forecast probability at each lead, days warned, days above warning
    level, peak flow relative to warning level
  * the same signals for the station upstream on the Brahmaputra
  * the district's flood history from OTHER years only (exposure / vulnerability)
  * month of the year

Evaluation: leave-one-year-out. For each year 2021-2026, train on the other five
years, tune the alert cut-off inside those five, predict the held-out year.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import math
import os

from .forecast import load_rain
from .validate_asdma import MIN_AFFECTED, centroid, district_station_map, load_truth, score

USE_RAIN = os.getenv("FW_RAIN", "0") == "1"


def district_rain_month() -> pd.DataFrame:
    """Per district-month: total rain and wettest 3-day spell at the district's nearest rain point."""
    rain = load_rain()
    pts = {f.stem: json.loads(f.read_text()) for f in Path("data/rain").glob("*.json")}
    geo = json.loads(Path("data/geo/assam_districts.geojson").read_text())
    r3 = rain.rolling(3, min_periods=1).sum()
    rows = []
    for f in geo["features"]:
        name = f["properties"]["dtname"]; lat, lon = centroid(f["geometry"])
        near = min(pts, key=lambda k: math.hypot(pts[k]["lat"] - lat, (pts[k]["lon"] - lon) * math.cos(math.radians(lat))))
        m = pd.DataFrame({"tot": rain[near], "max3": r3[near], "basin": rain.mean(axis=1)})
        g = m.groupby([m.index.year, m.index.month]).agg(rain_month=("tot", "sum"), rain_max3=("max3", "max"), rain_basin=("basin", "sum"))
        g.index.names = ["year", "month"]
        g = g.reset_index(); g["district"] = name
        rows.append(g)
    out = pd.concat(rows)
    for c in ("rain_month", "rain_max3", "rain_basin"):
        out[c] = np.log1p(out[c])
    return out

HORIZONS = [1, 2, 3, 5, 7]
UPSTREAM = {  # next station up the main stem (tributary stations use the main-stem station they join near)
    "Dhubri": "Goalpara", "Goalpara": "Guwahati", "Guwahati": "Tezpur", "Tezpur": "Neamatighat",
    "Neamatighat": "Dibrugarh", "Dibrugarh": "Dibrugarh", "Badatighat": "Dibrugarh", "Jia Bharali": "Neamatighat",
    "Numaligarh": "Neamatighat", "Dharamtul": "Tezpur", "Beki Road": "Guwahati", "Silchar": "Silchar",
}


def station_month_features(preds: pd.DataFrame) -> pd.DataFrame:
    p = preds.copy()
    p["year"], p["month"] = p.date.dt.year, p.date.dt.month
    p["ratio"] = p["q_raw"] / p["threshold"]
    agg = {"ratio": ["max", "mean"], "flood_now": "sum"}
    for h in HORIZONS:
        p[f"warn_{h}"] = (p[f"p_{h}"] >= p[f"cut_{h}"]).astype(int)
        agg[f"p_{h}"] = "max"
        agg[f"warn_{h}"] = "sum"
    f = p.groupby(["station_name", "year", "month"]).agg(agg)
    f.columns = ["_".join(c) for c in f.columns]
    return f.reset_index()


def build_table(preds_path: str | None = None) -> pd.DataFrame:
    preds_path = preds_path or ("reports/backtest_predictions_rain.csv.gz" if USE_RAIN else "reports/backtest_predictions.csv.gz")
    preds = pd.read_csv(preds_path, parse_dates=["date"])
    sm = station_month_features(preds)
    dmap = district_station_map()
    t = load_truth()
    t = t[t.district.isin(dmap)].copy()
    t["station_name"] = t.district.map(dmap)
    t["up_station"] = t.station_name.map(UPSTREAM)
    t = t.merge(sm, on=["station_name", "year", "month"], how="inner")
    up = sm.add_prefix("up_").rename(columns={"up_station_name": "up_station", "up_year": "year", "up_month": "month"})
    t = t.merge(up, on=["up_station", "year", "month"], how="left")
    if USE_RAIN:
        t = t.merge(district_rain_month(), on=["district", "year", "month"], how="left")
    t["y"] = (t.pop_affected >= MIN_AFFECTED).astype(int)
    for m in range(5, 11):
        t[f"m{m}"] = (t.month == m).astype(int)
    return t


def feature_cols(t: pd.DataFrame) -> list[str]:
    skip = {"district", "year", "month", "pop_affected", "station_name", "up_station", "y"}
    return [c for c in t.columns if c not in skip]


def add_history(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """District flood rate from training years only (smoothed toward the overall rate)."""
    prior = train.y.mean()
    g = train.groupby("district").y.agg(["sum", "count"])
    rate = (g["sum"] + 2 * prior) / (g["count"] + 2)
    gm = train.groupby(["district", "month"]).y.agg(["sum", "count"])
    mrate = (gm["sum"] + 2 * prior) / (gm["count"] + 2)
    out = []
    for df in (train, test):
        df = df.copy()
        df["hist_rate"] = df.district.map(rate).fillna(prior)
        df["hist_month_rate"] = [mrate.get((d, m), prior) for d, m in zip(df.district, df.month)]
        out.append(df)
    return out[0], out[1]


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced"))


def best_cut(y, p):
    grid = np.linspace(0.1, 0.9, 33)
    return float(max(grid, key=lambda c: score(y.astype(bool), p >= c)["csi"]))


def leave_one_year_out(t: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for year in sorted(t.year.unique()):
        train, test = add_history(t[t.year != year], t[t.year == year])
        cols = feature_cols(train)
        # tune cut-off with an inner leave-one-year-out on the training years
        inner_p, inner_y = [], []
        for vy in sorted(train.year.unique()):
            a, b = add_history(train[train.year != vy], train[train.year == vy])
            inner_p.append(make_model().fit(a[cols], a.y).predict_proba(b[cols])[:, 1]); inner_y.append(b.y.to_numpy())
        cut = best_cut(np.concatenate(inner_y), np.concatenate(inner_p))
        model = make_model().fit(train[cols], train.y)
        out = test[["district", "year", "month", "pop_affected", "y", "warn_3_sum", "flood_now_sum"]].copy()
        out["p"] = model.predict_proba(test[cols])[:, 1]
        out["warn"] = out.p >= cut
        out["cut"] = cut
        out["hist_only_warn"] = test.hist_month_rate >= 0.5
        rows.append(out)
    res = pd.concat(rows)
    y = res.y.astype(bool).to_numpy()
    summary = {
        "unit": "district-month, May-Oct 2021-2026 (leave-one-year-out)",
        "n": int(len(res)),
        "impact_model": score(y, res.warn.to_numpy()),
        "river_forecast_only": score(y, (res.warn_3_sum > 0).to_numpy()),
        "river_observed_above_warning": score(y, (res.flood_now_sum > 0).to_numpy()),
        "district_history_only": score(y, res.hist_only_warn.to_numpy()),
        "climatology_jun_jul": score(y, res.month.isin([6, 7]).to_numpy()),
        "per_year_impact_csi": {int(k): score(g.y.astype(bool).to_numpy(), g.warn.to_numpy())["csi"] for k, g in res.groupby("year")},
        "share_of_affected_people_in_warned_months": round(float(res.loc[res.warn, "pop_affected"].sum() / res.pop_affected.sum()), 3),
    }
    return res, summary


def strict_walkforward(t: pd.DataFrame, first: int = 2022) -> dict:
    """Each year predicted by a model trained ONLY on earlier years (cut-off tuned on the latest past year)."""
    rows = []
    for year in sorted(y for y in t.year.unique() if y >= first):
        train, test = add_history(t[t.year < year], t[t.year == year])
        cols = feature_cols(train)
        if train.year.nunique() >= 2:
            a, b = add_history(train[train.year < year - 1], train[train.year == year - 1])
            cut = best_cut(b.y.to_numpy(), make_model().fit(a[cols], a.y).predict_proba(b[cols])[:, 1])
        else:
            cut = 0.5
        out = test[["year", "month", "y", "warn_3_sum"]].copy()
        out["warn"] = make_model().fit(train[cols], train.y).predict_proba(test[cols])[:, 1] >= cut
        rows.append(out)
    r = pd.concat(rows)
    sc = lambda g: score(g.y.astype(bool).to_numpy(), g.warn.to_numpy())
    late = r[r.year >= 2023]
    return {"all_years": sc(r), "from_2023": sc(late),
            "climatology_from_2023": score(late.y.astype(bool).to_numpy(), late.month.isin([6, 7]).to_numpy()),
            "per_year": {int(k): sc(g)["csi"] for k, g in r.groupby("year")}}


if __name__ == "__main__":
    t = build_table()
    res, summary = leave_one_year_out(t)
    summary["strict_walkforward"] = strict_walkforward(t)
    tag = "_rain" if USE_RAIN else ""
    Path(f"reports/impact_validation{tag}.json").write_text(json.dumps(summary, indent=2))
    res.to_csv(f"reports/impact_predictions{tag}.csv", index=False)
    print(json.dumps(summary, indent=1))
