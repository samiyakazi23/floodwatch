"""Day-by-day check against ASDMA's daily district reports for the 2022 monsoon.

Truth: a district is "flooded" on a day when ASDMA reported >= MIN_AFFECTED people affected.
Question: when a district's flooding STARTS (first flooded day after a quiet week), had the
forecast for its river station warned 1, 3 and 5 days before?

Compared: the river-only model, the river + rainfall model, and two simple references:
  rain_only   heavy local rain in the past 3 days (top 10% of 3-day totals at the nearest rain point)
  near_level  river already above 85% of its warning level
Data: 2022 daily FRIMS reports as compiled by CivicDataLab (data/asdma/daily_2022.csv).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .forecast import load_rain
from .validate_asdma import RENAME, centroid, district_station_map

MIN_AFFECTED = 1000
QUIET_DAYS = 7
LEADS = (1, 3, 5)
OLD_NAMES = {"KAMRUP(M)": "Kamrup Metropolitan", "KAMRUP (M)": "Kamrup Metropolitan", "KAMRUP METRO": "Kamrup Metropolitan",
             "SOUTH SALMARA": "South Salmara Mancachar", "WEST KARBI ANGLONG": "West Karbi Anglong", "BISWANATH": "Biswanath",
             "DIMA-HASAO": "Dima Hasao", "KARBI ANGLONG WEST": "West Karbi Anglong", "SIVSAGAR": "Sivasagar",
             "WESTKARIMGANJ": "Karimganj"}  # spellings used in the 2022 bulletins


def load_daily_truth(path: str = "data/asdma/daily_2022.csv") -> pd.DataFrame:
    d = pd.read_csv(path, parse_dates=["date"])
    d["district"] = d["district"].str.strip().str.upper().map(lambda x: OLD_NAMES.get(x, RENAME.get(x, x.title())))
    return d.groupby(["district", "date"], as_index=False)["affected"].sum()


def onsets(truth: pd.DataFrame, districts: list[str]) -> pd.DataFrame:
    days = pd.date_range(truth.date.min(), truth.date.max())
    rows = []
    for dname in districts:
        s = truth[truth.district == dname].set_index("date")["affected"].reindex(days).fillna(0) >= MIN_AFFECTED
        f = s.to_numpy()
        for i in range(QUIET_DAYS, len(f)):
            if f[i] and not f[i - QUIET_DAYS:i].any():
                rows.append({"district": dname, "onset": days[i]})
    return pd.DataFrame(rows)


def warned_before(preds: pd.DataFrame, station: str, onset: pd.Timestamp, lead: int) -> bool | None:
    """Did a forecast issued `lead` days before onset (with horizon >= lead) cross its alert cut-off?
    Uses the h-day forecast whose window [issue+1, issue+h] contains the onset day."""
    g = preds[preds.station_name == station].set_index("date")
    issue = onset - pd.Timedelta(days=lead)
    if issue not in g.index:
        return None
    for h in (1, 2, 3, 5, 7):
        if h >= lead and g.at[issue, f"p_{h}"] >= g.at[issue, f"cut_{h}"]:
            return True
    return False


def run() -> dict:
    dmap = district_station_map()
    truth = load_daily_truth()
    truth = truth[truth.district.isin(dmap)]
    ons = onsets(truth, sorted(dmap))
    models = {"river_only": "reports/backtest_predictions.csv.gz", "river_plus_rain": "reports/backtest_predictions_rain.csv.gz"}
    preds = {k: pd.read_csv(v, parse_dates=["date"]) for k, v in models.items()}

    # references
    rain = load_rain()
    r3 = rain.rolling(3, min_periods=1).sum()
    heavy = r3 >= r3[r3.index.year < 2021].quantile(0.9)  # threshold from pre-test years
    pts = {f.stem: json.loads(f.read_text()) for f in Path("data/rain").glob("*.json")}
    geo = {f["properties"]["dtname"]: centroid(f["geometry"]) for f in json.loads(Path("data/geo/assam_districts.geojson").read_text())["features"]}
    near_pt = {d: min(pts, key=lambda k: math.hypot(pts[k]["lat"] - la, (pts[k]["lon"] - lo) * math.cos(math.radians(la)))) for d, (la, lo) in geo.items()}
    base = preds["river_only"].set_index(["station_name", "date"])

    res = {"onsets": int(len(ons)), "season": f"{truth.date.min().date()} to {truth.date.max().date()}", "min_affected": MIN_AFFECTED}
    for name, p in preds.items():
        res[name] = {}
        for L in LEADS:
            w = [warned_before(p, dmap[r.district], r.onset, L) for r in ons.itertuples()]
            w = [x for x in w if x is not None]
            res[name][f"warned_{L}d_before"] = round(float(np.mean(w)), 3)
    for L in LEADS:
        rr, nn = [], []
        for r in ons.itertuples():
            day = r.onset - pd.Timedelta(days=L)
            rr.append(bool(heavy.at[day, near_pt[r.district]]) if day in heavy.index else False)
            key = (dmap[r.district], day)
            nn.append(bool(base.at[key, "q_raw"] > 0.85 * base.at[key, "threshold"]) if key in base.index else False)
        res.setdefault("ref_heavy_local_rain", {})[f"warned_{L}d_before"] = round(float(np.mean(rr)), 3)
        res.setdefault("ref_river_near_level", {})[f"warned_{L}d_before"] = round(float(np.mean(nn)), 3)

    # false alarms on quiet days: share of district-days with no one affected for the next 5 days that were warned (3-day forecast)
    days = pd.date_range(truth.date.min(), truth.date.max() - pd.Timedelta(days=5))
    for name, p in preds.items():
        g = p.set_index(["station_name", "date"])
        fa, n = 0, 0
        for dname in sorted(dmap):
            s = truth[truth.district == dname].set_index("date")["affected"].reindex(pd.date_range(truth.date.min(), truth.date.max())).fillna(0)
            for d in days:
                if (s.loc[d:d + pd.Timedelta(days=5)] < MIN_AFFECTED).all():
                    n += 1
                    key = (dmap[dname], d)
                    fa += bool(g.at[key, "p_3"] >= g.at[key, "cut_3"]) if key in g.index else 0
        res[name]["false_alarm_rate_quiet_days"] = round(fa / max(n, 1), 3)
    # same quiet-day check for the heavy-rain reference
    fa, n = 0, 0
    for dname in sorted(dmap):
        s = truth[truth.district == dname].set_index("date")["affected"].reindex(pd.date_range(truth.date.min(), truth.date.max())).fillna(0)
        for d in days:
            if (s.loc[d:d + pd.Timedelta(days=5)] < MIN_AFFECTED).all():
                n += 1
                fa += bool(heavy.at[d, near_pt[dname]]) if d in heavy.index else 0
    res["ref_heavy_local_rain"]["false_alarm_rate_quiet_days"] = round(fa / max(n, 1), 3)
    return res


if __name__ == "__main__":
    r = run()
    Path("reports/daily_validation_2022.json").write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=1))
