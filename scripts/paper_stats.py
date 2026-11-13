"""Significance tests and confidence intervals reported in the FloodWatch paper.

Run after the backtests:  python scripts/paper_stats.py
- River forecasts: block bootstrap over station-years (72 blocks, 2,000 resamples, seed 0).
- Impact model: block bootstrap over year-months (35 blocks), since one monsoon month
  affects many districts at once.
- 2022 daily onsets and station onsets: exact McNemar test, rainfall vs river-only forecast.
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from floodwatch import validate_daily as vd  # noqa: E402
from floodwatch.validate_asdma import district_station_map  # noqa: E402

RNG_SEED, N_BOOT = 0, 2000
csi = lambda s: s[0] / s.sum()  # s = [tp, fn, fp]


def blocks(key, y, w):
    return pd.DataFrame({"k": key, "tp": y & w, "fn": y & ~w, "fp": ~y & w}).groupby("k").sum()


def boot(mats, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(next(iter(mats.values())))
    out = {k: [] for k in mats}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        for k, m in mats.items():
            out[k].append(csi(m[i].sum(0)))
    return {k: np.array(v) for k, v in out.items()}


ci = lambda v: [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])]
res = {}

# river forecasts
pr = pd.read_csv("reports/backtest_predictions_rain.csv.gz", parse_dates=["date"])
pn = pd.read_csv("reports/backtest_predictions.csv.gz", parse_dates=["date"])
res["river"] = {}
for h in [1, 2, 3, 5, 7]:
    a, b = pr.dropna(subset=[f"y_{h}"]), pn.dropna(subset=[f"y_{h}"])
    key = a.station_name + "_" + a.date.dt.year.astype(str)
    ya = a[f"y_{h}"].astype(bool).values
    m = {"rain": blocks(key, ya, (a[f"p_{h}"] >= a[f"cut_{h}"]).values),
         "persistence": blocks(key, ya, a.flood_now.astype(bool).values)}
    kb = b.station_name + "_" + b.date.dt.year.astype(str)
    m["river_only"] = blocks(kb, b[f"y_{h}"].astype(bool).values, (b[f"p_{h}"] >= b[f"cut_{h}"]).values).reindex(m["rain"].index)
    mats = {k: v.values for k, v in m.items()}
    bs = boot(mats)
    res["river"][h] = {"blocks": len(m["rain"]), "csi": round(csi(mats["rain"].sum(0)), 3), "csi_ci": ci(bs["rain"]),
                       "gain_vs_persistence_ci": ci(bs["rain"] - bs["persistence"]),
                       "gain_from_rain_ci": ci(bs["rain"] - bs["river_only"])}

# impact model
r = pd.read_csv("reports/impact_predictions_rain.csv")
y = r.y.astype(bool)
key = r.year.astype(str) + "_" + r.month.astype(str)
ws = {"impact": r.warn.astype(bool), "calendar": r.month.isin([6, 7]), "river": r.warn_3_sum > 0, "history": r.hist_only_warn.astype(bool)}
mats = {k: blocks(key, y, w).values for k, w in ws.items()}
bs = boot(mats)
d = bs["impact"] - bs["calendar"]
res["impact"] = {"blocks": len(mats["impact"]), **{f"{k}_ci": ci(v) for k, v in bs.items()},
                 "gain_vs_calendar_ci": ci(d), "one_sided_p": round(float(np.mean(d <= 0)), 3)}

# McNemar, 2022 district onsets
truth = vd.load_daily_truth(); dmap = district_station_map(); ons = vd.onsets(truth, sorted(dmap))
res["daily_2022_mcnemar"] = {}
for L in [1, 3, 5]:
    wr = np.array([bool(vd.warned_before(pr, dmap[o.district], o.onset, L)) for o in ons.itertuples()])
    wn = np.array([bool(vd.warned_before(pn, dmap[o.district], o.onset, L)) for o in ons.itertuples()])
    b, c = int((wr & ~wn).sum()), int((~wr & wn).sum())
    res["daily_2022_mcnemar"][L] = {"rain_only": b, "river_only": c, "p": round(binomtest(b, b + c, 0.5).pvalue, 4)}

# McNemar, station onsets warned 3+ days ahead
er = {(e["station"], e["onset"]): e["warned_days_ahead"] >= 3 for e in json.load(open("reports/forecast_backtest_rain.json"))["events"]["list"]}
en = {(e["station"], e["onset"]): e["warned_days_ahead"] >= 3 for e in json.load(open("reports/forecast_backtest.json"))["events"]["list"]}
ks = [k for k in er if k in en]
b, c = sum(er[k] and not en[k] for k in ks), sum(en[k] and not er[k] for k in ks)
res["station_onsets_mcnemar"] = {"onsets": len(ks), "rain_only": b, "river_only": c, "p": round(binomtest(b, b + c, 0.5).pvalue, 3)}

Path("reports/paper_stats.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
