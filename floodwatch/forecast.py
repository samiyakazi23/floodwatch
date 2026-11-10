"""Flood forecasting for Assam's rivers, 1-7 days ahead, with honest walk-forward backtesting.

Data: daily river discharge (GloFAS v4 via Open-Meteo, 1997-2026) at 12 river
stations: six on the Brahmaputra main stem from Dibrugarh to Dhubri, plus the
Subansiri, Jia Bharali, Dhansiri, Kopili, Beki/Manas and Barak.

Flood definition: flow above the station's 2-year return-period level, the
standard "flood alert" threshold used by GloFAS. It is computed from the
annual maxima of the TRAINING years only, so the test never leaks.

Why it can see the future: water takes days to travel down the Brahmaputra.
A surge passing Dibrugarh today reaches Guwahati around 3 days later, so
upstream stations are informative features for downstream forecasts.

Backtest: expanding-window walk-forward. To forecast year Y, the model is
trained only on 1997..Y-1. Test years: 2021-2026.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

DATA = Path("data/river")
HORIZONS = [1, 2, 3, 5, 7]
LAGS = [0, 1, 2, 3, 5, 7]
TEST_YEARS = list(range(2021, 2027))
FIRST_YEAR = 1997
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7, num_threads=1)
ROUNDS = 400


def load_flows() -> pd.DataFrame:
    """Wide frame: one column of daily discharge per station."""
    meta = json.loads((DATA / "stations.json").read_text())
    cols = {}
    for name in meta:
        d = json.loads((DATA / f"{name}.json").read_text())
        cols[name] = pd.Series(d["q"], index=pd.to_datetime(d["time"]), dtype=float)
    df = pd.DataFrame(cols)
    df = df[df.index.year >= FIRST_YEAR]
    return df.interpolate(limit=3)


LEVELS = {"warning": 0.25, "severe": 0.5}  # quantile of annual maxima: exceeded ~3 in 4 years / ~1 in 2 years
LEVEL = "warning"


def thresholds(flows: pd.DataFrame, last_train_year: int, level: str | None = None) -> pd.Series:
    """Flood level per station from annual maxima of the TRAINING years only.
    warning = 25th percentile of annual maxima (reached in ~3 of 4 years), severe = median (2-year return level)."""
    train = flows[flows.index.year <= last_train_year]
    return train.groupby(train.index.year).max().quantile(LEVELS[level or LEVEL])


def build_features(flows: pd.DataFrame) -> pd.DataFrame:
    """Long frame: one row per (date, station) with own + all-station features available on that date."""
    logq = np.log1p(flows)
    shared = {}
    for s in flows.columns:  # every station's recent state is visible to every forecast (upstream routing)
        for lag in (0, 1, 2, 3):
            shared[f"all_{s}_lag{lag}"] = logq[s].shift(lag)
        shared[f"all_{s}_chg2"] = logq[s] - logq[s].shift(2)
        shared[f"all_{s}_chg1"] = logq[s] - logq[s].shift(1)
        shared[f"all_{s}_max3"] = logq[s].rolling(3).max()
    shared = pd.DataFrame(shared, index=flows.index)
    doy = flows.index.dayofyear
    shared["doy_sin"], shared["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)

    rows = []
    for i, s in enumerate(flows.columns):
        f = pd.DataFrame(index=flows.index)
        f["station"] = i
        for lag in LAGS:
            f[f"q_lag{lag}"] = logq[s].shift(lag)
        f["chg1"] = logq[s] - logq[s].shift(1)
        f["chg3"] = logq[s] - logq[s].shift(3)
        f["max30"] = logq[s].rolling(30).max()
        f["mean7"] = logq[s].rolling(7).mean()
        f["q_raw"] = flows[s]
        for h in range(1, max(HORIZONS) + 1):
            f[f"q_future_{h}"] = flows[s].shift(-h)
        rows.append(pd.concat([f, shared], axis=1).assign(station_name=s))
    out = pd.concat(rows)
    out.index.name = "date"
    return out.reset_index()


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {"date", "station_name", "q_raw"} and not c.startswith("q_future_") and not c.startswith("y_")]


def add_targets(df: pd.DataFrame, thr: pd.Series) -> pd.DataFrame:
    df = df.copy()
    t = df["station_name"].map(thr)
    df["ratio"] = np.log(df["q_raw"] / t)  # how close to flood level today
    df["flood_now"] = (df["q_raw"] > t).astype(int)
    # target: flood level reached on ANY day in the next h days (what an early warning has to say)
    fut = np.column_stack([df[f"q_future_{h}"].to_numpy() for h in range(1, max(HORIZONS) + 1)])
    for h in HORIZONS:
        window = fut[:, :h]
        known = ~np.isnan(window).any(axis=1)
        df[f"y_{h}"] = np.where(known, (np.nan_to_num(window) > t.to_numpy()[:, None]).any(axis=1).astype(float), np.nan)
    return df


def contingency(y: np.ndarray, warn: np.ndarray) -> dict:
    tp = int(((y == 1) & warn).sum()); fp = int(((y == 0) & warn).sum()); fn = int(((y == 1) & ~warn).sum())
    return {
        "hit_rate": round(tp / max(tp + fn, 1), 3),          # share of flood days that were warned
        "false_alarm_ratio": round(fp / max(tp + fp, 1), 3),  # share of warnings that were wrong
        "csi": round(tp / max(tp + fp + fn, 1), 3),           # critical success index (hydrology standard)
        "flood_days": int(y.sum()), "warnings": int(warn.sum()),
    }


def onsets(flag: pd.Series, quiet: int = 7) -> list[pd.Timestamp]:
    """Flood onset = first flood day after at least `quiet` non-flood days."""
    f = flag.to_numpy().astype(bool)
    idx = [i for i in range(len(f)) if f[i] and not f[max(0, i - quiet):i].any() and i >= quiet]
    return [flag.index[i] for i in idx]


def best_cut(y: np.ndarray, p: np.ndarray) -> float:
    """Alert cut-off that maximises CSI on validation years (never on the test year)."""
    grid = np.linspace(0.05, 0.9, 35)
    scores = [contingency(y.astype(int), p >= c)["csi"] for c in grid]
    return float(grid[int(np.argmax(scores))])


def backtest() -> dict:
    flows = load_flows()
    base = build_features(flows)
    results: dict = {"stations": list(flows.columns), "test_years": TEST_YEARS, "level": LEVEL, "target": "flood level reached within the next h days", "horizons": {}}
    preds_all, cuts = [], {}
    for year in TEST_YEARS:
        thr = thresholds(flows, year - 1)
        df = add_targets(base, thr)
        cols = feature_cols(df)
        train = df[(df.date.dt.year < year)]
        test = df[(df.date.dt.year == year)]
        out = test[["date", "station_name", "q_raw", "flood_now"]].copy()
        out["threshold"] = out["station_name"].map(thr)
        for h in HORIZONS:
            tr = train.dropna(subset=[f"y_{h}"])
            tr = tr[tr.date <= pd.Timestamp(f"{year - 1}-12-31") - pd.Timedelta(days=h)]  # label must be known by Jan 1
            fit, val = tr[tr.date.dt.year < year - 3], tr[tr.date.dt.year >= year - 3]
            m_val = lgb.train(PARAMS, lgb.Dataset(fit[cols], fit[f"y_{h}"], categorical_feature=["station"]), ROUNDS)
            cuts[(year, h)] = best_cut(val[f"y_{h}"].to_numpy(), m_val.predict(val[cols]))
            model = lgb.train(PARAMS, lgb.Dataset(tr[cols], tr[f"y_{h}"], categorical_feature=["station"]), ROUNDS)
            out[f"p_{h}"] = model.predict(test[cols])
            out[f"y_{h}"] = test[f"y_{h}"].to_numpy()
            out[f"cut_{h}"] = cuts[(year, h)]
        preds_all.append(out)
    preds = pd.concat(preds_all, ignore_index=True)

    for h in HORIZONS:
        p = preds.dropna(subset=[f"y_{h}"])
        y = p[f"y_{h}"].to_numpy().astype(int)
        model_warn = p[f"p_{h}"].to_numpy() >= p[f"cut_{h}"].to_numpy()
        persist_warn = p["flood_now"].to_numpy().astype(bool)
        # second baseline: "the river is rising and already near flood level"
        near = (p["q_raw"] > 0.85 * p["threshold"]).to_numpy()
        mons = p.date.dt.month.between(5, 10).to_numpy()
        results["horizons"][h] = {
            "model": contingency(y, model_warn),
            "persistence_baseline": contingency(y, persist_warn),
            "near_threshold_baseline": contingency(y, near),
            "model_monsoon_only": contingency(y[mons], model_warn[mons]),
            "pr_auc": round(float(average_precision_score(y, p[f"p_{h}"])), 3),
            "brier": round(float(brier_score_loss(y, p[f"p_{h}"])), 4),
        }

    # event view: for each flood onset, how many days ahead did the warning start?
    events = []
    for s, g in preds.groupby("station_name"):
        g = g.set_index("date").sort_index()
        flood = g["q_raw"] > g["threshold"]
        for d in onsets(flood):
            lead = 0
            for h in sorted(HORIZONS, reverse=True):
                issue = d - pd.Timedelta(days=h)
                if issue in g.index and g.at[issue, f"p_{h}"] >= g.at[issue, f"cut_{h}"]:
                    lead = h
                    break
            events.append({"station": s, "onset": d.strftime("%Y-%m-%d"), "warned_days_ahead": lead,
                           "peak_q": round(float(g.loc[d:d + pd.Timedelta(days=20), "q_raw"].max()))})
    ev = pd.DataFrame(events)
    results["events"] = {
        "n_onsets": len(ev),
        "warned_1plus_days": round(float((ev.warned_days_ahead >= 1).mean()), 3) if len(ev) else None,
        "warned_3plus_days": round(float((ev.warned_days_ahead >= 3).mean()), 3) if len(ev) else None,
        "median_lead_days": float(ev.warned_days_ahead.median()) if len(ev) else None,
        "list": events,
    }
    results["cutoffs"] = {f"{y}_h{h}": round(c, 3) for (y, h), c in cuts.items()}
    results["per_year_csi_3day"] = {
        int(y): contingency(g.dropna(subset=["y_3"])["y_3"].to_numpy().astype(int), (g.dropna(subset=["y_3"])["p_3"] >= g.dropna(subset=["y_3"])["cut_3"]).to_numpy())["csi"]
        for y, g in preds.groupby(preds.date.dt.year)
    }
    return results, preds


def train_final(out_dir: Path = Path("models")) -> dict:
    """Fit on ALL labelled data (1997 to now) for live use.
    Flood levels come from all years; alert cut-offs are the median of the cut-offs
    tuned during the walk-forward backtest (so they were chosen on held-out years)."""
    flows = load_flows()
    thr = thresholds(flows, int(flows.index.year.max()))
    df = add_targets(build_features(flows), thr)
    cols = feature_cols(df)
    out_dir.mkdir(exist_ok=True)
    bt = json.loads(Path("reports/forecast_backtest.json").read_text())["cutoffs"]
    cuts = {h: float(np.median([v for k, v in bt.items() if k.endswith(f"_h{h}")])) for h in HORIZONS}
    for h in HORIZONS:
        tr = df.dropna(subset=[f"y_{h}"])
        lgb.train(PARAMS, lgb.Dataset(tr[cols], tr[f"y_{h}"], categorical_feature=["station"]), ROUNDS).save_model(str(out_dir / f"forecast_h{h}.txt"))
    meta = {"trained_through": str(flows.index.max().date()), "level": LEVEL, "stations": list(flows.columns),
            "thresholds": thr.round(1).to_dict(), "cutoffs": cuts, "features": cols, "horizons": HORIZONS}
    (out_dir / "forecast_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    import sys
    if "--final" in sys.argv:
        print(json.dumps(train_final(), indent=1)[:600]); raise SystemExit
    res, preds = backtest()
    Path("reports").mkdir(exist_ok=True)
    Path("reports/forecast_backtest.json").write_text(json.dumps(res, indent=2, default=str))
    preds.to_csv("reports/backtest_predictions.csv.gz", index=False)
    summary = {h: {"model": v["model"], "persistence": v["persistence_baseline"], "pr_auc": v["pr_auc"]} for h, v in res["horizons"].items()}
    print(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in res["events"].items() if k != "list"}, indent=1))
    print("per-year CSI (3-day):", res["per_year_csi_3day"])
