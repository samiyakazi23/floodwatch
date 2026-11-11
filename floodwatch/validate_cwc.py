"""Check FloodWatch's flood days against CWC's official gauge reports (monsoon 2021-2026).

CWC's daily flood situation report lists every gauge site that is Above Normal
(at or above warning level), Severe (above danger level) or Extreme. A site that
is not listed on a report day was below warning level. Days with no report are
left out.

Two questions:
  1. Label check: is GloFAS flow above FloodWatch's threshold on the same days
     CWC reports the gauge in flood?
  2. Forecast check: do FloodWatch's 1-7 day warnings catch CWC-reported flood days?

Input: data/cwc/text/YYYY-MM-DD.txt (from scripts/fetch_cwc_reports.py)
       reports/backtest_predictions_rain.csv.gz
Output: reports/cwc_validation.json
Run:   python -m floodwatch.validate_cwc
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

TEXT = Path("data/cwc/text")
PRED = Path("reports/backtest_predictions_rain.csv.gz")
OUT = Path("reports/cwc_validation.json")
HORIZONS = [1, 2, 3, 5, 7]

# FloodWatch station -> how CWC writes the site name at the start of a table row
SITES = {
    "Dibrugarh": [r"Dibrugarh"],
    "Neamatighat": [r"Neamati\s*ghat"],
    "Tezpur": [r"Tezpur"],
    "Guwahati": [r"Guwahati", r"Pandu"],
    "Goalpara": [r"Goalpara"],
    "Dhubri": [r"Dhubri"],
    "Numaligarh": [r"Numaligarh"],
    "Badatighat": [r"Badati\s*ghat"],
    "Jia Bharali": [r"N\.?\s*T\.?\s*Road", r"Jia\s*Bharali"],
    "Beki Road": [r"Beki\s*Road"],
    "Dharamtul": [r"Dharamtul"],
    "Silchar": [r"Annapurna\s*ghat", r"Silchar"],
}
LEVEL = re.compile(r"\b(Above\s+Normal|Severe|Extreme)\b", re.I)
RANK = {"above normal": 1, "severe": 2, "extreme": 3}


def parse_report(text: str) -> dict[str, int]:
    """Highest CWC flood level per FloodWatch station on one report (0 = not listed)."""
    found = {s: 0 for s in SITES}
    for line in text.splitlines():
        m = LEVEL.search(line)
        if not m:
            continue
        head = line.strip()
        for station, pats in SITES.items():
            # site name is the first column, so it must start the row
            if any(re.match(p + r"\b", head, re.I) for p in pats):
                lvl = RANK[re.sub(r"\s+", " ", m.group(1).lower())]
                found[station] = max(found[station], lvl)
    return found


def load_cwc() -> pd.DataFrame:
    rows = []
    for f in sorted(TEXT.glob("*.txt")):
        txt = f.read_text(errors="ignore")
        if "flood" not in txt.lower():  # empty or broken extraction
            continue
        for station, lvl in parse_report(txt).items():
            rows.append({"date": pd.Timestamp(f.stem), "station_name": station, "cwc_level": lvl})
    return pd.DataFrame(rows)


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def scores(obs: np.ndarray, warn: np.ndarray) -> dict:
    tp = int((obs & warn).sum()); fn = int((obs & ~warn).sum()); fp = int((~obs & warn).sum())
    tn = int((~obs & ~warn).sum())
    return {
        "hits": tp, "misses": fn, "false_alarms": fp, "correct_negatives": tn,
        "hit_rate": round(tp / max(tp + fn, 1), 3), "hit_rate_ci95": wilson(tp, tp + fn),
        "false_alarm_ratio": round(fp / max(tp + fp, 1), 3), "false_alarm_ratio_ci95": wilson(fp, tp + fp),
        "csi": round(tp / max(tp + fn + fp, 1), 3),
        "agreement": round((tp + tn) / max(len(obs), 1), 3),
    }


def main() -> dict:
    cwc = load_cwc()
    if cwc.empty:
        raise SystemExit(f"No CWC reports in {TEXT}. Run scripts/fetch_cwc_reports.py first (from an Indian network).")
    pred = pd.read_csv(PRED, parse_dates=["date"])
    df = cwc.merge(pred, on=["date", "station_name"], how="inner")
    obs = (df["cwc_level"] >= 1).to_numpy()
    out = {
        "report_days": int(cwc["date"].nunique()),
        "station_days": int(len(df)),
        "cwc_flood_station_days": int(obs.sum()),
        "stations_ever_listed": sorted(cwc.loc[cwc.cwc_level > 0, "station_name"].unique().tolist()),
        "label_check": scores(obs, (df["q_raw"] > df["threshold"]).to_numpy()),
        "label_check_severe_only": scores((df["cwc_level"] >= 2).to_numpy(), (df["q_raw"] > df["threshold"]).to_numpy()),
        "per_station_label_csi": {
            s: scores((g.cwc_level >= 1).to_numpy(), (g.q_raw > g.threshold).to_numpy())["csi"]
            for s, g in df.groupby("station_name")
        },
        "forecast_vs_cwc": {},
        "by_year": {},
    }
    # forecast issued on day d for "flood within the next h days" vs CWC reporting flood on any of d+1..d+h
    cw = cwc.pivot(index="date", columns="station_name", values="cwc_level")
    for h in HORIZONS:
        fut = (cw >= 1).astype(float).where(cw.notna())
        ahead = pd.concat([fut.shift(-k) for k in range(1, h + 1)]).groupby(level=0).max()
        long = ahead.stack().rename("cwc_ahead").reset_index()
        m = pred[["date", "station_name", f"p_{h}", f"cut_{h}"]].merge(long, on=["date", "station_name"])
        out["forecast_vs_cwc"][f"{h}d"] = scores(m["cwc_ahead"].astype(bool).to_numpy(), (m[f"p_{h}"] >= m[f"cut_{h}"]).to_numpy())
    for y, g in df.groupby(df["date"].dt.year):
        out["by_year"][int(y)] = {"report_days": int(g["date"].nunique()),
                                  **scores((g.cwc_level >= 1).to_numpy(), (g.q_raw > g.threshold).to_numpy())}
    OUT.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: r[k] for k in ("report_days", "station_days", "cwc_flood_station_days", "stations_ever_listed", "label_check")}, indent=1))
    for h, s in r["forecast_vs_cwc"].items():
        print(f"forecast {h}: hit {s['hit_rate']:.0%}  FAR {s['false_alarm_ratio']:.0%}  CSI {s['csi']:.2f}")
