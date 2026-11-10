"""Export the real model outputs the website shows (all from the backtest; nothing invented)."""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.geometry.polygon import orient
from floodwatch.validate_asdma import district_station_map, load_truth

OUT = Path("docs/data"); OUT.mkdir(parents=True, exist_ok=True)

# 1. districts, simplified
geo = json.loads(Path("data/geo/assam_districts.geojson").read_text())
dmap = district_station_map()
feats = []
for f in geo["features"]:
    g = shape(f["geometry"]).simplify(0.01, preserve_topology=True)
    # d3-geo expects clockwise exterior rings; counter-clockwise ones render as "the whole world minus the district"
    if isinstance(g, MultiPolygon):  # drop zero-area slivers left by simplification (d3 reads them as the whole globe)
        parts = [x for x in g.geoms if x.area > 1e-4]
        g = parts[0] if len(parts) == 1 else MultiPolygon(parts)
    g = orient(g, sign=-1.0) if isinstance(g, Polygon) else MultiPolygon([orient(x, sign=-1.0) for x in g.geoms])
    gj = json.loads(json.dumps(mapping(g), default=list))
    def rnd(c):
        return [rnd(x) for x in c] if isinstance(c[0], (list, tuple)) else [round(c[0], 3), round(c[1], 3)]
    gj["coordinates"] = rnd(gj["coordinates"])
    n = f["properties"]["dtname"]
    feats.append({"type": "Feature", "properties": {"name": n, "station": dmap[n]}, "geometry": gj})
(OUT / "districts.json").write_text(json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")))

# 2. stations
st = json.loads(Path("data/river/stations.json").read_text())
fmeta = json.loads(Path("reports/forecast_backtest_rain.json").read_text())
fmeta_river = json.loads(Path("reports/forecast_backtest.json").read_text())
(OUT / "stations.json").write_text(json.dumps({k: {"river": v["river"], "lat": v["cell_lat"], "lon": v["cell_lon"]} for k, v in st.items()}))

# 3. daily station series, monsoon months, from the walk-forward backtest
p = pd.read_csv("reports/backtest_predictions_rain.csv.gz", parse_dates=["date"])  # the deployed model (river + rainfall)
# all months, so any date in 2021-2026 can be checked
daily = {}
for s, g in p.groupby("station_name"):
    g = g.sort_values("date")
    daily[s] = {
        "date": g.date.dt.strftime("%Y-%m-%d").tolist(),
        "q": g.q_raw.round(0).astype(int).tolist(),
        "thr": g.threshold.round(0).astype(int).tolist(),
        **{f"p{h}": g[f"p_{h}"].round(3).tolist() for h in [1, 2, 3, 5, 7]},
        **{f"c{h}": g[f"cut_{h}"].round(3).tolist() for h in [1, 2, 3, 5, 7]},
    }
(OUT / "daily.json").write_text(json.dumps(daily, separators=(",", ":")))

# 4. ASDMA truth + impact model (leave-one-year-out and strict) per district-month
t = load_truth()
imp = pd.read_csv("reports/impact_predictions_rain.csv")
m = t.merge(imp[["district", "year", "month", "p", "cut"]], on=["district", "year", "month"], how="left")
rows = [{"d": r.district, "y": int(r.year), "m": int(r.month), "pop": int(r.pop_affected),
         "p": None if pd.isna(r.p) else round(float(r.p), 3), "cut": None if pd.isna(r.cut) else round(float(r.cut), 3)} for r in m.itertuples()]
(OUT / "impacts.json").write_text(json.dumps(rows, separators=(",", ":")))

# 5. headline metrics
summary = {
    "forecast": {h: {k: v["model"][k] for k in ("hit_rate", "false_alarm_ratio", "csi")} | {"persistence_csi": v["persistence_baseline"]["csi"]} for h, v in fmeta["horizons"].items()},
    "events": {k: v for k, v in fmeta["events"].items() if k != "list"},
    "forecast_river_only": {h: v["model"]["csi"] for h, v in fmeta_river["horizons"].items()},
    "events_river_only": {k: v for k, v in fmeta_river["events"].items() if k != "list"},
    "impact_loyo": json.loads(Path("reports/impact_validation_rain.json").read_text()),
    "impact_strict": json.loads(Path("reports/impact_validation_rain.json").read_text())["strict_walkforward"],
    "daily_2022": json.loads(Path("reports/daily_validation_2022.json").read_text()),
    "segmentation": json.loads(Path("reports/segmentation.json").read_text())["test"] | {"bolivia": json.loads(Path("reports/segmentation.json").read_text())["bolivia_unseen_event"]["unet"]},
}
summary["segmentation"].pop("per_event_iou", None)
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
for f in OUT.iterdir(): print(f.name, f.stat().st_size // 1024, "KB")
