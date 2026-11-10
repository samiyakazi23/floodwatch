"""Fast checks of the rules that make the results trustworthy. Run: pytest -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from floodwatch import forecast as F  # noqa: E402
from floodwatch.impact import add_history  # noqa: E402
from floodwatch.validate_asdma import district_station_map, load_truth, score  # noqa: E402


@pytest.fixture(scope="module")
def flows():
    return F.load_flows()


def test_twelve_stations_with_realistic_flows(flows):
    assert flows.shape[1] == 12
    peak = flows.max()
    assert peak["Dhubri"] > peak["Guwahati"] > peak["Dibrugarh"]  # flow grows downstream
    assert peak["Silchar"] < 10_000  # the Barak is far smaller than the Brahmaputra


def test_thresholds_use_training_years_only(flows):
    """Changing data after the training cut-off must not change the flood level."""
    thr = F.thresholds(flows, 2020)
    tampered = flows.copy()
    tampered.loc[tampered.index.year > 2020] *= 10
    pd.testing.assert_series_equal(thr, F.thresholds(tampered, 2020))


def test_features_never_look_ahead(flows):
    """Features on a given day must be identical whether or not later days exist."""
    full = F.build_features(flows)
    cut = F.build_features(flows[flows.index <= "2022-06-15"])
    cols = F.feature_cols(full)
    a = full[full.date == "2022-06-15"].set_index("station_name")[cols].sort_index()
    b = cut[cut.date == "2022-06-15"].set_index("station_name")[cols].sort_index()
    pd.testing.assert_frame_equal(a, b)


def test_target_is_flood_within_next_h_days():
    idx = pd.date_range("2024-07-01", periods=10)
    flows = pd.DataFrame({"A": [1, 1, 1, 1, 9, 1, 1, 1, 1, 1]}, index=idx, dtype=float)
    df = F.add_targets(F.build_features(flows), pd.Series({"A": 5.0}))
    day0 = df[df.date == idx[0]].iloc[0]
    assert day0["y_3"] == 0 and day0["y_5"] == 1  # the flood on day 5 is 4 days after day 1


def test_live_models_match_feature_list():
    meta = json.loads((ROOT / "models/forecast_meta.json").read_text())
    import lightgbm as lgb

    for h in meta["horizons"]:
        # LightGBM stores spaces as underscores; columns are matched by position, so order is what matters
        expected = [c.replace(" ", "_") for c in meta["features"]]
        assert lgb.Booster(model_file=str(ROOT / f"models/forecast_h{h}.txt")).feature_name() == expected


def test_every_district_maps_to_a_station():
    m = district_station_map()
    assert len(m) == 33
    assert {m["Cachar"], m["Karimganj"], m["Hailakandi"]} == {"Silchar"}
    assert set(load_truth().district) <= set(m)


def test_district_history_excludes_test_year():
    t = pd.DataFrame({"district": ["X"] * 4, "month": [6, 6, 7, 7], "year": [2021, 2022, 2021, 2022], "y": [1, 1, 1, 1]})
    test = pd.DataFrame({"district": ["X"], "month": [6], "year": [2023], "y": [0]})
    _, te = add_history(t, test)
    assert te.hist_rate.iloc[0] > 0.5  # learned only from 2021-2022, where X always flooded


def test_score_counts():
    y = np.array([1, 1, 0, 0], bool)
    w = np.array([1, 0, 1, 0], bool)
    assert score(y, w) == {"hit_rate": 0.5, "false_alarm_ratio": 0.5, "csi": round(1 / 3, 3), "flooded": 2, "warned": 2}


def test_rain_features_never_look_ahead(flows):
    rain = F.load_rain()
    if rain is None:
        pytest.skip("rainfall not downloaded")
    full = F.build_features(flows, rain)
    cut = F.build_features(flows[flows.index <= "2022-06-15"], rain[rain.index <= "2022-06-15"])
    cols = [c for c in F.feature_cols(full) if c.startswith("rain_")]
    a = full[full.date == "2022-06-15"].set_index("station_name")[cols].sort_index()
    b = cut[cut.date == "2022-06-15"].set_index("station_name")[cols].sort_index()
    pd.testing.assert_frame_equal(a, b)
