# FloodWatch Assam

Flood forecasting and satellite flood mapping for Assam, India, tested on six real monsoons (2021–2026) against river data and against the Assam State Disaster Management Authority's (ASDMA) district flood reports.

**Live site:** https://samiyakazi23.github.io/floodwatch/ · today's forecast (updated every morning), a live river dashboard, and a date checker that compares any day's forecast from 2021 to 2026 with what actually happened.

Every year, Brahmaputra floods affect millions of people in Assam. FloodWatch asks three questions:

1. **Will the rivers flood in the next 1–7 days?** A gradient-boosted model reads daily flow at 12 river stations and rainfall at 15 points across the basin, and learns how surges travel down the valley.
2. **Will people actually be affected, and where?** An impact model turns river forecasts, local rainfall and district history into the chance that ASDMA reports people affected.
3. **Where is the water right now?** A U-Net maps floodwater from Sentinel-1 radar, which sees through monsoon cloud.

## Results

All numbers are out-of-sample. CSI (critical success index) = hits ÷ (hits + misses + false alarms), the standard score for flood warnings; 1 is perfect.

### River flood forecasts: trained on 1997–2020, run day by day through 2021–2026

Each test year uses only the years before it, and alert cut-offs are tuned on earlier years too. The deployed model uses river flow and rainfall.

| Lead time | Flood days caught | False alarms | CSI | CSI, river flow only | CSI if "tomorrow = today" |
|---|---|---|---|---|---|
| 1 day | 90% | 10% | **0.82** | 0.81 | 0.73 |
| 2 days | 80% | 12% | **0.72** | 0.73 | 0.65 |
| 3 days | 73% | 16% | **0.64** | 0.64 | 0.59 |
| 5 days | 65% | 31% | **0.51** | 0.51 | 0.49 |
| 7 days | 59% | 39% | **0.43** | 0.45 | 0.42 |

On river levels, rainfall barely changes the overall score. Its value shows up in early warning: floods warned three or more days before they started rose from 6% to 18% of the 62 onsets, and the daily check below shows the same effect on real impacts.

### Early warning, day by day: ASDMA's daily reports for the 2022 floods

ASDMA's daily district bulletins, 15 May – 20 October 2022, give 45 district flood onsets (a district's first day with 1,000+ people affected after a quiet week). Had the forecast for its river warned in advance?

| Method | 1 day before | 3 days before | 5 days before | False alarms on quiet days |
|---|---|---|---|---|
| **FloodWatch, river + rainfall** | **51%** | **42%** | **18%** | **0.9%** |
| FloodWatch, river flow only | 38% | 11% | 2% | 0.9% |
| Heavy local rain rule (top 10% of 3-day totals) | 78% | 71% | 58% | 15.6% |
| River already above 85% of warning level | 42% | 13% | 4% | — |

Rainfall nearly quadruples the floods warned three days ahead, with no rise in false alarms. A simple heavy-rain rule warns more often but raises a false alarm on one quiet day in six.

### Real impacts: ASDMA district reports, May–October 2021–2026

Unit: a district in a month (1,155 cases). "Flooded" = ASDMA reported at least 1,000 people affected (311 cases).

| Method | Floods warned | False alarms | CSI |
|---|---|---|---|
| **Impact model** (each year held out in turn) | 60% | 44% | **0.41** |
| River forecast alone | 42% | 44% | 0.32 |
| Flag every June and July | 55% | 57% | 0.32 |
| District flood history alone | 42% | 47% | 0.30 |
| River *observed* above warning level | 37% | 41% | 0.30 |

Two findings:

- **River levels miss much of the real damage.** Even knowing the river actually crossed its warning level scores only 0.30 against reported impacts. Much of Assam's flood damage comes from local rain, small tributaries and embankment breaches.
- **Forecasting impacts directly helps, once there is enough history.** ASDMA's district records start in 2021. Under a strict test where each year sees only earlier years, the impact model scores 0.13 in 2022 (one year of history), then 0.45, 0.41, 0.36 and 0.48 in 2023–2026. That is 0.42 for 2023–2026 against 0.35 for the calendar baseline. At monthly resolution rainfall adds little: a month's total is too coarse to say which week a district flooded.

### Satellite flood mapping: Sen1Floods11 held-out scenes

| | U-Net | Otsu threshold (operational baseline) |
|---|---|---|
| Water IoU, test set (10 countries) | **0.65** | 0.23 |
| Water IoU, Assam scenes | **0.72** | — |
| Water IoU, Bolivia (event never seen) | **0.63** | 0.40 |
| Pixel accuracy, test set | 95% | 70% |

## How it works

```
GloFAS river flow (12 stations)   NASA POWER rainfall (15 points)   1997–2026
          └──────────────┬──────────────────────┘
                         │  point-in-time features: own lags, every station's recent rise,
                         │  rain over the last 1/3/7/30 days at each point
                         ▼
LightGBM, one model per lead time (1, 2, 3, 5, 7 days)
   │  target: flow reaches warning level within the next h days
   │  warning level = 25th percentile of annual maxima (reached ~3 years in 4), training years only
   ▼
River forecast ──► impact model (logistic regression) ◄── district flood history (other years only)
                        │                              ◄── district rainfall that month
                        ▼
              P(ASDMA reports ≥ 1,000 people affected in this district this month)

Sentinel-1 VV/VH radar ──► U-Net (ResNet-18 encoder, trained from scratch) ──► flood mask
```

Leakage guards, each covered by a test in `tests/`:
- flood levels and alert cut-offs are computed from training years only;
- a day's river and rainfall features are identical whether or not later days exist;
- district history for a test year never includes that year.

## Run it

```bash
pip install -r requirements.txt
pytest -q                                      # leakage and sanity tests (seconds)
FW_RAIN=1 python -m floodwatch.forecast        # walk-forward backtest 2021–2026 with rainfall (~25 min, 2 CPU cores)
python -m floodwatch.forecast                  # same, river flow only
python -m floodwatch.validate_daily            # 2022 day-by-day early-warning check
FW_RAIN=1 python -m floodwatch.impact          # impact model: leave-one-year-out + strict walk-forward
FW_RAIN=1 python -m floodwatch.forecast --final   # retrain the live models on all data
python scripts/live_forecast.py                # today's forecast -> docs/data/live.json

# satellite model
pip install -r requirements-satellite.txt
./scripts/download_sen1floods11.sh             # ~700 MB
python -m floodwatch.segment --epochs 32       # ~45 min on CPU, minutes on a GPU
```

Data refresh: `scripts/find_gauges.py` (river flow) and `scripts/fetch_rain.py` (rainfall). The site is plain HTML and D3 in `docs/`; serve it with `python -m http.server -d docs`.

## Live forecasting

`.github/workflows/daily.yml` runs `scripts/live_forecast.py` every morning at 07:10 IST. The script:
1. fetches the latest river flow and rainfall (NASA POWER, with its last ~3 unpublished days filled from Open-Meteo);
2. forecasts 1–7 days ahead and updates the site;
3. appends the forecast to `docs/data/live_log.jsonl`, a timestamped record of forecasts issued *before* events, to be scored against ASDMA's reports after the 2027 monsoon.

## Limitations

- River flow comes from the GloFAS reanalysis, a model, not gauge readings, and the warning levels are statistical stand-ins for official danger levels.
- Rainfall is sampled at 15 points, so very local downpours can fall between them.
- ASDMA's district records go back only to 2021, are monthly, and daily district figures are available only for 2022.
- District-to-station matching is by nearest station, with a few manual corrections (Barak valley, Majuli, Biswanath).
- Not an official warning service: for alerts, follow ASDMA and the Central Water Commission.

## Data and credits

- River discharge: [GloFAS v4](https://www.globalfloods.eu/) via the [Open-Meteo Flood API](https://open-meteo.com/en/docs/flood-api) (CC BY 4.0).
- Rainfall: [NASA POWER](https://power.larc.nasa.gov/) daily PRECTOTCORR; recent days from the [Open-Meteo Forecast API](https://open-meteo.com/).
- Flood impacts: ASDMA's daily FRIMS reports, as compiled by [CivicDataLab's IDS-DRR project](https://github.com/CivicDataLab/IDS-DRR-Assam-Risk-Model). Only the district-level summaries derived from it are included here.
- Radar flood scenes: [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11), Bonafilia et al., CVPR Workshops 2020.
- District boundaries: [datta07/INDIAN-SHAPEFILES](https://github.com/datta07/INDIAN-SHAPEFILES) (Census 2011 districts).

Code: MIT licence. Built by [Samiya Kazi](https://github.com/samiyakazi23).
