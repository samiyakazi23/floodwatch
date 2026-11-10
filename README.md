# FloodWatch Assam

Flood forecasting and satellite flood mapping for Assam, India, tested on six real monsoons (2021–2026) against river data and against the Assam State Disaster Management Authority's district flood reports.

**Live site:** https://samiyakazi23.github.io/floodwatch/ · replay any monsoon from 2021 to 2026 district by district, or see today's forecast (updated every morning).

Every year, Brahmaputra floods affect millions of people in Assam. FloodWatch asks three questions:

1. **Will the rivers flood in the next 1–7 days?** A gradient-boosted model reads daily flow at 12 river stations and learns how surges travel down the valley.
2. **Will people actually be affected, and where?** An impact model turns river forecasts and district history into the chance that ASDMA reports people affected.
3. **Where is the water right now?** A U-Net maps floodwater from Sentinel-1 radar, which sees through monsoon cloud.

## Results

All numbers are out-of-sample. CSI (critical success index) = hits ÷ (hits + misses + false alarms), the standard score for flood warnings; 1 is perfect.

### River flood forecasts: trained on 1997–2020, run day by day through 2021–2026

Each test year uses only the years before it, and alert cut-offs are tuned on earlier years too.

| Lead time | Flood days caught | False alarms | CSI | CSI if "tomorrow = today" |
|---|---|---|---|---|
| 1 day | 91% | 12% | **0.81** | 0.73 |
| 2 days | 78% | 10% | **0.73** | 0.65 |
| 3 days | 75% | 19% | **0.64** | 0.59 |
| 5 days | 58% | 20% | **0.51** | 0.49 |
| 7 days | 54% | 27% | **0.45** | 0.42 |

It warned ahead of 58% of the 62 flood onsets, usually by one day. Only 6% got three or more days' notice; rainfall upstream is the planned fix.

### Real impacts: ASDMA district reports, May–October 2021–2026

Unit: a district in a month (1,155 cases). "Flooded" = ASDMA reported at least 1,000 people affected (311 cases).

| Method | Floods warned | False alarms | CSI |
|---|---|---|---|
| **Impact model** (each year held out in turn) | 63% | 46% | **0.41** |
| River forecast alone | 43% | 44% | 0.33 |
| Flag every June and July | 55% | 57% | 0.32 |
| District flood history alone | 42% | 47% | 0.30 |
| River *observed* above warning level | 37% | 41% | 0.30 |

Two findings:

- **River levels miss much of the real damage.** Even knowing the river actually crossed its warning level scores only 0.30 against reported impacts. Much of Assam's flood damage comes from local rain, small tributaries and embankment breaches.
- **Forecasting impacts directly helps, once there is enough history.** ASDMA's district records start in 2021. Under a strict test where each year sees only earlier years, the impact model scores 0.13 in 2022 (one year of history), then 0.47, 0.47, 0.29 and 0.41 in 2023–2026. That is 0.42 overall for 2023–2026 against 0.35 for the calendar baseline, and it wins three of those four years.

### Satellite flood mapping: Sen1Floods11 held-out scenes

| | U-Net | Otsu threshold (operational baseline) |
|---|---|---|
| Water IoU, test set (10 countries) | **0.65** | 0.23 |
| Water IoU, Assam scenes | **0.72** | — |
| Water IoU, Bolivia (event never seen) | **0.63** | 0.40 |
| Pixel accuracy, test set | 95% | 70% |

## How it works

```
GloFAS river flow (12 stations, 1997–2026)
   │  point-in-time features: own lags, every station's recent rise (upstream routing)
   ▼
LightGBM, one model per lead time (1, 2, 3, 5, 7 days)
   │  target: flow reaches warning level within the next h days
   │  warning level = 25th percentile of annual maxima (reached ~3 years in 4), training years only
   ▼
River forecast ──► impact model (logistic regression) ◄── district flood history (other years only)
                        │
                        ▼
              P(ASDMA reports ≥ 1,000 people affected in this district this month)

Sentinel-1 VV/VH radar ──► U-Net (ResNet-18 encoder, trained from scratch) ──► flood mask
```

Leakage guards, each covered by a test in `tests/`:
- flood levels and alert cut-offs are computed from training years only;
- a day's features are identical whether or not later days exist;
- district history for a test year never includes that year.

## Run it

```bash
pip install -r requirements.txt
pytest -q                                   # leakage and sanity tests (seconds)
python -m floodwatch.forecast               # walk-forward backtest 2021–2026 (~20 min on 2 CPU cores)
python -m floodwatch.validate_asdma         # river forecast vs ASDMA reports
python -m floodwatch.impact                 # impact model, leave-one-year-out
python scripts/live_forecast.py             # today's forecast -> docs/data/live.json

# satellite model
pip install -r requirements-satellite.txt
./scripts/download_sen1floods11.sh          # ~700 MB
python -m floodwatch.segment --epochs 32    # ~45 min on CPU, minutes on a GPU
```

The site is plain HTML and D3 in `docs/`; serve it with `python -m http.server -d docs`.

## Live forecasting

`.github/workflows/daily.yml` runs `scripts/live_forecast.py` every morning at 07:10 IST. The script fetches the latest river flow, forecasts 1–7 days ahead, updates the site, and appends the forecast to `docs/data/live_log.jsonl`. That log is a timestamped record of forecasts issued *before* events, to be scored against ASDMA's reports after the 2027 monsoon.

## Limitations

- River flow comes from the GloFAS reanalysis, a model, not gauge readings, and the warning levels are statistical stand-ins for official danger levels.
- Impact labels are monthly, so a warning anywhere in a month counts for the whole month. ASDMA's district records only go back to 2021.
- No rainfall yet. It is the next addition and should help most with local floods.
- District-to-station matching is by nearest station, with a few manual corrections (Barak valley, Majuli, Biswanath).
- Not an official warning service: for alerts, follow ASDMA and the Central Water Commission.

## Data and credits

- River discharge: [GloFAS v4](https://www.globalfloods.eu/) via the [Open-Meteo Flood API](https://open-meteo.com/en/docs/flood-api) (CC BY 4.0).
- Flood impacts: ASDMA daily FRIMS reports, as compiled by [CivicDataLab's IDS-DRR project](https://github.com/CivicDataLab/IDS-DRR-Assam-Risk-Model). Only the district-month summary derived from it is included here.
- Radar flood scenes: [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11), Bonafilia et al., CVPR Workshops 2020.
- District boundaries: [datta07/INDIAN-SHAPEFILES](https://github.com/datta07/INDIAN-SHAPEFILES) (Census 2011 districts).

Code: MIT licence. Built by [Samiya Kazi](https://github.com/samiyakazi23).
