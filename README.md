# Robust Carbon-Aware Workflow Scheduling

Schedule a deadline-bound workflow (a DAG of tasks) to minimise CO2 emissions when the hourly carbon forecast may be wrong by a bounded amount, with a cap on power per hour.
Uncertainty uses a Gamma budget: at most Gamma slots may reach their maximum forecast error at the same time. Gamma = 0 trusts the forecast.

## What is in this project

| Part | Files |
|---|---|
| Model, exact worst-case cost, independent checker, brute force, ASAP and HEFT-like baselines | `model.py` |
| Exact robust MILP (dual linear form, HiGHS via SciPy) | `milp.py` |
| Exact polynomial algorithm for a fixed-order chain on one processor (theta-threshold DP) | `chain.py` |
| NP-hardness reduction from 3-Partition and its check | `hardness.py` |
| RobSlack heuristic (threshold construction, local search, never worse than ASAP) | `heur.py` |
| Instance generators | `gen.py`, `exp_common.py` |
| Synthetic carbon generator and CSV loader | `data.py` |
| Real data download | `fetch_uk.py` (UK), `fetch_de.py` (Germany) |
| Dataset screening | `screen_dataset.py` |
| Tests | `test_all.py`, `test_heur.py` |
| Experiments | `run_experiments.py` (E1-E6), `run_e7.py` (short chains), `sens.py` (scaled forecast errors) |
| Analysis and figures | `analyze.py` |

All `.py` files live in one folder and import each other as flat modules. Run everything from that folder.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\activate
pip install -r requirements.txt    # numpy, scipy>=1.9, matplotlib, pandas, requests
```

SciPy 1.9 or newer is required for `scipy.optimize.milp`.

## Quick start: verify the code

```bash
python3 test_all.py     # worst-case formula, MILP vs brute force, chain DP, concavity in Gamma
python3 test_heur.py    # RobSlack gap against the exact MILP
python3 hardness.py     # 3-Partition reduction check
```

## Data

The experiments need a CSV with columns `timestamp,actual,forecast` (hourly, at least 100 consecutive days: 60 to calibrate, 40 to test).

**UK (used in the paper results).** National Grid Carbon Intensity API: free, no key, half-hourly actual and forecast values.
```bash
python3 fetch_uk.py     # writes uk_carbon.csv (1 Jan to 31 May 2025 in our run)
```

**Germany (optional).** Energy-Charts API, 2 requests per minute, so the download takes 30 to 40 minutes and is cached.
```bash
python3 fetch_de.py probe
python3 fetch_de.py     # writes de_carbon.csv
```
The German actual column is computed from generation by fuel with assumed emission factors (edit the `EF` table in `fetch_de.py` and cite your sources). The forecast column is a model built from the published wind, solar and load forecasts, not a published carbon forecast. State this if you use it.

**Screen a dataset before running experiments.**
```bash
python3 screen_dataset.py uk_carbon.csv
```
It prints the forecast error as a fraction of the average daily range. The UK data scores about 0.09. In our scaled-error tests robustness started to matter around 0.3 and above, so low-error grids are expected to show little difference between nominal and robust plans.

**Using a CSV in the experiments.** In `exp_common.py`, change `data()` to load your CSV with `load_csv` from `data.py`, calibrate on the first 60 days, and return the next 40 days as test data. By default the code uses the synthetic generator.

## Run the experiments

```bash
python3 run_experiments.py     # E1-E6, about 8 to 10 minutes, writes results.json
python3 run_e7.py              # short chains with the exact algorithm, writes e7.json
python3 sens.py                # forecast errors scaled by k = 1, 3, 6 (needs uk_carbon.csv)
python3 analyze.py             # writes summary.json, fig_slack.png, fig_cap.png, fig_price.png
```

| Experiment | What it measures |
|---|---|
| E1 | RobSlack and baselines against the proven MILP optimum on small instances |
| E2 | Savings against ASAP on 30-task workflows over held-out days (Gamma and slack sweep) |
| E3 | Effect of the power cap |
| E4 | Static plan versus rolling re-planning |
| E5 | Chain algorithm against the MILP, and its runtime |
| E6 | Price of robustness (exact, small instance) |
| E7 | Short chains solved exactly |
| `sens.py` | Real forecast errors scaled by k, to show when robustness pays |

Tuning options: raise `ILS = 10` in `run_experiments.py` to 30 for a better heuristic at longer runtime.

## Summary of results (real UK data, Jan to May 2025)

- The exact methods agree with brute force: closed form 200/200, MILP 58 instances, chain DP 59 instances and 20 against the MILP, 3-Partition reduction 200/200.
- RobSlack is within 2.9% of the exact optimum on average (median 1.5%, maximum 14.3%) on 40 small instances.
- Savings over ASAP on 30-task workflows reach 22.7%. Forecast-trusting greedy plans are worse than ASAP on 8 to 35% of days, up to 2.06 times ASAP's emissions.
- The UK forecast is accurate (mean absolute error 9 g/kWh), so robust and nominal-optimal plans are statistically indistinguishable on large workflows.
- Robustness helps for single tasks (worst day 1.63 times down to 1.32 times) and when errors are scaled up (at 6 times the errors, worst day 1.25 times down to 1.09 times on 30-task workflows).

These are results of this code on this data. The scaled-error test is a stress test built on real errors, not a separate dataset. The DAG sensitivity runs use 20 test days.

## Limitations

- One grid and five months of data for the main results.
- The forecast value for each period is a single number, so lead-time structure is not modelled.
- Fixed task-to-processor mapping, constant per-task power, uncertainty only in carbon intensity.
- RobSlack has no approximation ratio. It is guaranteed feasible and never worse than ASAP.
- The chain algorithm needs one processor and a fixed task order; the general problem is strongly NP-hard.
- The worst-case bound is a planning device, not a guarantee: actual emissions stayed under it on only 52.5% (Gamma = 0) to 94.6% (Gamma = 8) of cases.

## Paper

The IEEE-format paper is in `paper/` (`paper.tex`, `IEEEtran.cls`, figures). Build with `latexmk -pdf paper.tex`. Numbers in the paper come from `summary.json`, `e7.json` and the printed output of `sens.py`; update them by hand after any new run.

## Data attribution

UK data: National Grid ESO Carbon Intensity API. German data: energy-charts.info, Fraunhofer ISE, CC BY 4.0. Check each provider's current terms before redistributing data.
