
"""Build a Germany hourly carbon dataset (timestamp,actual,forecast) from the Energy-Charts API (CC BY 4.0, Fraunhofer ISE).
actual   = generation-weighted emission factor from /v2/public_power (15-min -> hourly)
forecast = learned map  residual_load -> carbon intensity, applied to the FORECAST residual load
           (load forecast - solar forecast - wind forecasts) from /v2/public_power_forecast.
The forecast column is therefore a MODEL built from published forecasts, not a published carbon forecast: say so in the paper.
Run:  python fetch_de.py probe      (1 request per forecast series: check they work)
      python fetch_de.py            (download ~160 days, ~30-40 min because of the 2 requests/min limit; results are cached)
"""
import os, sys, json, time, requests, numpy as np, pandas as pd
BASE = "http://api.energy-charts.info/v2"
COUNTRY, START, END = "de", "2025-01-01", "2025-06-09"        # ~160 days: 30 fit + 60 calibrate + 40 test + margin
CHUNK_DAYS, SLEEP, FIT_DAYS = 14, 32, 30
FORECAST_TYPE = "current"                                      # check the allowed values in http://api.energy-charts.info/v2
FORECASTS = {"solar": "solar", "wind_onshore": "wind_onshore", "wind_offshore": "wind_offshore"}  # name -> production_type (these work)
LOAD_TYPES = ["day-ahead", "intraday", "current"]               # forecast_type values tried for the load forecast; if none works, a weekday-hour load profile is used
# ASSUMED emission factors, g CO2eq/kWh. Replace with values you can cite (IPCC AR5/AR6 lifecycle, or UBA direct factors).
EF = {"fossil_brown_coal_lignite": 1030, "fossil_hard_coal": 820, "fossil_oil": 650, "fossil_coal_derived_gas": 700, "fossil_gas": 490,
      "biomass": 230, "waste": 300, "geothermal": 38, "hydro_run_of_river": 24, "hydro_water_reservoir": 24, "hydro_pumped_storage": 24,
      "others": 500, "wind_offshore": 12, "wind_onshore": 12, "solar": 45}
CACHE = "cache"; os.makedirs(CACHE, exist_ok=True)

def get(endpoint, params, tag):
    path = f"{CACHE}/{tag}.json"
    if os.path.exists(path): return json.load(open(path))
    while True:
        r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=90)
        if r.status_code == 429: time.sleep(float(r.headers.get("Retry-After", 40)) + 1); continue
        r.raise_for_status(); js = r.json(); json.dump(js, open(path, "w")); time.sleep(SLEEP); return js

def to_frame(js):
    df = pd.DataFrame([{"timestamp": d["timestamp"], **d["values"]} for d in js["data"]])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp")

def chunks(start, end):
    s = pd.Timestamp(start)
    while s <= pd.Timestamp(end):
        e = min(s + pd.Timedelta(days=CHUNK_DAYS - 1), pd.Timestamp(end))
        yield s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d"); s = e + pd.Timedelta(days=1)

def find_load_mode():
    path = f"{CACHE}/load_mode.json"
    if os.path.exists(path): return json.load(open(path))
    mode = {"mode": "profile"}
    for ft in LOAD_TYPES:
        try:
            get("public_power_forecast", dict(country=COUNTRY, production_type="load", forecast_type=ft, start=START, end=START), f"probe_load_{ft}")
            mode = {"mode": "api", "forecast_type": ft}; break
        except requests.HTTPError as ex:
            print("load forecast_type", ft, "failed:", ex.response.status_code); time.sleep(SLEEP)
    json.dump(mode, open(path, "w")); print("load forecast mode:", mode); return mode

def download():
    mode = find_load_mode()
    for s, e in chunks(START, END):
        get("public_power", dict(country=COUNTRY, start=s, end=e), f"pp_{s}")
        for name, ptype in FORECASTS.items():
            get("public_power_forecast", dict(country=COUNTRY, production_type=ptype, forecast_type=FORECAST_TYPE, start=s, end=e), f"fc_{name}_{s}")
        if mode["mode"] == "api":
            get("public_power_forecast", dict(country=COUNTRY, production_type="load", forecast_type=mode["forecast_type"], start=s, end=e), f"fc_load_{s}")
        print("downloaded", s, "to", e, flush=True)

def build(start=START, end=END, fit_days=FIT_DAYS, out="de_carbon.csv"):
    pp = pd.concat([to_frame(json.load(open(f"{CACHE}/pp_{s}.json"))) for s, _ in chunks(start, end)])
    pp = pp[~pp.index.duplicated()].sort_index()
    gen = pp[list(EF)].fillna(0).clip(lower=0)
    ok = pp["solar"].notna() & pp["wind_onshore"].notna()
    ci = ((gen * pd.Series(EF)).sum(axis=1) / gen.sum(axis=1))[ok]
    act = pd.DataFrame({"actual": ci, "resid_act": pp["residual_load"][ok], "load_act": pp["load"][ok]}).resample("1h").mean()
    fc = {}
    for name in FORECASTS:
        f = pd.concat([to_frame(json.load(open(f"{CACHE}/fc_{name}_{s}.json"))).iloc[:, 0] for s, _ in chunks(start, end)])
        fc[name] = f[~f.index.duplicated()].sort_index()
    F = pd.DataFrame(fc).resample("1h").mean()
    mode = find_load_mode() if os.path.exists(f"{CACHE}/load_mode.json") else {"mode": "profile"}
    if mode["mode"] == "api":
        fl = pd.concat([to_frame(json.load(open(f"{CACHE}/fc_load_{s}.json"))).iloc[:, 0] for s, _ in chunks(start, end)])
        F["load"] = fl[~fl.index.duplicated()].sort_index().resample("1h").mean()
    else:   # fallback: expected load = mean actual load by (weekday, hour) over the fit period only
        fitp = act.iloc[:fit_days * 24].dropna(subset=["load_act"])
        prof = fitp.groupby([fitp.index.dayofweek, fitp.index.hour])["load_act"].mean()
        F["load"] = [prof.get((t.dayofweek, t.hour), np.nan) for t in F.index]
        print("NOTE: using weekday-hour load profile instead of a load forecast")
    F["resid_fc"] = F["load"] - F["solar"] - F["wind_onshore"] - F["wind_offshore"]
    df = act.join(F[["resid_fc"]], how="inner").dropna()
    n_fit = fit_days * 24; fit = df.iloc[:n_fit]
    coef = np.polyfit(fit["resid_act"], fit["actual"], 2)         # fitted ONLY on the first `fit_days` days
    df["forecast"] = np.polyval(coef, df["resid_fc"])
    df = df.iloc[n_fit:][["actual", "forecast"]].dropna()
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="1h"))
    print("missing hourly rows:", int(df.actual.isna().sum()), "of", len(df)); df = df.dropna()
    df.index.name = "timestamp"; df.reset_index().to_csv(out, index=False)
    err = (df.actual - df.forecast).abs(); rng = df.groupby(df.index.date).actual.agg(lambda x: x.max() - x.min()).mean()
    print(f"saved {out}: {len(df)} rows (~{len(df)//24} days) | MAE {err.mean():.1f} | daily range {rng:.1f} | error/range {err.mean()/rng:.2f}")

def probe():
    for name, ptype in FORECASTS.items():
        js = get("public_power_forecast", dict(country=COUNTRY, production_type=ptype, forecast_type=FORECAST_TYPE, start=START, end=START), f"probe_{name}")
        print(name, "->", js["attributes"], "rows:", len(js["data"]))
    find_load_mode()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "probe": probe()
    elif len(sys.argv) > 1 and sys.argv[1] == "build": build()
    else: download(); build()

