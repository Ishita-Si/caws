import requests, pandas as pd

def fetch(start, end, out="uk_carbon.csv"):
    rows, bad = [], 0
    t = pd.Timestamp(start, tz="UTC"); end = pd.Timestamp(end, tz="UTC")
    while t < end:
        t2 = min(t + pd.Timedelta(days=13), end)
        url = f"http://api.energy-charts.info/v2"
        r = requests.get(url, timeout=30, headers={"Accept": "application/json"})
        js = r.json()
        data = js.get("data") if isinstance(js, dict) else None
        if not isinstance(data, list):
            print("Unexpected response for", url, "->", str(js)[:300]); t = t2; continue
        for d in data:
            inten = d.get("intensity", {})
            a, f = inten.get("actual"), inten.get("forecast")
            if a is None or f is None:
                bad += 1
                if bad <= 3: print("Record missing a value:", d)
                continue
            rows.append((d["from"], a, f))
        t = t2
    print("rows kept:", len(rows), "skipped:", bad)
    df = pd.DataFrame(rows, columns=["timestamp", "actual", "forecast"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates("timestamp").set_index("timestamp").resample("1h").mean().dropna()
    df.reset_index().to_csv(out, index=False)
    print("hourly rows:", len(df), "| days approx:", len(df) // 24)

fetch("2025-01-01", "2025-06-01")